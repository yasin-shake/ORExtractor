"""Build an interactive 3D geological model from a confirmed spatial extraction.

Usage:
    python build_gempy_model.py <report.pdf|stem> [--resolution 60] [--out model.html]
                                [--default-orientation DIP,DIPDIR[,SURFACE]]
                                [--allow-unconfirmed]

Reads spatial_data/{stem}.json (refusing confirmed=false unless overridden),
desurveys the holes, derives contact points and orientations (spatial_model.py),
interpolates with GemPy, and writes a self-contained Plotly HTML: interpolated
surfaces + borehole traces + contact points, each with source provenance in the
hover text. A {stem}_model_meta.json sits next to it with every assumption the
model rests on.

GemPy/Plotly are NOT in requirements.txt (they stay out of the API container,
like marker-pdf): pip install -r requirements-spatial.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from spatial_model import assemble_model_inputs, load_confirmed_spatial

_SURFACE_POINT_COLS = ["X", "Y", "Z", "surface"]
_ORIENTATION_COLS = ["X", "Y", "Z", "azimuth", "dip", "polarity", "surface"]


def _write_gempy_csvs(inputs: dict, tmp: Path) -> tuple:
    sp_path = tmp / "surface_points.csv"
    or_path = tmp / "orientations.csv"
    with sp_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_SURFACE_POINT_COLS)
        for p in inputs["surface_points"]:
            w.writerow([p["x"], p["y"], p["z"], p["surface"]])
    with or_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_ORIENTATION_COLS)
        for o in inputs["orientations"]:
            # GemPy's CSV importer calls the dip-direction column 'azimuth'.
            w.writerow([o["x"], o["y"], o["z"], o["dip_direction"], o["dip"], o["polarity"], o["surface"]])
    return sp_path, or_path


def build_model(inputs: dict, project_name: str, resolution: int):
    """Interpolate with GemPy and return the computed geo_model."""
    import gempy as gp

    with tempfile.TemporaryDirectory() as tmpdir:
        sp_path, or_path = _write_gempy_csvs(inputs, Path(tmpdir))
        geo_model = gp.create_geomodel(
            project_name=project_name,
            extent=list(inputs["extent"]),
            resolution=[resolution] * 3,
            importer_helper=gp.data.ImporterHelper(
                path_to_surface_points=str(sp_path),
                path_to_orientations=str(or_path),
            ),
        )
    gp.map_stack_to_surfaces(geo_model, {k: tuple(v) for k, v in inputs["series_mapping"].items()})
    gp.compute_model(geo_model)
    return geo_model


def _mesh_traces(geo_model, opacity: float) -> List[object]:
    import plotly.graph_objects as go

    traces: List[object] = []
    for element in geo_model.structural_frame.structural_elements:
        v = getattr(element, "vertices", None)
        e = getattr(element, "edges", None)
        if v is None or e is None or not len(v) or not len(e):
            continue
        traces.append(
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=e[:, 0], j=e[:, 1], k=e[:, 2],
                name=element.name,
                color=element.color,
                opacity=opacity,
                showlegend=True,
                hovertemplate=f"{element.name}<extra></extra>",
            )
        )
    return traces


def build_figure(geo_model, inputs: dict, title: str):
    """Plotly figure: interpolated surfaces + borehole traces + contact points."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for tr in _mesh_traces(geo_model, opacity=0.55):
        fig.add_trace(tr)

    first_hole = True
    for hole_id, pts in sorted(inputs["hole_traces"].items()):
        if len(pts) < 2:
            continue
        xs, ys, zs = zip(*pts)
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line={"color": "#444", "width": 3},
                name="boreholes",
                legendgroup="boreholes",
                showlegend=first_hole,
                hovertemplate=f"{hole_id}<extra></extra>",
            )
        )
        first_hole = False

    pts = inputs["surface_points"]
    if pts:
        fig.add_trace(
            go.Scatter3d(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                z=[p["z"] for p in pts],
                mode="markers",
                marker={"size": 3, "color": "#111", "symbol": "circle"},
                name="contact points",
                text=[f"{p['surface']}<br>{p.get('source') or ''}" for p in pts],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        scene={
            "aspectmode": "data",
            "xaxis_title": "Easting",
            "yaxis_title": "Northing",
            "zaxis_title": "Elevation",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def _parse_default_orientation(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) not in (2, 3):
        raise SystemExit("--default-orientation expects DIP,DIPDIR or DIP,DIPDIR,SURFACE")
    return {
        "dip": float(parts[0]),
        "dip_direction": float(parts[1]),
        "surface": parts[2] if len(parts) == 3 else None,
    }


def _apply_default_orientation(spatial: dict, default: dict) -> None:
    """Inject a user-supplied orientation assumption before assembly.

    The user is the source here — an explicit, recorded assumption, not a
    fabrication. It is anchored at the collar centroid and tagged
    'user-assumption via --default-orientation' so it survives into the
    model meta file.
    """
    boreholes = spatial.get("boreholes") or []
    xs = [b["x"] for b in boreholes if b.get("x") is not None]
    ys = [b["y"] for b in boreholes if b.get("y") is not None]
    zs = [b.get("z_collar") or 0.0 for b in boreholes if b.get("x") is not None]
    if not xs:
        raise SystemExit("--default-orientation needs at least one collar to anchor to.")
    surface = default["surface"]
    if surface is None:
        pile = spatial.get("stratigraphic_pile") or []
        named = [u.get("unit_name") for u in pile if u.get("unit_name")]
        units = {(iv.get("unit_name") or "").strip() for iv in spatial.get("lithology_intervals") or []}
        surface = named[0] if named else (sorted(u for u in units if u) or [None])[0]
        if surface is None:
            raise SystemExit("--default-orientation: no surface name available; pass DIP,DIPDIR,SURFACE.")
    spatial.setdefault("orientations", []).append(
        {
            "surface_name": surface,
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "z": sum(zs) / len(zs),
            "dip": default["dip"],
            "dip_direction": default["dip_direction"],
            "polarity": 1,
            "source": "user-assumption via --default-orientation",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 3D geological model from a confirmed spatial extraction.")
    parser.add_argument("filename", help="Source PDF filename or stem (matches spatial_data/{stem}.json).")
    parser.add_argument("--resolution", type=int, default=60, help="Regular-grid resolution per axis (default 60).")
    parser.add_argument("--out", default=None, help="Output HTML path (default spatial_data/{stem}_model.html).")
    parser.add_argument(
        "--default-orientation",
        default=None,
        metavar="DIP,DIPDIR[,SURFACE]",
        help="Explicit orientation assumption when the report gives none (recorded as user-assumption).",
    )
    parser.add_argument(
        "--envelope",
        default=None,
        metavar="UNIT[,UNIT...]",
        help=(
            "Opt-in interpretation: treat the named unit's intervals (e.g. intercept tables) "
            "as body top/base contacts. Use when the report has no lithology logs."
        ),
    )
    parser.add_argument(
        "--allow-unconfirmed",
        action="store_true",
        help="DEVELOPMENT ONLY: build from an extraction no human has reviewed.",
    )
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv
    load_dotenv()
    spatial_dir = Path(os.getenv("RAG_SPATIAL_DIR", "spatial_data"))

    spatial = load_confirmed_spatial(spatial_dir, args.filename, allow_unconfirmed=args.allow_unconfirmed)
    if args.allow_unconfirmed and not spatial.get("confirmed"):
        print("WARNING: building from an UNCONFIRMED extraction — do not circulate this model.")

    default = _parse_default_orientation(args.default_orientation)
    if default:
        _apply_default_orientation(spatial, default)

    envelope_units = [u.strip() for u in args.envelope.split(",")] if args.envelope else None
    inputs = assemble_model_inputs(spatial, envelope_units=envelope_units)
    stem = Path(args.filename).stem
    for line in inputs["logs"]:
        print(f"  note: {line}")
    print(
        f"Model inputs: {len(inputs['surface_points'])} contact points across "
        f"{sum(len(v) for v in inputs['series_mapping'].values())} surfaces, "
        f"{len(inputs['orientations'])} orientations, {len(inputs['hole_traces'])} holes."
    )

    geo_model = build_model(inputs, project_name=stem, resolution=args.resolution)

    out_html = Path(args.out) if args.out else spatial_dir / f"{stem}_model.html"
    fig = build_figure(geo_model, inputs, title=stem)
    fig.write_html(str(out_html), include_plotlyjs=True)

    meta = {
        "source_file": spatial.get("source_file"),
        "confirmed": bool(spatial.get("confirmed")),
        "coordinate_system": spatial.get("coordinate_system"),
        "resolution": args.resolution,
        "extent": inputs["extent"],
        "series_mapping": inputs["series_mapping"],
        "orientation_sources": [o["source"] for o in inputs["orientations"]],
        "assumptions_and_notes": inputs["logs"],
        "counts": {
            "surface_points": len(inputs["surface_points"]),
            "orientations": len(inputs["orientations"]),
            "holes": len(inputs["hole_traces"]),
        },
    }
    # Meta sits next to the HTML it describes — per-zone builds via --out must
    # not overwrite each other's assumption records.
    meta_path = out_html.with_name(out_html.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Model written -> {out_html}")
    print(f"Assumptions/meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
