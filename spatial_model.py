"""Pure geometry/data-prep for building a 3D geological model from a SpatialExtraction.

Everything here is deliberately free of GemPy/Plotly imports so the math is unit-
testable (tests/test_spatial_model.py) without the heavy modeling stack installed;
build_gempy_model.py holds the GemPy/Plotly glue.

Conventions:
- Azimuth: degrees clockwise from grid north. Dip: degrees from horizontal; holes
  are assumed to point DOWN — ``abs(dip)`` is used as the depression angle, so the
  -60 / 60 sign inconsistency across reports is harmless. Up-holes (underground
  drilling) are not supported in this POC.
- A contact between an overlying and an underlying unit is assigned to the
  UNDERLYING unit's surface (GemPy surfaces represent unit tops).
- Nothing is silently fabricated: every assumption (vertical hole with no survey,
  inferred stratigraphic order, plane-fit orientation) is recorded in the returned
  ``logs`` so the reviewer sees exactly what the model rests on.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Minimum-curvature desurvey
# ---------------------------------------------------------------------------


def _lerp_angle_deg(a: float, b: float, t: float) -> float:
    """Interpolate two angles along the shortest path (handles 350°→10°)."""
    d = ((b - a + 180.0) % 360.0) - 180.0
    return (a + d * t) % 360.0


def min_curvature_step(
    delta_md: float,
    az1_deg: float,
    dip1_deg: float,
    az2_deg: float,
    dip2_deg: float,
) -> Tuple[float, float, float]:
    """Displacement (dE, dN, dZ) over one survey segment via minimum curvature.

    Dips are depression angles below horizontal (sign ignored); dZ is negative
    downward. Reduces exactly to a straight-line step when the angles are equal.
    """
    # Inclination measured from vertical, as the standard formula expects.
    i1 = math.radians(90.0 - abs(dip1_deg))
    i2 = math.radians(90.0 - abs(dip2_deg))
    a1 = math.radians(az1_deg % 360.0)
    a2 = math.radians(az2_deg % 360.0)

    cos_dl = math.cos(i2 - i1) - math.sin(i1) * math.sin(i2) * (1.0 - math.cos(a2 - a1))
    cos_dl = max(-1.0, min(1.0, cos_dl))
    dl = math.acos(cos_dl)
    rf = 1.0 if dl < _EPS else (2.0 / dl) * math.tan(dl / 2.0)

    dn = 0.5 * delta_md * (math.sin(i1) * math.cos(a1) + math.sin(i2) * math.cos(a2)) * rf
    de = 0.5 * delta_md * (math.sin(i1) * math.sin(a1) + math.sin(i2) * math.sin(a2)) * rf
    dv = 0.5 * delta_md * (math.cos(i1) + math.cos(i2)) * rf  # vertical, down-positive
    return de, dn, -dv


@dataclass
class Station:
    md: float
    azimuth: float
    dip: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class HoleTrace:
    hole_id: str
    stations: List[Station] = field(default_factory=list)

    @property
    def total_md(self) -> float:
        return self.stations[-1].md if self.stations else 0.0

    def position_at(self, md: float) -> Tuple[float, float, float]:
        """XYZ at measured depth, min-curvature within the bracketing segment."""
        sts = self.stations
        if not sts:
            raise ValueError(f"hole {self.hole_id}: no stations")
        md = max(sts[0].md, min(md, sts[-1].md))
        for i in range(len(sts) - 1):
            s1, s2 = sts[i], sts[i + 1]
            if md > s2.md + _EPS:
                continue
            seg = s2.md - s1.md
            t = 0.0 if seg < _EPS else (md - s1.md) / seg
            az_md = _lerp_angle_deg(s1.azimuth, s2.azimuth, t)
            dip_md = s1.dip + (s2.dip - s1.dip) * t
            de, dn, dz = min_curvature_step(md - s1.md, s1.azimuth, s1.dip, az_md, dip_md)
            return s1.x + de, s1.y + dn, s1.z + dz
        last = sts[-1]
        return last.x, last.y, last.z


def desurvey_hole(borehole: dict, logs: List[str]) -> Optional[HoleTrace]:
    """Build a 3D trace for one borehole dict (as stored in SpatialExtraction JSON).

    Survey points are used where present; otherwise the collar azimuth/dip define
    a straight hole; if those are missing too, the hole is assumed vertical and
    the assumption is logged.
    """
    hole_id = (borehole.get("hole_id") or "").strip()
    x, y = borehole.get("x"), borehole.get("y")
    if not hole_id or x is None or y is None:
        logs.append(f"skipped hole '{hole_id or '?'}': missing collar coordinates")
        return None
    z = borehole.get("z_collar")
    if z is None:
        z = 0.0
        logs.append(f"hole {hole_id}: no collar elevation, default-assumed z=0")

    az, dip = borehole.get("azimuth"), borehole.get("dip")
    if dip is None:
        az, dip = 0.0, 90.0
        logs.append(f"hole {hole_id}: no collar dip, default-assumed vertical")
    elif az is None:
        if abs(abs(dip) - 90.0) > 1.0:
            logs.append(
                f"hole {hole_id}: inclined ({dip}°) but no azimuth — default-assumed vertical"
            )
        az, dip = 0.0, 90.0

    raw = [
        s for s in (borehole.get("survey_points") or [])
        if s.get("depth_m") is not None and s.get("dip") is not None
    ]
    raw.sort(key=lambda s: s["depth_m"])
    angle_sts: List[Station] = []
    if not raw or raw[0]["depth_m"] > _EPS:
        angle_sts.append(Station(md=0.0, azimuth=float(az), dip=float(dip)))
    for s in raw:
        angle_sts.append(
            Station(
                md=float(s["depth_m"]),
                azimuth=float(s["azimuth"]) if s.get("azimuth") is not None else angle_sts[-1].azimuth,
                dip=float(s["dip"]),
            )
        )

    total = borehole.get("total_depth_m")
    if total is not None and total > angle_sts[-1].md + _EPS:
        tail = angle_sts[-1]
        angle_sts.append(Station(md=float(total), azimuth=tail.azimuth, dip=tail.dip))

    angle_sts[0].x, angle_sts[0].y, angle_sts[0].z = float(x), float(y), float(z)
    for i in range(1, len(angle_sts)):
        s1, s2 = angle_sts[i - 1], angle_sts[i]
        de, dn, dz = min_curvature_step(s2.md - s1.md, s1.azimuth, s1.dip, s2.azimuth, s2.dip)
        s2.x, s2.y, s2.z = s1.x + de, s1.y + dn, s1.z + dz

    return HoleTrace(hole_id=hole_id, stations=angle_sts)


# ---------------------------------------------------------------------------
# Contact points from lithology intervals
# ---------------------------------------------------------------------------


def _norm_id(hole_id: Optional[str]) -> str:
    return (hole_id or "").strip().upper()


def _canon_id(hole_id: Optional[str]) -> str:
    """Canonical hole ID for cross-table joins: 'HX06-01' == 'HX-06-1' == 'hx 6 1'.

    Reports routinely punctuate/zero-pad hole IDs differently between the collar
    table and downhole tables; separators are dropped and leading zeros stripped
    from every digit run. Exact-normalized matches always take priority — this is
    only the fallback."""
    s = _norm_id(hole_id)
    parts = re.findall(r"\d+|[A-Z]+", s)
    return "".join(p.lstrip("0") or "0" if p.isdigit() else p for p in parts)


def build_contact_points(spatial: dict, logs: List[str]) -> Tuple[List[dict], Dict[str, HoleTrace]]:
    """Derive one surface point per unit change down each hole.

    Returns (contact_points, traces). Each point: {x, y, z, surface, hole_id, source}
    where `surface` is the UNDERLYING unit (its top).
    """
    traces: Dict[str, HoleTrace] = {}
    canon_traces: Dict[str, HoleTrace] = {}
    for bh in spatial.get("boreholes") or []:
        trace = desurvey_hole(bh, logs)
        if trace:
            traces[_norm_id(trace.hole_id)] = trace
            canon_traces.setdefault(_canon_id(trace.hole_id), trace)

    by_hole: Dict[str, List[dict]] = {}
    for iv in spatial.get("lithology_intervals") or []:
        if iv.get("from_m") is None or not iv.get("unit_name"):
            continue
        by_hole.setdefault(_norm_id(iv.get("hole_id")), []).append(iv)

    points: List[dict] = []
    missing_holes: set = set()
    canon_matched: set = set()
    for hole_key, intervals in by_hole.items():
        trace = traces.get(hole_key)
        if trace is None:
            trace = canon_traces.get(_canon_id(hole_key))
            if trace is not None:
                canon_matched.add(f"{hole_key} -> {trace.hole_id}")
        if trace is None:
            missing_holes.add(hole_key or "?")
            continue
        intervals.sort(key=lambda iv: iv["from_m"])
        for prev, cur in zip(intervals, intervals[1:]):
            u1 = (prev.get("unit_name") or "").strip()
            u2 = (cur.get("unit_name") or "").strip()
            if not u1 or not u2 or u1.lower() == u2.lower():
                continue
            boundary_md = cur["from_m"]
            px, py, pz = trace.position_at(float(boundary_md))
            points.append(
                {
                    "x": px,
                    "y": py,
                    "z": pz,
                    "surface": u2,
                    "hole_id": trace.hole_id,
                    "source": f"{trace.hole_id} @ {boundary_md} m ({u1}/{u2} contact)",
                }
            )
    if canon_matched:
        logs.append(
            f"{len(canon_matched)} hole ID(s) joined via canonical matching "
            f"(punctuation/zero-padding differs between tables): "
            + ", ".join(sorted(canon_matched)[:8])
        )
    if missing_holes:
        logs.append(
            f"lithology for {len(missing_holes)} hole(s) with no collar in the extraction, "
            f"skipped: {', '.join(sorted(missing_holes)[:8])}"
        )
    return points, traces


def build_envelope_points(
    spatial: dict,
    traces: Dict[str, HoleTrace],
    envelope_units: List[str],
    logs: List[str],
) -> List[dict]:
    """OPT-IN interpretation: treat a unit's intervals as a body envelope.

    NI 43-101 reports rarely publish full lithology logs, but intercept tables
    give from/to depths of a named body (e.g. 'Porphyry') per hole. With this
    flag the caller asserts those intervals delimit the body: each interval's
    from_m becomes a point on '<Unit> (top)' and its to_m a point on
    '<Unit> (base)'. Always logged — this is an interpretation, not disclosure.
    """
    canon_traces: Dict[str, HoleTrace] = {}
    for tr in traces.values():
        canon_traces.setdefault(_canon_id(tr.hole_id), tr)
    wanted = {u.strip().lower() for u in envelope_units if u.strip()}
    points: List[dict] = []
    used_holes: set = set()
    for iv in spatial.get("lithology_intervals") or []:
        unit = (iv.get("unit_name") or "").strip()
        if unit.lower() not in wanted or iv.get("from_m") is None:
            continue
        trace = traces.get(_norm_id(iv.get("hole_id"))) or canon_traces.get(_canon_id(iv.get("hole_id")))
        if trace is None:
            continue
        used_holes.add(trace.hole_id)
        src = iv.get("source") or "interval"
        px, py, pz = trace.position_at(float(iv["from_m"]))
        points.append(
            {
                "x": px, "y": py, "z": pz,
                "surface": f"{unit} (top)", "hole_id": trace.hole_id,
                "source": f"envelope interpretation: {trace.hole_id} @ {iv['from_m']} m ({src})",
            }
        )
        if iv.get("to_m") is not None:
            px, py, pz = trace.position_at(float(iv["to_m"]))
            points.append(
                {
                    "x": px, "y": py, "z": pz,
                    "surface": f"{unit} (base)", "hole_id": trace.hole_id,
                    "source": f"envelope interpretation: {trace.hole_id} @ {iv['to_m']} m ({src})",
                }
            )
    if points:
        logs.append(
            f"ENVELOPE INTERPRETATION (--envelope): {len(points)} points on "
            f"{'/'.join(sorted(wanted))} top/base from intervals in {len(used_holes)} hole(s) — "
            "intercept boundaries treated as body contacts; a geologist must judge whether "
            "that approximation holds for this deposit."
        )
    return points


# ---------------------------------------------------------------------------
# Plane-fit orientation (data-derived, never invented)
# ---------------------------------------------------------------------------


def fit_plane_orientation(points: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
    """Least-squares plane through ≥3 points → (dip°, dip_direction°), or None.

    Returns None for degenerate inputs (collinear points, near-vertical fits are
    still returned — the caller decides whether to trust steep fits).
    """
    if len(points) < 3:
        return None
    import numpy as np

    arr = np.asarray(points, dtype=float)
    centered = arr - arr.mean(axis=0)
    # Normal = singular vector of the smallest singular value.
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s[1] < _EPS:  # rank < 2: points are collinear — no unique plane
        return None
    normal = vt[-1]
    if abs(normal[2]) < _EPS:
        return None  # vertical plane: dip direction defined, but unstable — reject
    if normal[2] < 0:
        normal = -normal
    nx, ny, nz = (normal / np.linalg.norm(normal)).tolist()
    dip = math.degrees(math.acos(max(-1.0, min(1.0, nz))))
    dip_dir = math.degrees(math.atan2(nx, ny)) % 360.0 if dip > _EPS else 0.0
    return dip, dip_dir


# ---------------------------------------------------------------------------
# Model input assembly
# ---------------------------------------------------------------------------


def _series_mapping(spatial: dict, surfaces_with_points: List[str], logs: List[str],
                    surface_mean_z: Dict[str, float]) -> Dict[str, List[str]]:
    """Order surfaces into series from the stratigraphic pile, youngest first.

    Surfaces seen in lithology but absent from the pile are appended in
    mean-elevation order (highest = youngest) — data-derived, and logged.
    """
    pile = [u for u in (spatial.get("stratigraphic_pile") or []) if u.get("unit_name")]
    pile.sort(key=lambda u: (u.get("order_top_down") is None, u.get("order_top_down") or 0))

    known = {u["unit_name"].strip().lower() for u in pile}
    mapping: Dict[str, List[str]] = {}
    surf_lookup = {s.lower(): s for s in surfaces_with_points}

    for u in pile:
        name = u["unit_name"].strip()
        actual = surf_lookup.get(name.lower())
        if actual is None:
            logs.append(f"pile unit '{name}' has no contact points — not modeled")
            continue
        series = (u.get("series_name") or "Strat_Series").strip() or "Strat_Series"
        mapping.setdefault(series, []).append(actual)

    orphans = [s for s in surfaces_with_points if s.lower() not in known]
    if orphans:
        orphans.sort(key=lambda s: -surface_mean_z.get(s, 0.0))
        mapping.setdefault("Strat_Series", []).extend(orphans)
        logs.append(
            "units not in stratigraphic pile, order inferred from mean contact elevation "
            f"(youngest first): {', '.join(orphans)}"
        )
    return mapping


def assemble_model_inputs(spatial: dict, envelope_units: Optional[List[str]] = None) -> dict:
    """Turn a SpatialExtraction dict into GemPy-ready inputs.

    Returns {surface_points, orientations, series_mapping, extent, hole_traces,
    faults_skipped, logs}. Raises ValueError when there is not enough data to
    model at all. `envelope_units` opts in to the intercept-envelope
    interpretation (see build_envelope_points).
    """
    logs: List[str] = []
    contact_points, traces = build_contact_points(spatial, logs)
    if envelope_units:
        contact_points.extend(build_envelope_points(spatial, traces, envelope_units, logs))

    # Manually digitized cross-section points join the same pool.
    for p in spatial.get("cross_section_points") or []:
        if None in (p.get("x"), p.get("y"), p.get("z")) or not p.get("surface_name"):
            continue
        contact_points.append(
            {
                "x": p["x"], "y": p["y"], "z": p["z"],
                "surface": p["surface_name"].strip(),
                "hole_id": None,
                "source": p.get("source") or "digitized cross-section",
            }
        )

    by_surface: Dict[str, List[dict]] = {}
    for p in contact_points:
        by_surface.setdefault(p["surface"], []).append(p)
    dropped = {s: pts for s, pts in by_surface.items() if len(pts) < 2}
    for s, pts in dropped.items():
        logs.append(f"surface '{s}' has only {len(pts)} point(s) (<2) — not modeled")
    by_surface = {s: pts for s, pts in by_surface.items() if len(pts) >= 2}
    if not by_surface:
        raise ValueError(
            "No surface has >=2 contact points — nothing to model. "
            "Check lithology_intervals/hole_id matching in the extraction."
        )

    surface_mean_z = {
        s: sum(p["z"] for p in pts) / len(pts) for s, pts in by_surface.items()
    }
    mapping = _series_mapping(spatial, list(by_surface), logs, surface_mean_z)
    modeled = {s for names in mapping.values() for s in names}
    surface_points = [p for s, pts in by_surface.items() if s in modeled for p in pts]

    # Orientations: extracted ones with full coordinates first.
    orientations: List[dict] = []
    for o in spatial.get("orientations") or []:
        name = (o.get("surface_name") or "").strip()
        target = next((s for s in modeled if s.lower() == name.lower()), None)
        if target is None:
            if name:
                logs.append(f"orientation for '{name}' matches no modeled surface — skipped")
            continue
        if None in (o.get("x"), o.get("y"), o.get("z"), o.get("dip"), o.get("dip_direction")):
            logs.append(
                f"orientation for '{target}' lacks coordinates or dip/dip_direction — skipped "
                f"(source: {o.get('source') or '?'})"
            )
            continue
        orientations.append(
            {
                "x": o["x"], "y": o["y"], "z": o["z"],
                "dip": o["dip"], "dip_direction": o["dip_direction"],
                "surface": target, "polarity": 1,
                "source": o.get("source") or "extracted",
            }
        )

    # Every series needs ≥1 orientation: fall back to a plane fit through the
    # series' best-populated surface. Data-derived, and logged.
    covered = {o["surface"] for o in orientations}
    for series, names in mapping.items():
        if any(s in covered for s in names):
            continue
        for s in sorted(names, key=lambda n: -len(by_surface.get(n, []))):
            pts = by_surface.get(s, [])
            fit = fit_plane_orientation([(p["x"], p["y"], p["z"]) for p in pts])
            if fit is None:
                continue
            dip, dip_dir = fit
            cx = sum(p["x"] for p in pts) / len(pts)
            cy = sum(p["y"] for p in pts) / len(pts)
            cz = sum(p["z"] for p in pts) / len(pts)
            orientations.append(
                {
                    "x": cx, "y": cy, "z": cz,
                    "dip": round(dip, 2), "dip_direction": round(dip_dir, 2),
                    "surface": s, "polarity": 1,
                    "source": f"derived: plane fit through {len(pts)} contact points",
                }
            )
            logs.append(
                f"series '{series}': no extracted orientation — plane-fit on '{s}' "
                f"gives dip {dip:.1f}° toward {dip_dir:.0f}°"
            )
            break
        else:
            raise ValueError(
                f"series '{series}' has no orientation and no surface admits a plane fit. "
                "Supply one via --default-orientation DIP,DIPDIR[,SURFACE]."
            )

    xs = [p["x"] for p in surface_points]
    ys = [p["y"] for p in surface_points]
    zs = [p["z"] for p in surface_points]
    for tr in traces.values():
        for st in tr.stations:
            xs.append(st.x); ys.append(st.y); zs.append(st.z)
    pad_xy = max((max(xs) - min(xs)), (max(ys) - min(ys)), 100.0) * 0.15
    pad_z = max(max(zs) - min(zs), 50.0) * 0.25
    extent = [
        min(xs) - pad_xy, max(xs) + pad_xy,
        min(ys) - pad_xy, max(ys) + pad_xy,
        min(zs) - pad_z, max(zs) + pad_z,
    ]

    faults = spatial.get("faults") or []
    if faults:
        logs.append(
            f"{len(faults)} fault(s) recorded in the extraction are NOT modeled "
            "(fault surface construction is out of POC scope): "
            + ", ".join((f.get("fault_name") or "?") for f in faults)
        )

    return {
        "surface_points": surface_points,
        "orientations": orientations,
        "series_mapping": mapping,
        "extent": extent,
        "hole_traces": {
            hid: [(s.x, s.y, s.z) for s in tr.stations] for hid, tr in traces.items()
        },
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# Confirmed-gate loader
# ---------------------------------------------------------------------------


def load_confirmed_spatial(
    spatial_dir: Path,
    filename: str,
    allow_unconfirmed: bool = False,
) -> dict:
    """Load spatial_data/{stem}.json, refusing unreviewed extractions."""
    stem = Path(filename).stem
    path = Path(spatial_dir) / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No spatial extraction at {path}. Run: python rag_app.py extract --spatial --file {filename}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("confirmed") and not allow_unconfirmed:
        raise ValueError(
            f"{path} has confirmed=false — a human must review the extraction against the "
            "source PDF and set confirmed=true before a model is built from it. "
            "(--allow-unconfirmed overrides for development only.)"
        )
    return data
