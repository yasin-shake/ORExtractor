"""Deterministic chart/diagram validation and reconstruction."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from ingestion.cache import visual_model_signature
from ingestion.models import (
    SUPPORTED_CHART_TYPES,
    SUPPORTED_DIAGRAM_TYPES,
    UNSUPPORTED_RECONSTRUCTION_TYPES,
    ChartSpecification,
    DiagramSpecification,
    ElementRecord,
    PIPELINE_VERSION,
    VISUAL_PROMPT_VERSION,
    VISUAL_SCHEMA_VERSION,
    VisualAnalysis,
)


def normalize_figure_type(figure_type: str) -> str:
    return (figure_type or "unknown").strip().lower().replace(" ", "_").replace("-", "_")


_UNSUPPORTED_CONTENT_MARKERS = (
    "geological_map",
    "geologic_map",
    "geological_cross_section",
    "geologic_cross_section",
    "mine_plan",
    "drill_hole_map",
    "pit_shell",
    "resource_block_model",
    "block_model",
    "contour_map",
    "3d_geological",
    "three_dimensional_geological",
)


def reconstruction_allowed(analysis: VisualAnalysis, settings) -> Tuple[bool, str]:
    ftype = normalize_figure_type(analysis.figure_type)
    if ftype in UNSUPPORTED_RECONSTRUCTION_TYPES:
        return False, "unsupported_category"
    safety_text = normalize_figure_type(
        " ".join(
            (
                analysis.caption,
                analysis.description,
                *analysis.labels,
            )
        )
    )
    if any(
        marker in safety_text
        for marker in _UNSUPPORTED_CONTENT_MARKERS
    ):
        return False, "unsupported_category"
    if not analysis.reconstruction_supported:
        return False, "model_rejected"
    threshold = float(getattr(settings, "bedrock_visual_confidence_threshold", 0.85))
    if analysis.confidence < threshold:
        return False, "low_confidence"
    if ftype in SUPPORTED_CHART_TYPES:
        if not getattr(settings, "visual_reconstruct_charts", True):
            return False, "charts_disabled"
        if analysis.values_are_estimated:
            # Allow but warn — caller records values_estimated
            pass
        if not analysis.chart or not analysis.chart.series:
            return False, "missing_chart_data"
        if analysis.reconstruction_method not in {"plotly", ""}:
            return False, "invalid_reconstruction_method"
        return True, "chart"
    if ftype in SUPPORTED_DIAGRAM_TYPES:
        if not getattr(settings, "visual_reconstruct_diagrams", True):
            return False, "diagrams_disabled"
        if not analysis.diagram or not analysis.diagram.nodes:
            return False, "missing_diagram_data"
        if analysis.reconstruction_method not in {"graphviz", ""}:
            return False, "invalid_reconstruction_method"
        return True, "diagram"
    return False, "unsupported_category"


def validate_chart(chart: ChartSpecification) -> List[str]:
    warnings: List[str] = []
    if not chart.series:
        warnings.append("no_series")
        return warnings
    if (
        chart.expected_series_count is not None
        and chart.expected_series_count != len(chart.series)
    ):
        warnings.append(
            f"series_count_mismatch:{chart.expected_series_count}!={len(chart.series)}"
        )
    for axis, lower, upper in (
        ("x", chart.x_min, chart.x_max),
        ("y", chart.y_min, chart.y_max),
    ):
        if lower is not None and upper is not None and lower > upper:
            warnings.append(f"invalid_{axis}_range:{lower}>{upper}")
    for series in chart.series:
        if not series.points:
            warnings.append(f"empty_series:{series.name or '?'}")
            continue
        ys = [p.y for p in series.points if p.y is not None]
        if not ys:
            warnings.append(f"missing_y:{series.name or '?'}")
            continue
        for point in series.points:
            if point.y is None or not math.isfinite(point.y):
                warnings.append(f"invalid_y:{series.name or '?'}")
                continue
            if chart.y_min is not None and point.y < chart.y_min:
                warnings.append(f"y_below_axis:{series.name or '?'}:{point.y}")
            if chart.y_max is not None and point.y > chart.y_max:
                warnings.append(f"y_above_axis:{series.name or '?'}:{point.y}")
            if isinstance(point.x, float) and not math.isfinite(point.x):
                warnings.append(f"invalid_x:{series.name or '?'}")
            if isinstance(point.x, (int, float)):
                if chart.x_min is not None and point.x < chart.x_min:
                    warnings.append(f"x_below_axis:{series.name or '?'}:{point.x}")
                if chart.x_max is not None and point.x > chart.x_max:
                    warnings.append(f"x_above_axis:{series.name or '?'}:{point.x}")
        if chart.chart_type == "pie" and any(y < 0 for y in ys):
            warnings.append(f"negative_pie_value:{series.name or '?'}")
    return warnings


def validate_diagram(diagram: DiagramSpecification) -> List[str]:
    warnings: List[str] = []
    ids = [node.id for node in diagram.nodes]
    node_ids = set(ids)
    if not node_ids:
        warnings.append("no_nodes")
    if len(ids) != len(node_ids):
        warnings.append("duplicate_node_ids")
    for edge in diagram.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            warnings.append(f"dangling_edge:{edge.source}->{edge.target}")
    return warnings


def render_chart(chart: ChartSpecification, out_path: Path) -> Optional[Path]:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    chart_type = (chart.chart_type or "line").lower()
    fig = go.Figure()
    for series in chart.series:
        xs = [p.x if p.x is not None else (p.label or "") for p in series.points]
        ys = [p.y for p in series.points]
        name = series.name or "series"
        if chart_type == "bar":
            fig.add_trace(go.Bar(x=xs, y=ys, name=name))
        elif chart_type == "scatter":
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=name))
        elif chart_type == "pie":
            fig = go.Figure(data=[go.Pie(labels=xs, values=ys, name=name)])
            break
        else:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name))

    fig.update_layout(
        title=chart.title or "",
        xaxis_title=f"{chart.x_label} {chart.x_unit}".strip(),
        yaxis_title=f"{chart.y_label} {chart.y_unit}".strip(),
        template="plotly_white",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(out_path))
    except Exception:
        # Fallback to HTML if kaleido unavailable
        html_path = out_path.with_suffix(".html")
        fig.write_html(str(html_path))
        return html_path
    return out_path


def render_diagram(diagram: DiagramSpecification, out_path: Path) -> Optional[Path]:
    try:
        from graphviz import Digraph
    except ImportError:
        return None

    g = Digraph(comment=diagram.title or "diagram", format="png")
    g.attr(rankdir="LR")
    for node in diagram.nodes:
        g.node(node.id, node.label or node.id)
    for edge in diagram.edges:
        g.edge(edge.source, edge.target, label=edge.label or "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered = g.render(filename=out_path.stem, directory=str(out_path.parent), cleanup=True)
        return Path(rendered)
    except Exception:
        # Write DOT source for audit even if graphviz binary missing
        dot_path = out_path.with_suffix(".dot")
        dot_path.write_text(g.source, encoding="utf-8")
        return dot_path


def reconstruct_visuals(
    elements: List[ElementRecord],
    analyses: dict[str, VisualAnalysis],
    settings,
    artifact_dir: Path,
) -> Tuple[dict[str, dict], List[str]]:
    """
    Reconstruct supported visuals. Returns mapping element_id -> artifact info
    and list of warning strings.
    """
    recon_dir = artifact_dir / "reconstructed"
    recon_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    warnings: List[str] = []
    visual_model = visual_model_signature(settings)

    for el in elements:
        analysis = analyses.get(el.element_id)
        if analysis is None:
            continue
        # Always persist analysis JSON beside original
        analysis_path = artifact_dir / "enrichments" / f"{el.element_id}.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")

        allowed, kind = reconstruction_allowed(analysis, settings)
        info = {
            "figure_type": normalize_figure_type(analysis.figure_type),
            "reconstruction_allowed": allowed,
            "reason": kind,
            "source_image": el.image_path,
            "analysis_path": str(analysis_path),
            "reconstructed_path": None,
            "values_estimated": analysis.values_are_estimated,
            "confidence": analysis.confidence,
            "warnings": list(analysis.warnings),
            "visual_model_provider": visual_model["provider"],
            "visual_model_id": visual_model["model"],
            "visual_prompt_version": VISUAL_PROMPT_VERSION,
            "visual_schema_version": VISUAL_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if analysis.values_are_estimated and "values_estimated" not in info["warnings"]:
            info["warnings"].append("values_estimated")
        if not allowed:
            results[el.element_id] = info
            continue

        if kind == "chart" and analysis.chart:
            chart_warnings = validate_chart(analysis.chart)
            info["warnings"].extend(chart_warnings)
            severe_prefixes = (
                "no_series",
                "series_count_mismatch:",
                "empty_series:",
                "missing_y:",
                "invalid_y:",
                "invalid_x:",
                "invalid_x_range:",
                "invalid_y_range:",
                "x_below_axis:",
                "x_above_axis:",
                "y_below_axis:",
                "y_above_axis:",
                "negative_pie_value:",
            )
            if any(w.startswith(severe_prefixes) for w in chart_warnings):
                info["reconstruction_allowed"] = False
                info["reason"] = "chart_validation_failed"
                results[el.element_id] = info
                warnings.extend(chart_warnings)
                continue
            # Normalize chart_type from figure_type if empty
            if not analysis.chart.chart_type:
                ft = normalize_figure_type(analysis.figure_type)
                analysis.chart.chart_type = ft.replace("_chart", "") if ft.endswith("_chart") else ft  # type: ignore
            out = recon_dir / f"{el.element_id}-chart.png"
            rendered = render_chart(analysis.chart, out)
            info["reconstructed_path"] = str(rendered) if rendered else None
            if rendered is None:
                info["reconstruction_allowed"] = False
                info["reason"] = "renderer_unavailable"
                info["warnings"].append("chart_renderer_unavailable")
            # Persist chart spec
            (recon_dir / f"{el.element_id}-chart.json").write_text(
                analysis.chart.model_dump_json(indent=2), encoding="utf-8"
            )
        elif kind == "diagram" and analysis.diagram:
            diag_warnings = validate_diagram(analysis.diagram)
            info["warnings"].extend(diag_warnings)
            if any(
                warning == "no_nodes"
                or warning == "duplicate_node_ids"
                or warning.startswith("dangling_edge:")
                for warning in diag_warnings
            ):
                info["reconstruction_allowed"] = False
                info["reason"] = "diagram_validation_failed"
                results[el.element_id] = info
                warnings.extend(diag_warnings)
                continue
            out = recon_dir / f"{el.element_id}-diagram.png"
            rendered = render_diagram(analysis.diagram, out)
            info["reconstructed_path"] = str(rendered) if rendered else None
            if rendered is None:
                info["reconstruction_allowed"] = False
                info["reason"] = "renderer_unavailable"
                info["warnings"].append("diagram_renderer_unavailable")
            (recon_dir / f"{el.element_id}-diagram.json").write_text(
                analysis.diagram.model_dump_json(indent=2), encoding="utf-8"
            )

        results[el.element_id] = info
        warnings.extend(info["warnings"])

    (recon_dir / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results, warnings
