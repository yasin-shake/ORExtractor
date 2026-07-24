from ingestion.models import (
    ChartSpecification,
    DiagramEdge,
    DiagramNode,
    DiagramSpecification,
    ElementRecord,
    VisualAnalysis,
)
from ingestion.visuals import (
    reconstruct_visuals,
    reconstruction_allowed,
    validate_chart,
    validate_diagram,
)


class _Settings:
    bedrock_visual_confidence_threshold = 0.85
    visual_reconstruct_charts = True
    visual_reconstruct_diagrams = True


def test_visual_schema_accepts_valid_payload():
    v = VisualAnalysis(
        figure_type="line_chart",
        description="Recovery vs grind",
        contains_quantitative_data=True,
        reconstruction_supported=True,
        reconstruction_method="plotly",
        confidence=0.94,
        chart=ChartSpecification(
            chart_type="line",
            series=[{"name": "Au", "points": [{"x": 1, "y": 90}, {"x": 2, "y": 92}]}],
        ),
    )
    assert v.confidence == 0.94
    assert v.chart.series[0].name == "Au"


def test_unsupported_geological_map_rejected():
    analysis = VisualAnalysis(
        figure_type="geological_map",
        reconstruction_supported=True,
        confidence=0.99,
    )
    allowed, reason = reconstruction_allowed(analysis, _Settings())
    assert allowed is False
    assert reason == "unsupported_category"


def test_low_confidence_rejected():
    analysis = VisualAnalysis(
        figure_type="line_chart",
        reconstruction_supported=True,
        confidence=0.4,
        chart=ChartSpecification(
            chart_type="line",
            series=[{"name": "a", "points": [{"x": 1, "y": 2}]}],
        ),
    )
    allowed, reason = reconstruction_allowed(analysis, _Settings())
    assert allowed is False
    assert reason == "low_confidence"


def test_chart_validation_missing_series():
    warnings = validate_chart(ChartSpecification(chart_type="bar", series=[]))
    assert "no_series" in warnings


def test_diagram_dangling_edge():
    diag = DiagramSpecification(
        diagram_type="flowchart",
        nodes=[DiagramNode(id="a", label="A")],
        edges=[DiagramEdge(source="a", target="b")],
    )
    warnings = validate_diagram(diag)
    assert any("dangling_edge" in w for w in warnings)


def test_supported_line_chart_allowed():
    analysis = VisualAnalysis(
        figure_type="line_chart",
        reconstruction_supported=True,
        reconstruction_method="plotly",
        confidence=0.9,
        chart=ChartSpecification(
            chart_type="line",
            series=[{"name": "a", "points": [{"x": 1, "y": 2}]}],
        ),
    )
    allowed, reason = reconstruction_allowed(analysis, _Settings())
    assert allowed is True
    assert reason == "chart"


def test_chart_axis_and_series_mismatches_are_rejected(tmp_path):
    chart = ChartSpecification(
        chart_type="line",
        y_min=0,
        y_max=100,
        expected_series_count=2,
        series=[{"name": "a", "points": [{"x": 1, "y": 120}]}],
    )
    warnings = validate_chart(chart)
    assert any(w.startswith("series_count_mismatch:") for w in warnings)
    assert any(w.startswith("y_above_axis:") for w in warnings)

    element = ElementRecord(
        element_id="f1",
        source_file="r.pdf",
        category="Image",
        image_path=str(tmp_path / "source.png"),
    )
    analysis = VisualAnalysis(
        figure_type="line_chart",
        reconstruction_supported=True,
        reconstruction_method="plotly",
        confidence=0.99,
        chart=chart,
    )
    results, _ = reconstruct_visuals(
        [element], {"f1": analysis}, _Settings(), tmp_path
    )
    assert results["f1"]["reconstruction_allowed"] is False
    assert results["f1"]["reason"] == "chart_validation_failed"


def test_dangling_diagram_is_not_reconstructed(tmp_path):
    element = ElementRecord(
        element_id="d1",
        source_file="r.pdf",
        category="Image",
    )
    analysis = VisualAnalysis(
        figure_type="flowchart",
        reconstruction_supported=True,
        reconstruction_method="graphviz",
        confidence=0.99,
        diagram=DiagramSpecification(
            diagram_type="flowchart",
            nodes=[DiagramNode(id="a", label="A")],
            edges=[DiagramEdge(source="a", target="missing")],
        ),
    )
    results, _ = reconstruct_visuals(
        [element], {"d1": analysis}, _Settings(), tmp_path
    )
    assert results["d1"]["reconstruction_allowed"] is False
    assert results["d1"]["reason"] == "diagram_validation_failed"
