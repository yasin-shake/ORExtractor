from ingestion.models import (
    ChartSpecification,
    DiagramEdge,
    DiagramNode,
    DiagramSpecification,
    VisualAnalysis,
)
from ingestion.visuals import reconstruction_allowed, validate_chart, validate_diagram


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
        confidence=0.9,
        chart=ChartSpecification(
            chart_type="line",
            series=[{"name": "a", "points": [{"x": 1, "y": 2}]}],
        ),
    )
    allowed, reason = reconstruction_allowed(analysis, _Settings())
    assert allowed is True
    assert reason == "chart"
