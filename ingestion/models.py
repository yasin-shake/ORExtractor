"""Canonical Pydantic models for the Unstructured ingestion pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


PIPELINE_VERSION = "2"
VISUAL_PROMPT_VERSION = "1"
VISUAL_SCHEMA_VERSION = "1"
PARTITIONER_VERSION = "1"


class ElementRecord(BaseModel):
    element_id: str
    source_file: str
    category: str

    text: str = ""
    text_as_html: str = ""

    page_number: int = 1
    parent_id: Optional[str] = None
    category_depth: Optional[int] = None

    coordinates: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    ni_item: int = 0
    section_title: str = ""
    section_path: List[str] = Field(default_factory=list)

    caption: str = ""
    preceding_text: str = ""
    following_text: str = ""

    is_duplicate: bool = False
    skip_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentContext(BaseModel):
    report_name: str
    page_number: int
    ni_item: int = 0
    section_title: str = ""
    section_path: List[str] = Field(default_factory=list)
    caption: str = ""
    preceding_text: str = ""
    following_text: str = ""
    table_html: Optional[str] = None
    task: str = "Classify and analyse the attached visual."


class ChartPoint(BaseModel):
    x: Optional[float | str] = None
    y: Optional[float] = None
    label: Optional[str] = None


class ChartSeries(BaseModel):
    name: str = ""
    points: List[ChartPoint] = Field(default_factory=list)


class ChartSpecification(BaseModel):
    chart_type: Literal["bar", "line", "scatter", "pie", ""] = ""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    series: List[ChartSeries] = Field(default_factory=list)


class DiagramNode(BaseModel):
    id: str
    label: str = ""


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: str = ""


class DiagramSpecification(BaseModel):
    diagram_type: Literal["flowchart", "process", ""] = ""
    title: str = ""
    nodes: List[DiagramNode] = Field(default_factory=list)
    edges: List[DiagramEdge] = Field(default_factory=list)


class VisualAnalysis(BaseModel):
    figure_type: str = "unknown"
    caption: str = ""
    description: str = ""
    contains_quantitative_data: bool = False
    reconstruction_supported: bool = False
    reconstruction_method: Literal["plotly", "graphviz", "none", ""] = "none"
    values_are_estimated: bool = False
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    chart: Optional[ChartSpecification] = None
    diagram: Optional[DiagramSpecification] = None


class TableValidation(BaseModel):
    is_valid: bool = True
    description: str = ""
    issues: List[str] = Field(default_factory=list)
    normalized_markdown: str = ""
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)


class IngestionError(BaseModel):
    element_id: str = ""
    stage: str = ""
    message: str = ""


class ReportIngestStats(BaseModel):
    filename: str
    pages: int = 0
    elements: int = 0
    text_elements: int = 0
    tables: int = 0
    figures: int = 0
    bedrock_calls: int = 0
    cache_hits: int = 0
    reconstructed_charts: int = 0
    reconstructed_diagrams: int = 0
    warnings: int = 0
    failed_elements: List[str] = Field(default_factory=list)
    indexed_chunks: int = 0


class IngestionMetrics(BaseModel):
    partition_ms: float = 0.0
    normalize_ms: float = 0.0
    enrich_ms: float = 0.0
    reconstruct_ms: float = 0.0
    chunk_ms: float = 0.0
    embed_ms: float = 0.0
    total_ms: float = 0.0
    bedrock_calls: int = 0
    cache_hits: int = 0


class IngestionResult(BaseModel):
    status: str = "completed"
    files: List[str] = Field(default_factory=list)
    reports: List[ReportIngestStats] = Field(default_factory=list)
    errors: List[IngestionError] = Field(default_factory=list)
    metrics: IngestionMetrics = Field(default_factory=IngestionMetrics)


UNSUPPORTED_RECONSTRUCTION_TYPES = frozenset(
    {
        "geological_map",
        "geological_cross_section",
        "cross_section",
        "mine_plan",
        "drill_hole_map",
        "pit_shell",
        "resource_block_model",
        "contour_map",
        "3d_geological_view",
        "map",
        "photo",
        "logo",
        "unknown",
    }
)

SUPPORTED_CHART_TYPES = frozenset({"bar_chart", "line_chart", "scatter_chart", "pie_chart", "bar", "line", "scatter", "pie"})
SUPPORTED_DIAGRAM_TYPES = frozenset({"flowchart", "process_diagram", "process", "diagram"})
