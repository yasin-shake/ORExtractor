"""Reproducible quality and performance benchmarks for visual-model adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from ingestion.models import ParserResult, TableValidation, VisualAnalysis
from ingestion.visual_model import VisualRequest


_NUMBER_RE = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?%?")


@dataclass(frozen=True)
class BenchmarkExpectation:
    """Independent gold expectations for a benchmark case."""

    figure_type: str | None = None
    reconstruction_supported: bool | None = None
    table_is_valid: bool | None = None
    required_text: tuple[str, ...] = ()
    expected_numbers: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualBenchmarkCase:
    case_id: str
    request: VisualRequest
    expectation: BenchmarkExpectation | None = None
    source: str = "synthetic"


@dataclass(frozen=True)
class VisualBenchmarkCaseResult:
    case_id: str
    source: str
    task: str
    schema_valid: bool
    classification_correct: bool | None
    required_text_recall: float | None
    numeric_recall: float | None
    unexpected_numbers: tuple[str, ...]
    passed: bool | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    output: dict | None
    error: str = ""


@dataclass(frozen=True)
class VisualBenchmarkSummary:
    total_cases: int
    gold_cases: int
    successful_cases: int
    passed_cases: int
    schema_valid_rate: float
    classification_accuracy: float | None
    required_text_recall: float | None
    numeric_recall: float | None
    mean_latency_ms: float | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class VisualBenchmarkReport:
    provider: str
    model_id: str
    summary: VisualBenchmarkSummary
    cases: tuple[VisualBenchmarkCaseResult, ...]


def _normal_number(value: str) -> str:
    raw = value.replace(",", "").strip().removesuffix("%")
    try:
        normalized = format(Decimal(raw).normalize(), "f")
    except InvalidOperation:
        return raw
    return "0" if normalized in {"-0", "+0"} else normalized


def _content_text(value: VisualAnalysis | TableValidation) -> str:
    if isinstance(value, VisualAnalysis):
        fields = {
            "caption": value.caption,
            "description": value.description,
            "labels": value.labels,
            "chart": value.chart.model_dump(mode="json") if value.chart else None,
            "diagram": (
                value.diagram.model_dump(mode="json") if value.diagram else None
            ),
        }
    else:
        fields = {
            "description": value.description,
            "issues": value.issues,
            "normalized_markdown": value.normalized_markdown,
            "warnings": value.warnings,
        }
    return str(fields)


def _numeric_text(value: VisualAnalysis | TableValidation) -> str:
    """Return only fields that may legitimately contain transcribed values."""

    if isinstance(value, TableValidation):
        return value.normalized_markdown
    chart = (
        value.chart.model_dump(
            mode="json",
            exclude={"expected_series_count"},
        )
        if value.chart
        else None
    )
    return str(
        {
            "caption": value.caption,
            "description": value.description,
            "labels": value.labels,
            "chart": chart,
        }
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _benchmark_font(size: int, *, bold: bool = False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _encode_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _new_canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), title, fill="black", font=_benchmark_font(30, bold=True))
    return image, draw


def _draw_table(
    title: str,
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> Image.Image:
    image, draw = _new_canvas(title)
    x0, y0, width = 40, 100, 920
    row_height = 70
    columns = len(headers)
    col_width = width / columns
    for row_index in range(len(rows) + 2):
        y = y0 + row_index * row_height
        draw.line((x0, y, x0 + width, y), fill="black", width=2)
    for column in range(columns + 1):
        x = x0 + column * col_width
        draw.line(
            (x, y0, x, y0 + row_height * (len(rows) + 1)),
            fill="black",
            width=2,
        )
    for column, value in enumerate(headers):
        draw.text(
            (x0 + column * col_width + 12, y0 + 20),
            value,
            fill="black",
            font=_benchmark_font(20, bold=True),
        )
    for row_index, row in enumerate(rows, start=1):
        for column, value in enumerate(row):
            draw.text(
                (
                    x0 + column * col_width + 12,
                    y0 + row_index * row_height + 20,
                ),
                value,
                fill="black",
                font=_benchmark_font(20),
            )
    return image


def _draw_chart(
    title: str,
    kind: str,
    labels: tuple[str, ...],
    values: tuple[float, ...],
    y_label: str,
) -> Image.Image:
    image, draw = _new_canvas(title)
    left, top, right, bottom = 110, 100, 940, 510
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    draw.text((15, 275), y_label, fill="black", font=_benchmark_font(18))
    max_value = max(values) * 1.2
    points: list[tuple[float, float]] = []
    spacing = (right - left) / (len(values) + 1)
    for index, (label, value) in enumerate(zip(labels, values), start=1):
        x = left + index * spacing
        y = bottom - (value / max_value) * (bottom - top)
        points.append((x, y))
        draw.text((x - 28, bottom + 15), label, fill="black", font=_benchmark_font(18))
        draw.text((x - 20, y - 30), str(value), fill="black", font=_benchmark_font(18, bold=True))
        if kind == "bar":
            draw.rectangle((x - 45, y, x + 45, bottom), fill="#4472C4", outline="black")
        elif kind == "scatter":
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#C00000")
    if kind == "line":
        draw.line(points, fill="#4472C4", width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#4472C4")
    return image


def build_synthetic_visual_cases(output_dir: Path) -> list[VisualBenchmarkCase]:
    """Create a fixed eight-case NI 43-101 visual benchmark with gold facts."""

    from ingestion.enrichment import (
        FIGURE_ANALYSIS_INSTRUCTIONS,
        TABLE_VALIDATION_INSTRUCTIONS,
    )

    table_prompt = TABLE_VALIDATION_INSTRUCTIONS
    figure_prompt = FIGURE_ANALYSIS_INSTRUCTIONS
    definitions: list[
        tuple[str, str, Image.Image, BenchmarkExpectation]
    ] = [
        (
            "resource-table",
            "table",
            _draw_table(
                "Mineral Resource Estimate",
                ("Category", "Tonnes (Mt)", "Gold Grade (g/t)"),
                (
                    ("Measured", "12.5", "1.20"),
                    ("Indicated", "30.0", "0.85"),
                ),
            ),
            BenchmarkExpectation(
                table_is_valid=True,
                required_text=("measured", "indicated"),
                expected_numbers=("12.5", "1.20", "30.0", "0.85"),
            ),
        ),
        (
            "economics-table",
            "table",
            _draw_table(
                "Project Economics",
                ("Metric", "Value", "Unit"),
                (
                    ("After-tax NPV", "425", "USD M"),
                    ("IRR", "18.5", "%"),
                    ("Initial Capex", "310", "USD M"),
                    ("Mine Life", "14", "years"),
                ),
            ),
            BenchmarkExpectation(
                table_is_valid=True,
                required_text=("after-tax npv", "initial capex", "mine life"),
                expected_numbers=("425", "18.5", "310", "14"),
            ),
        ),
        (
            "production-bar-chart",
            "figure",
            _draw_chart(
                "Annual Production",
                "bar",
                ("2025", "2026", "2027"),
                (80, 110, 95),
                "Copper (kt)",
            ),
            BenchmarkExpectation(
                figure_type="bar_chart",
                reconstruction_supported=True,
                required_text=("annual production",),
                expected_numbers=("2025", "2026", "2027", "80", "110", "95"),
            ),
        ),
        (
            "recovery-line-chart",
            "figure",
            _draw_chart(
                "Gold Recovery by Grind Size",
                "line",
                ("150 um", "100 um", "75 um"),
                (82, 88, 92),
                "Recovery (%)",
            ),
            BenchmarkExpectation(
                figure_type="line_chart",
                reconstruction_supported=True,
                required_text=("gold recovery",),
                expected_numbers=("150", "100", "75", "82", "88", "92"),
            ),
        ),
        (
            "grade-scatter-chart",
            "figure",
            _draw_chart(
                "Drillhole Grade by Depth",
                "scatter",
                ("50 m", "100 m", "150 m"),
                (1.2, 2.4, 1.8),
                "Gold (g/t)",
            ),
            BenchmarkExpectation(
                figure_type="scatter_chart",
                reconstruction_supported=True,
                required_text=("drillhole grade",),
                expected_numbers=("50", "100", "150", "1.2", "2.4", "1.8"),
            ),
        ),
    ]

    process, draw = _new_canvas("Copper Concentrator Process Flow")
    stages = ("ROM Ore", "Crushing", "Grinding", "Flotation", "Copper Concentrate")
    for index, stage in enumerate(stages):
        x = 30 + index * 190
        draw.rounded_rectangle((x, 245, x + 150, 335), radius=10, outline="black", width=3)
        draw.text((x + 12, 275), stage, fill="black", font=_benchmark_font(16, bold=True))
        if index < len(stages) - 1:
            draw.line((x + 150, 290, x + 185, 290), fill="black", width=4)
            draw.polygon(
                ((x + 185, 290), (x + 170, 280), (x + 170, 300)),
                fill="black",
            )
    definitions.append(
        (
            "process-flow-diagram",
            "figure",
            process,
            BenchmarkExpectation(
                figure_type="process_diagram",
                reconstruction_supported=True,
                required_text=("crushing", "grinding", "flotation"),
            ),
        )
    )

    cross_section, draw = _new_canvas("Geological Cross Section A-A'")
    draw.polygon(((60, 180), (940, 210), (940, 500), (60, 500)), fill="#D9EAD3")
    draw.polygon(((60, 280), (940, 330), (940, 500), (60, 500)), fill="#FCE5CD")
    draw.line((250, 160, 600, 510), fill="red", width=8)
    draw.text((280, 210), "Fault F1", fill="red", font=_benchmark_font(22, bold=True))
    draw.text((500, 360), "Quartz Vein", fill="purple", font=_benchmark_font(22, bold=True))
    draw.text((65, 515), "WEST", fill="black", font=_benchmark_font(20))
    draw.text((870, 515), "EAST", fill="black", font=_benchmark_font(20))
    definitions.append(
        (
            "geological-cross-section",
            "figure",
            cross_section,
            BenchmarkExpectation(
                figure_type="geological_cross_section",
                reconstruction_supported=False,
                required_text=("fault", "quartz vein"),
            ),
        )
    )

    mine_plan, draw = _new_canvas("Open Pit Mine Plan - Phase 2")
    for inset, color in ((0, "#D9D2E9"), (35, "#B4A7D6"), (70, "#8E7CC3")):
        draw.ellipse(
            (150 + inset, 110 + inset, 850 - inset, 540 - inset),
            outline=color,
            width=16,
        )
    draw.arc((180, 130, 820, 520), start=15, end=325, fill="black", width=6)
    draw.text((620, 430), "Haul Road", fill="black", font=_benchmark_font(22, bold=True))
    draw.line((900, 170, 900, 100), fill="black", width=5)
    draw.polygon(((900, 80), (885, 110), (915, 110)), fill="black")
    draw.text((885, 55), "N", fill="black", font=_benchmark_font(20, bold=True))
    definitions.append(
        (
            "mine-plan",
            "figure",
            mine_plan,
            BenchmarkExpectation(
                figure_type="mine_plan",
                reconstruction_supported=False,
                required_text=("open pit", "haul road"),
            ),
        )
    )

    cases: list[VisualBenchmarkCase] = []
    for case_id, task, image, expectation in definitions:
        encoded = _encode_image(image, output_dir / f"{case_id}.png")
        cases.append(
            VisualBenchmarkCase(
                case_id=case_id,
                request=VisualRequest(
                    task=task,
                    prompt=table_prompt if task == "table" else figure_prompt,
                    image_base64=encoded,
                    media_type="image/png",
                ),
                expectation=expectation,
            )
        )
    return cases


def sample_real_visual_cases(
    artifact_dir: Path,
    settings,
    *,
    limit: int = 20,
) -> list[VisualBenchmarkCase]:
    """Deterministically sample production-routed visuals from retained artifacts."""

    from ingestion.context import build_visual_context, needs_table_validation
    from ingestion.enrichment import (
        _figure_message,
        _table_message,
        requires_deterministic_table_fallback,
        should_enrich_figure,
    )

    if limit <= 0:
        return []
    figures: list[VisualBenchmarkCase] = []
    tables: list[VisualBenchmarkCase] = []
    for result_path in sorted(artifact_dir.rglob("parser_result.json")):
        try:
            parser_result = ParserResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        for element in parser_result.elements:
            request = None
            if element.category in {"Image", "Figure"}:
                selected, _ = should_enrich_figure(element, settings)
                if selected:
                    context = build_visual_context(element).model_dump()
                    request = _figure_message(element, context, settings)
            elif needs_table_validation(element):
                if requires_deterministic_table_fallback(element):
                    continue
                context = build_visual_context(
                    element,
                    task="Validate the attached table extraction.",
                ).model_dump()
                request = _table_message(element, context, settings)
            if request is None:
                continue
            case = VisualBenchmarkCase(
                case_id=(
                    f"{element.source_file}:p{element.page_number}:"
                    f"{element.element_id}"
                ),
                request=request,
                expectation=None,
                source="retained-artifact",
            )
            (figures if request.task == "figure" else tables).append(case)
        if len(figures) >= limit and len(tables) >= limit:
            break

    def stable_order(case: VisualBenchmarkCase) -> str:
        return hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()

    figures.sort(key=stable_order)
    tables.sort(key=stable_order)
    table_target = min(len(tables), limit // 2)
    figure_target = min(len(figures), limit - table_target)
    remaining = limit - table_target - figure_target
    if remaining:
        extra_tables = min(len(tables) - table_target, remaining)
        table_target += extra_tables
        remaining -= extra_tables
        figure_target += min(len(figures) - figure_target, remaining)
    selected = tables[:table_target] + figures[:figure_target]
    return sorted(selected, key=stable_order)


def run_visual_benchmark(
    cases: Iterable[VisualBenchmarkCase],
    visual_model,
) -> VisualBenchmarkReport:
    """Run cases through the public visual-model interface and score gold facts."""

    results: list[VisualBenchmarkCaseResult] = []
    for case in cases:
        try:
            response = visual_model.analyze(case.request)
            value = response.value
            schema_valid = (
                isinstance(value, VisualAnalysis)
                if case.request.task == "figure"
                else isinstance(value, TableValidation)
            )
            expectation = case.expectation
            classification_checks: list[bool] = []
            text_recall = None
            numeric_recall = None
            unexpected_numbers: tuple[str, ...] = ()
            passed = None
            if expectation is not None:
                if expectation.figure_type is not None:
                    classification_checks.append(
                        isinstance(value, VisualAnalysis)
                        and value.figure_type == expectation.figure_type
                    )
                if expectation.reconstruction_supported is not None:
                    classification_checks.append(
                        isinstance(value, VisualAnalysis)
                        and value.reconstruction_supported
                        is expectation.reconstruction_supported
                    )
                if expectation.table_is_valid is not None:
                    classification_checks.append(
                        isinstance(value, TableValidation)
                        and value.is_valid is expectation.table_is_valid
                    )
                evidence = _content_text(value)
                evidence_folded = evidence.casefold()
                if expectation.required_text:
                    matched = sum(
                        text.casefold() in evidence_folded
                        for text in expectation.required_text
                    )
                    text_recall = matched / len(expectation.required_text)
                if expectation.expected_numbers:
                    expected_numbers = {
                        _normal_number(number)
                        for number in expectation.expected_numbers
                    }
                    output_numbers = {
                        _normal_number(number)
                        for number in _NUMBER_RE.findall(
                            _numeric_text(value)
                        )
                    }
                    numeric_recall = (
                        len(expected_numbers & output_numbers)
                        / len(expected_numbers)
                    )
                    unexpected_numbers = tuple(
                        sorted(output_numbers - expected_numbers)
                    )
                classification_correct = (
                    all(classification_checks) if classification_checks else None
                )
                passed = (
                    schema_valid
                    and classification_correct is not False
                    and text_recall in {None, 1.0}
                    and numeric_recall in {None, 1.0}
                    and not unexpected_numbers
                )
            else:
                classification_correct = None
            results.append(
                VisualBenchmarkCaseResult(
                    case_id=case.case_id,
                    source=case.source,
                    task=case.request.task,
                    schema_valid=schema_valid,
                    classification_correct=classification_correct,
                    required_text_recall=text_recall,
                    numeric_recall=numeric_recall,
                    unexpected_numbers=unexpected_numbers,
                    passed=passed,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    output=value.model_dump(mode="json"),
                )
            )
        except Exception as exc:
            results.append(
                VisualBenchmarkCaseResult(
                    case_id=case.case_id,
                    source=case.source,
                    task=case.request.task,
                    schema_valid=False,
                    classification_correct=False
                    if case.expectation is not None
                    else None,
                    required_text_recall=0.0
                    if case.expectation and case.expectation.required_text
                    else None,
                    numeric_recall=0.0
                    if case.expectation and case.expectation.expected_numbers
                    else None,
                    unexpected_numbers=(),
                    passed=False if case.expectation is not None else None,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    output=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    gold = [result for result in results if result.passed is not None]
    successful = [result for result in results if result.schema_valid]
    classification = [
        float(result.classification_correct)
        for result in gold
        if result.classification_correct is not None
    ]
    text_recalls = [
        result.required_text_recall
        for result in gold
        if result.required_text_recall is not None
    ]
    numeric_recalls = [
        result.numeric_recall
        for result in gold
        if result.numeric_recall is not None
    ]
    latencies = [result.latency_ms for result in successful]
    summary = VisualBenchmarkSummary(
        total_cases=len(results),
        gold_cases=len(gold),
        successful_cases=len(successful),
        passed_cases=sum(result.passed is True for result in gold),
        schema_valid_rate=len(successful) / len(results) if results else 0.0,
        classification_accuracy=_mean(classification),
        required_text_recall=_mean(text_recalls),
        numeric_recall=_mean(numeric_recalls),
        mean_latency_ms=_mean(latencies),
        input_tokens=sum(result.input_tokens for result in results),
        output_tokens=sum(result.output_tokens for result in results),
    )
    return VisualBenchmarkReport(
        provider=str(getattr(visual_model, "provider", "unknown")),
        model_id=str(getattr(visual_model, "model_id", "unknown")),
        summary=summary,
        cases=tuple(results),
    )


def write_visual_benchmark_report(
    report: VisualBenchmarkReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist machine-readable results and a concise human report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "visual_benchmark.json"
    markdown_path = output_dir / "visual_benchmark.md"
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = report.summary
    gold_pass_rate = (
        summary.passed_cases / summary.gold_cases if summary.gold_cases else 0.0
    )
    lines = [
        "# Visual-model benchmark",
        "",
        f"- Provider: `{report.provider}`",
        f"- Model: `{report.model_id}`",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Total cases | {summary.total_cases} |",
        f"| Gold cases | {summary.gold_cases} |",
        f"| Successful/schema-valid | {summary.successful_cases} |",
        f"| Gold pass rate | {gold_pass_rate:.2%} |",
        f"| Schema-valid rate | {summary.schema_valid_rate:.2%} |",
        (
            "| Classification accuracy | "
            f"{summary.classification_accuracy:.2%} |"
            if summary.classification_accuracy is not None
            else "| Classification accuracy | n/a |"
        ),
        (
            f"| Numeric recall | {summary.numeric_recall:.2%} |"
            if summary.numeric_recall is not None
            else "| Numeric recall | n/a |"
        ),
        (
            f"| Mean model latency | {summary.mean_latency_ms:.1f} ms |"
            if summary.mean_latency_ms is not None
            else "| Mean model latency | n/a |"
        ),
        "",
        "## Cases",
        "",
        "| Case | Source | Task | Schema | Gold pass | Latency (ms) | Error |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for result in report.cases:
        gold_pass = "n/a" if result.passed is None else str(result.passed)
        error = result.error.replace("|", "\\|")
        lines.append(
            f"| {result.case_id} | {result.source} | {result.task} | "
            f"{result.schema_valid} | "
            f"{gold_pass} | {result.latency_ms:.1f} | {error} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
