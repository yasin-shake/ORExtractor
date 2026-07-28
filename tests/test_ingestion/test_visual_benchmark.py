import base64
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ingestion.models import (
    ChartSpecification,
    ElementRecord,
    ParserResult,
    TableValidation,
    VisualAnalysis,
)
from ingestion.visual_benchmark import (
    BenchmarkExpectation,
    VisualBenchmarkCase,
    build_synthetic_visual_cases,
    run_visual_benchmark,
    sample_real_visual_cases,
    write_visual_benchmark_report,
)
from ingestion.enrichment import (
    FIGURE_ANALYSIS_INSTRUCTIONS,
    TABLE_VALIDATION_INSTRUCTIONS,
)
from ingestion.visual_model import VisualRequest, VisualResponse


def test_visual_benchmark_scores_independent_gold_expectations(tmp_path):
    case = VisualBenchmarkCase(
        case_id="gold-bar-chart",
        request=VisualRequest(
            task="figure",
            prompt="Read the chart values.",
            image_base64="aW1hZ2U=",
        ),
        expectation=BenchmarkExpectation(
            figure_type="bar_chart",
            reconstruction_supported=True,
            required_text=("annual production",),
            expected_numbers=("10", "15", "12"),
        ),
    )

    class GoldModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def analyze(self, _request):
            return VisualResponse(
                value=VisualAnalysis(
                    figure_type="bar_chart",
                    caption="Annual production",
                    description="Values are 10, 15, and 12 tonnes.",
                    contains_quantitative_data=True,
                    reconstruction_supported=True,
                    reconstruction_method="plotly",
                    confidence=0.94,
                    chart=ChartSpecification(
                        expected_series_count=1,
                    ),
                ),
                input_tokens=80,
                output_tokens=40,
                latency_ms=250,
            )

    report = run_visual_benchmark([case], GoldModel())

    assert report.summary.total_cases == 1
    assert report.summary.gold_cases == 1
    assert report.summary.passed_cases == 1
    assert report.summary.schema_valid_rate == 1.0
    assert report.summary.classification_accuracy == 1.0
    assert report.summary.numeric_recall == 1.0
    assert report.cases[0].task == "figure"
    assert report.cases[0].unexpected_numbers == ()
    assert report.cases[0].passed is True

    json_path, markdown_path = write_visual_benchmark_report(report, tmp_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["model_id"] == "qwen3-vl:test"
    assert saved["summary"]["passed_cases"] == 1
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "qwen3-vl:test" in markdown
    assert "| Gold pass rate | 100.00% |" in markdown


def test_visual_benchmark_treats_equivalent_numeric_formats_as_equal():
    case = VisualBenchmarkCase(
        case_id="numeric-formatting",
        request=VisualRequest(
            task="table",
            prompt="Read the table values.",
            image_base64="aW1hZ2U=",
        ),
        expectation=BenchmarkExpectation(
            table_is_valid=True,
            expected_numbers=("1.20", "30.0"),
        ),
    )

    class FormattingModel:
        provider = "ollama"
        model_id = "qwen3-vl:test"
        cache_id = "ollama:qwen3-vl:test"

        def analyze(self, _request):
            return VisualResponse(
                value=TableValidation(
                    is_valid=True,
                    description="The NI 43-101 values are legible.",
                    normalized_markdown="| Grade | Tonnes |\n|---:|---:|\n| 1.2% | 30 |",
                    confidence=0.95,
                )
            )

    report = run_visual_benchmark([case], FormattingModel())

    assert report.summary.numeric_recall == 1.0
    assert report.cases[0].unexpected_numbers == ()


def test_synthetic_benchmark_covers_domain_visual_classes(tmp_path):
    cases = build_synthetic_visual_cases(tmp_path)

    assert len(cases) == 8
    assert {case.request.task for case in cases} == {"figure", "table"}
    assert all(
        case.request.prompt.startswith(
            FIGURE_ANALYSIS_INSTRUCTIONS
            if case.request.task == "figure"
            else TABLE_VALIDATION_INSTRUCTIONS
        )
        for case in cases
    )
    assert all(case.expectation is not None for case in cases)
    assert {case.expectation.reconstruction_supported for case in cases} >= {
        True,
        False,
    }
    for case in cases:
        with Image.open(BytesIO(base64.b64decode(case.request.image_base64))) as image:
            assert image.width >= 800
            assert image.height >= 500


def test_real_benchmark_samples_retained_parser_artifacts(tmp_path):
    artifact = tmp_path / "report"
    image_path = artifact / "parsers" / "docling" / "images" / "figure.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (600, 400), "white").save(image_path)
    parser_result = ParserResult(
        source_file="nested/report.pdf",
        parser="docling",
        elements=[
            ElementRecord(
                element_id="figure-1",
                source_file="nested/report.pdf",
                category="Figure",
                page_number=14,
                image_path=str(image_path),
                image_width=600,
                image_height=400,
                caption="Figure 14-2 Resource classification",
                preceding_text="The mineral resource classes are shown below.",
                ni_item=14,
            )
        ],
    )
    (artifact / "parser_result.json").write_text(
        parser_result.model_dump_json(),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        visual_min_width=250,
        visual_min_height=150,
        visual_max_width=4096,
        visual_max_height=4096,
    )

    cases = sample_real_visual_cases(tmp_path, settings, limit=1)

    assert len(cases) == 1
    assert cases[0].case_id == "nested/report.pdf:p14:figure-1"
    assert cases[0].source == "retained-artifact"
    assert cases[0].expectation is None
    assert cases[0].request.task == "figure"
    assert "Resource classification" in cases[0].request.prompt


def test_real_benchmark_excludes_text_only_tables_that_production_will_not_call(
    tmp_path,
):
    artifact = tmp_path / "report"
    artifact.mkdir()
    parser_result = ParserResult(
        source_file="report.pdf",
        parser="docling",
        elements=[
            ElementRecord(
                element_id="table-1",
                source_file="report.pdf",
                category="Table",
                page_number=14,
                text="Mineral resource table",
                text_as_html=(
                    "<table>"
                    + ("<tr><td>1.20</td></tr>" * 225)
                    + "</table>"
                ),
            )
        ],
    )
    (artifact / "parser_result.json").write_text(
        parser_result.model_dump_json(),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        visual_min_width=250,
        visual_min_height=150,
        visual_max_width=4096,
        visual_max_height=4096,
    )

    assert sample_real_visual_cases(tmp_path, settings, limit=1) == []


def test_visual_benchmark_is_exposed_as_a_read_only_cli():
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "rag_app.py", "benchmark-visuals", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0
    assert "--real-samples" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "--provider" in completed.stdout
