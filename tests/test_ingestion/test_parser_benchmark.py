import json
import subprocess
import sys
import time
from pathlib import Path

from ingestion.models import (
    ElementRecord,
    ParserQualityReport,
    ParserResult,
)
from ingestion.parser_benchmark import run_parser_benchmark


def _result(parser, *, missing_table=False, missing_title=False):
    elements = [
        ElementRecord(
            element_id=f"{parser}-body",
            source_file="report.pdf",
            category="NarrativeText",
            text="Measured and indicated resources total 10 million tonnes.",
            page_number=1,
            parser=parser,
        ),
    ]
    if not missing_title:
        elements.insert(
            0,
            ElementRecord(
                element_id=f"{parser}-title",
                source_file="report.pdf",
                category="Title",
                text="Item 14 - Mineral Resource Estimates",
                page_number=1,
                parser=parser,
            ),
        )
    if not missing_table:
        elements.append(
            ElementRecord(
                element_id=f"{parser}-table",
                source_file="report.pdf",
                category="Table",
                text="Grade Tonnes 1.2 1000",
                page_number=2,
                parser=parser,
            )
        )
    return ParserResult(
        source_file="report.pdf",
        parser=parser,
        parser_version="test",
        elements=elements,
        page_count=2,
        duration_ms=100.0,
        quality=ParserQualityReport(score=0.99),
    )


class _Parser:
    def __init__(self, result, delay=0.0):
        self.result = result
        self.calls = 0
        self.delay = delay

    def parse(self, _pdf_path, *, source_file, artifact_dir):
        self.calls += 1
        time.sleep(self.delay)
        assert source_file == "report.pdf"
        assert "benchmark" in str(artifact_dir)
        return self.result


def test_parser_benchmark_is_read_only_and_flags_table_regression(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF benchmark")
    parser = _Parser(_result("hybrid", missing_table=True))
    output_dir = tmp_path / "benchmark"

    report, json_path, markdown_path = run_parser_benchmark(
        pdf,
        source_file="report.pdf",
        baseline=_result("docling"),
        candidate_parser=parser,
        output_dir=output_dir,
    )

    assert parser.calls == 1
    assert report["metrics"]["table_page_recall"] == 0.0
    assert report["accepted"] is False
    assert "table_page_recall" in report["failed_gates"]
    assert json_path.exists()
    assert markdown_path.exists()
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["source_file"] == "report.pdf"


def test_parser_benchmark_rejects_semantic_heading_regression(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF benchmark")

    report, _, _ = run_parser_benchmark(
        pdf,
        source_file="report.pdf",
        baseline=_result("docling"),
        candidate_parser=_Parser(
            _result("hybrid", missing_title=True)
        ),
        output_dir=tmp_path / "benchmark",
    )

    assert report["metrics"]["title_page_recall"] == 0.0
    assert "title_page_recall" in report["failed_gates"]


def test_live_benchmark_requires_a_material_speedup(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF benchmark")

    report, _, _ = run_parser_benchmark(
        pdf,
        source_file="report.pdf",
        baseline=_result("docling"),
        candidate_parser=_Parser(_result("hybrid"), delay=0.02),
        timing_baseline_parser=_Parser(_result("docling")),
        output_dir=tmp_path / "benchmark",
    )

    assert report["performance_verified"] is True
    assert report["production_candidate"] is False
    assert "minimum_speedup" in report["failed_gates"]


def test_parser_benchmark_is_exposed_as_a_read_only_cli():
    result = subprocess.run(
        [sys.executable, "rag_app.py", "benchmark-ingestion", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--live-docling" in result.stdout
    assert "--pages" in result.stdout
