"""Read-only quality/performance benchmark for candidate document parsers."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from ingestion.models import ElementRecord, ParserResult


def _page_text(elements: list[ElementRecord]) -> dict[int, str]:
    values: dict[int, list[str]] = defaultdict(list)
    for element in elements:
        text = (
            element.text_as_markdown
            or element.text
            or element.text_as_html
        ).strip()
        if text:
            values[int(element.page_number)].append(text)
    return {
        page: "\n".join(parts)
        for page, parts in values.items()
    }


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+(?:[.,][0-9]+)?", text)
        if len(token) > 1
    }


def _micro_token_recall(
    baseline_pages: dict[int, str],
    candidate_pages: dict[int, str],
) -> float:
    expected = 0
    recovered = 0
    for page, baseline_text in baseline_pages.items():
        baseline_tokens = _tokens(baseline_text)
        candidate_tokens = _tokens(candidate_pages.get(page, ""))
        expected += len(baseline_tokens)
        recovered += len(
            baseline_tokens.intersection(candidate_tokens)
        )
    return recovered / expected if expected else 1.0


def _category_pages(
    elements: list[ElementRecord],
    category: str,
) -> set[int]:
    return {
        int(element.page_number)
        for element in elements
        if element.category == category
    }


def _semantic_title_pages(
    elements: list[ElementRecord],
) -> set[int]:
    titles = [
        element
        for element in elements
        if element.category == "Title"
        and element.text.strip()
    ]
    normalized_counts = Counter(
        re.sub(
            r"\W+",
            " ",
            element.text.casefold(),
        ).strip()
        for element in titles
    )
    return {
        int(element.page_number)
        for element in titles
        if normalized_counts[
            re.sub(
                r"\W+",
                " ",
                element.text.casefold(),
            ).strip()
        ]
        < 2
    }


def _page_recall(expected: set[int], actual: set[int]) -> float:
    if not expected:
        return 1.0
    return len(expected.intersection(actual)) / len(expected)


def _round(value: float) -> float:
    return round(float(value), 6)


def run_parser_benchmark(
    pdf_path: Path,
    *,
    source_file: str,
    baseline: ParserResult,
    candidate_parser,
    output_dir: Path,
    timing_baseline_parser=None,
    minimum_page_coverage: float = 0.98,
    minimum_token_recall: float = 0.95,
    minimum_table_page_recall: float = 0.98,
    minimum_title_page_recall: float = 0.95,
    minimum_speedup: float = 1.10,
) -> tuple[dict, Path, Path]:
    """Run one parser without touching Chroma and persist scored evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_artifacts = output_dir / "candidate_artifacts"
    started = time.perf_counter()
    candidate = candidate_parser.parse(
        pdf_path,
        source_file=source_file,
        artifact_dir=candidate_artifacts,
    )
    wall_duration_ms = (time.perf_counter() - started) * 1000
    live_baseline = None
    if timing_baseline_parser is not None:
        baseline_started = time.perf_counter()
        live_result = timing_baseline_parser.parse(
            pdf_path,
            source_file=source_file,
            artifact_dir=(
                output_dir / "live_baseline_artifacts"
            ),
        )
        live_wall_duration_ms = (
            time.perf_counter() - baseline_started
        ) * 1000
        live_baseline = {
            "parser": live_result.parser,
            "status": live_result.status,
            "quality_score": live_result.quality.score,
            "quality_reasons": live_result.quality.reasons,
            "wall_duration_ms": round(
                live_wall_duration_ms,
                3,
            ),
            "reported_duration_ms": round(
                float(live_result.duration_ms),
                3,
            ),
        }

    baseline_pages = _page_text(baseline.elements)
    candidate_pages = _page_text(candidate.elements)
    expected_pages = set(
        range(
            1,
            max(
                int(baseline.page_count or 0),
                max(baseline_pages, default=0),
            )
            + 1,
        )
    )
    covered_pages = {
        page
        for page in expected_pages
        if candidate_pages.get(page, "").strip()
    }
    page_coverage = (
        len(covered_pages) / len(expected_pages)
        if expected_pages
        else 1.0
    )
    table_page_recall = _page_recall(
        _category_pages(baseline.elements, "Table"),
        _category_pages(candidate.elements, "Table"),
    )
    title_page_recall = _page_recall(
        _semantic_title_pages(baseline.elements),
        _semantic_title_pages(candidate.elements),
    )
    baseline_characters = sum(
        len(re.sub(r"\s+", "", value))
        for value in baseline_pages.values()
    )
    candidate_characters = sum(
        len(re.sub(r"\s+", "", value))
        for value in candidate_pages.values()
    )
    character_recall = (
        min(1.0, candidate_characters / baseline_characters)
        if baseline_characters
        else 1.0
    )
    metrics = {
        "page_coverage": _round(page_coverage),
        "token_recall": _round(
            _micro_token_recall(
                baseline_pages,
                candidate_pages,
            )
        ),
        "character_recall": _round(character_recall),
        "table_page_recall": _round(table_page_recall),
        "title_page_recall": _round(title_page_recall),
        "candidate_wall_duration_ms": round(
            wall_duration_ms,
            3,
        ),
        "candidate_reported_duration_ms": round(
            float(candidate.duration_ms),
            3,
        ),
        "baseline_reported_duration_ms": round(
            float(baseline.duration_ms),
            3,
        ),
        "live_baseline_wall_duration_ms": (
            live_baseline["wall_duration_ms"]
            if live_baseline is not None
            else None
        ),
        "candidate_speedup_vs_live_baseline": (
            _round(
                live_baseline["wall_duration_ms"]
                / wall_duration_ms
            )
            if live_baseline is not None
            and wall_duration_ms > 0
            else None
        ),
    }
    gates = {
        "page_coverage": (
            metrics["page_coverage"] >= minimum_page_coverage
        ),
        "token_recall": (
            metrics["token_recall"] >= minimum_token_recall
        ),
        "table_page_recall": (
            metrics["table_page_recall"]
            >= minimum_table_page_recall
        ),
        "title_page_recall": (
            metrics["title_page_recall"]
            >= minimum_title_page_recall
        ),
        "candidate_quality": not candidate.quality.reasons,
        "candidate_status": candidate.status
        in {"success", "completed", "partial_success"},
    }
    if live_baseline is not None:
        gates["minimum_speedup"] = (
            metrics["candidate_speedup_vs_live_baseline"]
            >= minimum_speedup
        )
    failed_gates = [
        name for name, passed in gates.items() if not passed
    ]
    report = {
        "source_file": source_file,
        "baseline": {
            "parser": baseline.parser,
            "parser_version": baseline.parser_version,
            "page_count": baseline.page_count,
            "element_count": len(baseline.elements),
            "category_counts": dict(
                Counter(
                    element.category
                    for element in baseline.elements
                )
            ),
            "quality": baseline.quality.model_dump(mode="json"),
        },
        "candidate": {
            "parser": candidate.parser,
            "parser_version": candidate.parser_version,
            "status": candidate.status,
            "page_count": candidate.page_count,
            "element_count": len(candidate.elements),
            "category_counts": dict(
                Counter(
                    element.category
                    for element in candidate.elements
                )
            ),
            "quality": candidate.quality.model_dump(mode="json"),
            "metadata": candidate.metadata,
            "errors": candidate.errors,
            "warnings": candidate.warnings,
        },
        "live_baseline": live_baseline,
        "thresholds": {
            "minimum_page_coverage": minimum_page_coverage,
            "minimum_token_recall": minimum_token_recall,
            "minimum_table_page_recall": (
                minimum_table_page_recall
            ),
            "minimum_title_page_recall": (
                minimum_title_page_recall
            ),
            "minimum_speedup": minimum_speedup,
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": failed_gates,
        "accepted": not failed_gates,
        "performance_verified": live_baseline is not None,
        "production_candidate": (
            live_baseline is not None
            and not failed_gates
        ),
        "writes_to_chroma": False,
    }
    json_path = output_dir / "parser-benchmark.json"
    markdown_path = output_dir / "parser-benchmark.md"
    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        "\n".join(
            [
                f"# Parser benchmark — {source_file}",
                "",
                f"- Accepted: **{report['accepted']}**",
                f"- Candidate: `{candidate.parser}`",
                f"- Page coverage: {metrics['page_coverage']:.3f}",
                f"- Token recall: {metrics['token_recall']:.3f}",
                (
                    "- Table-page recall: "
                    f"{metrics['table_page_recall']:.3f}"
                ),
                (
                    "- Candidate wall time: "
                    f"{metrics['candidate_wall_duration_ms'] / 1000:.2f}s"
                ),
                (
                    "- Baseline reported time: "
                    f"{metrics['baseline_reported_duration_ms'] / 1000:.2f}s"
                ),
                *(
                    [
                        (
                            "- Live Docling baseline: "
                            f"{metrics['live_baseline_wall_duration_ms'] / 1000:.2f}s"
                        ),
                        (
                            "- Candidate speedup: "
                            f"{metrics['candidate_speedup_vs_live_baseline']:.2f}x"
                        ),
                    ]
                    if live_baseline is not None
                    else []
                ),
                (
                    "- Failed gates: "
                    + (
                        ", ".join(failed_gates)
                        if failed_gates
                        else "none"
                    )
                ),
                "- Chroma writes: none",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report, json_path, markdown_path
