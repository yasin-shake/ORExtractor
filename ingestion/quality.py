"""Deterministic parser-quality scoring and fallback gates."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from ingestion.models import ElementRecord, ParserQualityReport


def _is_table(element: ElementRecord) -> bool:
    return element.category.lower() in {"table", "tableitem"}


def _is_figure(element: ElementRecord) -> bool:
    return element.category.lower() in {"figure", "image", "picture"}


def _is_heading(element: ElementRecord) -> bool:
    return element.category.lower() in {"title", "heading", "sectionheader", "header"}


def assess_parser_quality(
    elements: Iterable[ElementRecord],
    *,
    page_count: int = 0,
    conversion_status: str = "success",
    min_text_page_coverage: float = 0.90,
    max_empty_page_ratio: float = 0.10,
    max_replacement_char_ratio: float = 0.01,
    min_table_valid_ratio: float = 0.80,
    require_picture_crops: bool = False,
) -> ParserQualityReport:
    """Score normalized output without parser-specific assumptions."""
    records = list(elements)
    observed_pages = max((record.page_number for record in records), default=0)
    total_pages = max(page_count, observed_pages)
    text_by_page: Counter[int] = Counter()
    body_pages: set[int] = set()
    all_text: list[str] = []

    for record in records:
        text = (record.text or record.text_as_markdown or "").strip()
        if text:
            text_by_page[record.page_number] += len(text)
            all_text.append(text)
        if record.category.lower() in {
            "narrativetext",
            "listitem",
            "title",
            "table",
            "formula",
        }:
            body_pages.add(record.page_number)

    text_pages = sum(1 for count in text_by_page.values() if count >= 20)
    coverage = text_pages / total_pages if total_pages else 0.0
    empty_ratio = 1.0 - coverage if total_pages else 1.0
    joined_text = "".join(all_text)
    replacement_ratio = (
        joined_text.count("\ufffd") / len(joined_text) if joined_text else 0.0
    )

    tables = [record for record in records if _is_table(record)]
    valid_tables = [
        record
        for record in tables
        if (record.text_as_html or record.text_as_markdown or record.text).strip()
    ]
    table_valid_ratio = len(valid_tables) / len(tables) if tables else 1.0
    table_consistency: list[float] = []
    for table in valid_tables:
        source = table.text_as_markdown or table.text_as_html or table.text
        if "<tr" in source.lower():
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", source, flags=re.I | re.S)
            counts = [
                len(re.findall(r"<t[dh][^>]*>", row, flags=re.I))
                for row in rows
            ]
        else:
            counts = [
                max(0, row.count("|") - 1)
                for row in source.splitlines()
                if "|" in row and "---" not in row
            ]
        counts = [count for count in counts if count]
        table_consistency.append(
            1.0 if not counts or len(set(counts)) == 1 else min(counts) / max(counts)
        )
    table_column_consistency = (
        sum(table_consistency) / len(table_consistency)
        if table_consistency
        else 1.0
    )
    figures = [record for record in records if _is_figure(record)]
    figures_with_crops = sum(1 for record in figures if record.image_path)
    captions = sum(1 for record in figures if record.caption.strip())
    headers_footers = [
        record
        for record in records
        if record.category.lower() in {"header", "footer"}
    ]
    duplicate_header_footer_ratio = (
        sum(1 for record in headers_footers if record.is_duplicate)
        / len(headers_footers)
        if headers_footers
        else 0.0
    )
    reading_order_anomalies = sum(
        1
        for previous, current in zip(records, records[1:])
        if current.page_number < previous.page_number
    )
    page_count_agreement = (
        min(page_count, observed_pages) / max(page_count, observed_pages)
        if page_count and observed_pages
        else (1.0 if page_count == observed_pages else 0.0)
    )

    reasons: list[str] = []
    if conversion_status.lower() not in {"success", "completed", "partial_success"}:
        reasons.append("conversion_failed")
    if coverage < min_text_page_coverage:
        reasons.append("low_text_page_coverage")
    if empty_ratio > max_empty_page_ratio:
        reasons.append("high_empty_page_ratio")
    if replacement_ratio > max_replacement_char_ratio:
        reasons.append("high_replacement_character_ratio")
    if table_valid_ratio < min_table_valid_ratio:
        reasons.append("table_structure_failures")
    if require_picture_crops and figures and figures_with_crops < len(figures):
        reasons.append("missing_picture_crops")
    if not records:
        reasons.append("no_elements")

    score = 1.0
    score -= max(0.0, min_text_page_coverage - coverage) * 0.55
    score -= max(0.0, empty_ratio - max_empty_page_ratio) * 0.25
    score -= min(0.20, replacement_ratio * 5)
    score -= max(0.0, min_table_valid_ratio - table_valid_ratio) * 0.15
    if "conversion_failed" in reasons:
        score -= 0.50
    if "no_elements" in reasons:
        score = 0.0

    return ParserQualityReport(
        score=round(max(0.0, min(1.0, score)), 4),
        conversion_status=conversion_status,
        expected_page_count=page_count,
        observed_page_count=observed_pages,
        page_count_agreement=round(page_count_agreement, 4),
        pages_with_body_elements=len(body_pages),
        pages_with_extracted_text=len(text_by_page),
        characters_per_page={
            str(page): text_by_page.get(page, 0)
            for page in range(1, total_pages + 1)
        },
        text_coverage=round(coverage, 4),
        suspicious_page_ratio=round(empty_ratio, 4),
        near_empty_page_ratio=round(empty_ratio, 4),
        duplicate_header_footer_ratio=round(duplicate_header_footer_ratio, 4),
        table_count=len(tables),
        valid_table_count=len(valid_tables),
        table_valid_ratio=round(table_valid_ratio, 4),
        table_row_consistency=round(table_column_consistency, 4),
        table_column_consistency=round(table_column_consistency, 4),
        figure_count=len(figures),
        pictures_with_crops=figures_with_crops,
        caption_association_rate=round(
            captions / len(figures) if figures else 1.0, 4
        ),
        heading_count=sum(1 for record in records if _is_heading(record)),
        heading_max_depth=max(
            (
                record.category_depth or 0
                for record in records
                if _is_heading(record)
            ),
            default=0,
        ),
        reading_order_anomaly_count=reading_order_anomalies,
        element_count=len(records),
        replacement_character_ratio=round(replacement_ratio, 6),
        reasons=list(dict.fromkeys(reasons)),
    )
