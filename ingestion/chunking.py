"""Section-aware chunking into LangChain Documents."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document

from ingestion.models import (
    PIPELINE_VERSION,
    ElementRecord,
    TableValidation,
    VisualAnalysis,
)

_TEXT_CATEGORIES = frozenset({"NarrativeText", "ListItem", "Title", "Caption", "Formula"})


def _item_label(ni_item: int, section_title: str) -> str:
    if ni_item:
        title = section_title or ""
        return f"Item {ni_item}" + (f" — {title}" if title else "")
    return section_title or "Unassigned"


def _header(source: str, ni_item: int, section_title: str, page: int, extra: str = "") -> str:
    lines = [
        f"Report: {source}",
        _item_label(ni_item, section_title),
    ]
    if section_title and ni_item:
        lines.append(f"Section: {section_title}")
    lines.append(f"Page: {page}")
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n\n"


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if len(text) <= chunk_size:
        return [text]

    overlap = min(max(0, chunk_overlap), max(0, chunk_size - 1))
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            minimum_boundary = start + max(1, chunk_size // 2)
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, minimum_boundary, end)
                if boundary >= minimum_boundary:
                    end = boundary + len(separator)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def _content_chunks(
    header: str,
    body: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Return hard-bounded chunks while repeating provenance in every chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if len(header) >= chunk_size:
        header = header[: max(0, chunk_size - 2)].rstrip() + "\n"
    body_size = max(1, chunk_size - len(header))
    body_overlap = min(max(0, chunk_overlap), max(0, body_size - 1))
    pieces = _chunk_text(body, body_size, body_overlap)
    return [header + piece for piece in pieces] if pieces else [header.rstrip()]


_INLINE_IMAGE_MARKDOWN = re.compile(
    r"!\[[^\]]*\]\(data:image/[^)]*\)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_embedded_image_markup(value: str) -> str:
    return _INLINE_IMAGE_MARKDOWN.sub("", value).strip()


def _base_meta(el: ElementRecord, chunk_idx: int, doc_type: str, **extra) -> dict:
    meta = {
        "source": el.source_file,
        "page": el.page_number,
        "chunk": chunk_idx,
        "type": doc_type,
        "ni_item": el.ni_item,
        "section_title": el.section_title or "",
        "element_id": el.element_id,
        "parser": el.parser or "unknown",
        "parser_version": el.parser_version or "",
        "pipeline_version": PIPELINE_VERSION,
    }
    if el.parser_element_id:
        meta["parser_element_id"] = el.parser_element_id
    if el.image_path:
        meta["asset_path"] = el.image_path
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        else:
            meta[k] = json.dumps(v, ensure_ascii=True)
    return meta


def elements_to_documents(
    elements: List[ElementRecord],
    analyses: Optional[Dict[str, VisualAnalysis]] = None,
    validations: Optional[Dict[str, TableValidation]] = None,
    reconstructions: Optional[Dict[str, dict]] = None,
    chunk_size: int = 1400,
    chunk_overlap: int = 150,
    table_confidence_threshold: float = 0.85,
) -> List[Document]:
    analyses = analyses or {}
    validations = validations or {}
    reconstructions = reconstructions or {}
    docs: List[Document] = []
    text_buf: List[ElementRecord] = []
    chunk_counter = 0
    table_counter = 1000
    figure_counter = 2000

    def flush_text():
        nonlocal chunk_counter
        if not text_buf:
            return
        # Group consecutive text under same section
        parts: List[str] = []
        lead = text_buf[0]
        for el in text_buf:
            if el.category == "Title":
                parts.append(el.text.strip())
            elif el.category == "ListItem":
                parts.append(f"- {el.text.strip()}")
            else:
                parts.append(el.text.strip())
        body = "\n\n".join(p for p in parts if p)
        header = _header(
            lead.source_file,
            lead.ni_item,
            lead.section_title,
            lead.page_number,
        )
        for content in _content_chunks(
            header, body, chunk_size, chunk_overlap
        ):
            docs.append(
                Document(
                    page_content=content,
                    metadata=_base_meta(lead, chunk_counter, "text"),
                )
            )
            chunk_counter += 1
        text_buf.clear()

    for el in elements:
        if el.category in _DECORATIVE:
            continue
        if el.category in _TEXT_CATEGORIES and el.category != "Formula":
            # Flush when section changes materially
            if text_buf and (
                text_buf[-1].ni_item != el.ni_item
                or text_buf[-1].section_title != el.section_title
                or el.page_number - text_buf[-1].page_number > 1
            ):
                flush_text()
            text_buf.append(el)
            continue

        flush_text()

        if el.category == "Formula" and el.text.strip():
            header = _header(
                el.source_file,
                el.ni_item,
                el.section_title,
                el.page_number,
                "Type: formula",
            )
            for content in _content_chunks(
                header, el.text.strip(), chunk_size, chunk_overlap
            ):
                docs.append(
                    Document(
                        page_content=content,
                        metadata=_base_meta(
                            el, chunk_counter, "formula"
                        ),
                    )
                )
                chunk_counter += 1
            continue

        if el.category == "Table":
            validation = validations.get(el.element_id)
            table_md = ""
            validation_reliable = bool(
                validation
                and validation.is_valid
                and validation.confidence >= table_confidence_threshold
            )
            if validation_reliable and validation.normalized_markdown:
                table_md = validation.normalized_markdown
            elif el.text_as_markdown:
                table_md = el.text_as_markdown
            elif el.text_as_html:
                table_md = _html_table_to_markdown(el.text_as_html) or el.text
            else:
                table_md = el.text
            desc = validation.description if validation else ""
            header = _header(
                el.source_file,
                el.ni_item,
                el.section_title,
                el.page_number,
                f"Table: {el.caption or el.element_id}",
            )
            body = (table_md or "").strip()
            if desc:
                body += f"\n\nTable context:\n{desc}"
            if validation and (validation.issues or validation.warnings):
                warnings = list(validation.issues) + list(validation.warnings)
                body += f"\n\nValidation warnings: {'; '.join(warnings)}"
            contents = _content_chunks(
                header, body, chunk_size, chunk_overlap
            )
            for part, content in enumerate(contents, start=1):
                docs.append(
                    Document(
                        page_content=content,
                        metadata=_base_meta(
                            el,
                            table_counter,
                            "table",
                            confidence=(
                                validation.confidence if validation else None
                            ),
                            table_valid=(
                                validation.is_valid if validation else None
                            ),
                            part=part,
                            parts=len(contents),
                        ),
                    )
                )
                table_counter += 1
            continue

        if el.category in {"Image", "Figure"}:
            analysis = analyses.get(el.element_id)
            recon = reconstructions.get(el.element_id) or {}
            figure_type = (analysis.figure_type if analysis else "unknown")
            caption = (analysis.caption if analysis and analysis.caption else el.caption) or ""
            description = analysis.description if analysis else ""
            if not description:
                description = _strip_embedded_image_markup(el.text)
                if not description:
                    description = "Figure with no extracted description."

            doc_type = "figure"
            extracted_values = ""
            if (
                analysis
                and analysis.chart
                and analysis.chart.series
                and recon.get("reconstruction_allowed")
                and recon.get("reason") == "chart"
            ):
                doc_type = "chart_data"
                extracted_values = analysis.chart.model_dump_json()
            elif (
                analysis
                and analysis.diagram
                and analysis.diagram.nodes
                and recon.get("reconstruction_allowed")
                and recon.get("reason") == "diagram"
            ):
                doc_type = "diagram"
                extracted_values = analysis.diagram.model_dump_json()

            header = _header(
                el.source_file,
                el.ni_item,
                el.section_title,
                el.page_number,
                f"Figure type: {figure_type}\nCaption: {caption}",
            )
            body = f"Description:\n{description}"
            if not analysis:
                surrounding = "\n\n".join(
                    part
                    for part in (el.preceding_text, el.following_text)
                    if part
                )
                if surrounding:
                    body += f"\n\nSurrounding context:\n{surrounding}"
                if el.skip_reason:
                    body += f"\n\nVisual enrichment status: {el.skip_reason}"
            if extracted_values:
                body += f"\n\nExtracted values:\n{extracted_values}"
            if analysis and analysis.warnings:
                body += f"\n\nWarnings: {'; '.join(analysis.warnings)}"

            contents = _content_chunks(
                header, body, chunk_size, chunk_overlap
            )
            for part, content in enumerate(contents, start=1):
                meta = _base_meta(
                    el,
                    figure_counter,
                    doc_type,
                    figure_type=figure_type,
                    confidence=analysis.confidence if analysis else None,
                    values_estimated=(
                        bool(analysis.values_are_estimated) if analysis else False
                    ),
                    reconstructed_path=recon.get("reconstructed_path"),
                    enrichment_status=(
                        "completed"
                        if analysis
                        else (el.skip_reason or "not_enriched")
                    ),
                    part=part,
                    parts=len(contents),
                )
                docs.append(
                    Document(page_content=content, metadata=meta)
                )
                figure_counter += 1

    flush_text()
    return docs


_DECORATIVE = frozenset({"Header", "Footer", "PageBreak"})


def _html_table_to_markdown(html: str) -> str:
    """Minimal HTML table → markdown (mirrors rag_app helper behaviour)."""
    import re

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
    if not rows:
        return ""
    md_rows: List[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.I | re.S)
        cleaned = [re.sub(r"<[^>]+>", "", c).strip().replace("\n", " ") for c in cells]
        if not cleaned:
            continue
        md_rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join("---" for _ in cleaned) + " |")
    return "\n".join(md_rows)
