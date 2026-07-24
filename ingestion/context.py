"""NI 43-101 section hierarchy and visual context building."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from chapter_index import detect_item_from_text
from routing_guide import get_item_title
from ingestion.models import DocumentContext, ElementRecord

_CAPTION_RE = re.compile(r"(?i)^\s*(figure|fig\.?|table|tbl\.?)\s*[\d\.\-]+")


def _is_title(el: ElementRecord) -> bool:
    return el.category == "Title"


def _is_caption(el: ElementRecord) -> bool:
    if el.category == "Caption":
        return True
    if el.category in {"NarrativeText", "Title"} and _CAPTION_RE.match(el.text or ""):
        return True
    return False


def annotate_hierarchy(elements: List[ElementRecord]) -> List[ElementRecord]:
    """Populate ni_item, section_title, section_path, captions, and surrounding text."""
    section_stack: List[str] = []
    current_item = 0
    current_title = ""

    # First pass: titles / NI items
    for el in elements:
        if _is_title(el) or (el.category == "NarrativeText" and len(el.text) < 120):
            detected = detect_item_from_text(el.text)
            if detected:
                current_item, title = detected
                current_title = title
                section_stack = [f"Item {current_item}", title]
            elif _is_title(el) and el.text.strip():
                # Nested section under current item
                depth = el.category_depth if el.category_depth is not None else min(2, len(section_stack))
                while len(section_stack) > max(1, depth):
                    section_stack.pop()
                if section_stack and section_stack[-1] == el.text.strip():
                    pass
                else:
                    section_stack.append(el.text.strip()[:120])
                if current_item and not current_title:
                    current_title = get_item_title(current_item)

        el.ni_item = current_item
        el.section_title = current_title or (section_stack[-1] if section_stack else "")
        el.section_path = list(section_stack)

    # Second pass: captions + neighboring narrative
    n = len(elements)
    for i, el in enumerate(elements):
        if el.category not in {"Image", "Table", "Figure"}:
            continue

        # Caption: prefer next Caption, else previous, else nearby matching text
        caption = ""
        for j in range(i + 1, min(i + 4, n)):
            if _is_caption(elements[j]):
                caption = elements[j].text.strip()
                break
        if not caption:
            for j in range(i - 1, max(-1, i - 4), -1):
                if _is_caption(elements[j]):
                    caption = elements[j].text.strip()
                    break
        el.caption = caption

        preceding: List[str] = []
        for j in range(i - 1, max(-1, i - 6), -1):
            if elements[j].category in {"NarrativeText", "ListItem", "Title"} and elements[j].text.strip():
                preceding.append(elements[j].text.strip())
            if sum(len(p) for p in preceding) > 600:
                break
        preceding.reverse()
        el.preceding_text = "\n".join(preceding)[:800]

        following: List[str] = []
        for j in range(i + 1, min(i + 6, n)):
            if elements[j].category in {"NarrativeText", "ListItem", "Title"} and elements[j].text.strip():
                following.append(elements[j].text.strip())
            if sum(len(p) for p in following) > 600:
                break
        el.following_text = "\n".join(following)[:800]

    return elements


def build_visual_context(el: ElementRecord, task: Optional[str] = None) -> DocumentContext:
    return DocumentContext(
        report_name=el.source_file,
        page_number=el.page_number,
        ni_item=el.ni_item,
        section_title=el.section_title,
        section_path=list(el.section_path),
        caption=el.caption,
        preceding_text=el.preceding_text,
        following_text=el.following_text,
        table_html=el.text_as_html or None,
        task=task or "Classify and analyse the attached visual.",
    )


def needs_table_validation(el: ElementRecord) -> bool:
    """Heuristic: only send important / suspicious tables to Bedrock."""
    if el.category != "Table":
        return False
    html = el.text_as_html or ""
    text = el.text or ""
    if not html and not text:
        return True
    if html.count("<tr") != html.count("</tr>") and html:
        return True
    # Important NI topics
    blob = f"{el.caption} {el.section_title} {text[:500]}".lower()
    keywords = (
        "resource", "reserve", "capex", "opex", "npv", "irr", "production",
        "recovery", "metallurg", "tonnage", "grade", "cut-off", "cutoff",
    )
    if any(k in blob for k in keywords):
        return True
    # Merged header hint
    if "rowspan" in html.lower() or "colspan" in html.lower():
        return True
    return False
