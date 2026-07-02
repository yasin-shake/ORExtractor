"""Parse NI 43-101 Form Item headings and tag document chunks by chapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from routing_guide import get_item_title, item_titles

# Match "Item 14", "ITEM 14 - Mineral Resource", "14. Mineral Resource Estimates"
_ITEM_HEADING_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:item\s*)?(\d{1,2})\s*[\.\:\-–—]\s*([^\n]{3,80})"
)
_ITEM_ONLY_RE = re.compile(r"(?i)(?:^|\n)\s*item\s*(\d{1,2})\b")


@dataclass
class ChapterEntry:
    item: int
    title: str
    page_start: int
    page_end: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


def _clean_title(raw: str) -> str:
    t = re.sub(r"\s+", " ", raw.strip())
    t = re.sub(r"[\.\s]+$", "", t)
    return t[:120]


def detect_item_from_text(text: str) -> Optional[Tuple[int, str]]:
    """Return (item_number, title) if text looks like an NI Item heading."""
    m = _ITEM_HEADING_RE.search(text)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 27:
            return num, _clean_title(m.group(2))
    m2 = _ITEM_ONLY_RE.search(text[:200])
    if m2:
        num = int(m2.group(1))
        if 1 <= num <= 27:
            return num, get_item_title(num)
    return None


def build_chapter_index_from_documents(docs: List[Document]) -> List[ChapterEntry]:
    """Scan parsed documents for Item headings and infer page ranges."""
    headings: List[Tuple[int, int, str]] = []  # page, item, title

    for doc in docs:
        page = int(doc.metadata.get("page", 0) or 0)
        detected = detect_item_from_text(doc.page_content)
        if detected:
            item_num, title = detected
            if not headings or headings[-1][1] != item_num or headings[-1][0] != page:
                headings.append((page, item_num, title))

    if not headings:
        return []

    headings.sort(key=lambda x: (x[0], x[1]))
    entries: List[ChapterEntry] = []
    for i, (page, item_num, title) in enumerate(headings):
        page_end = None
        if i + 1 < len(headings):
            page_end = max(page, headings[i + 1][0] - 1)
        entries.append(ChapterEntry(item=item_num, title=title, page_start=page, page_end=page_end))
    return entries


def item_for_page(chapters: List[ChapterEntry], page: int) -> Optional[ChapterEntry]:
    if not chapters or page <= 0:
        return None
    best: Optional[ChapterEntry] = None
    for ch in chapters:
        if ch.page_start <= page:
            if ch.page_end is None or page <= ch.page_end:
                return ch
            best = ch
    return best


def tag_documents_with_items(
    docs: List[Document],
    chapters: List[ChapterEntry],
) -> List[Document]:
    """Add ni_item and section_title metadata to each document chunk."""
    titles = item_titles()
    for doc in docs:
        page = int(doc.metadata.get("page", 0) or 0)
        ch = item_for_page(chapters, page)
        if ch:
            doc.metadata["ni_item"] = ch.item
            doc.metadata["section_title"] = ch.title
        else:
            inline = detect_item_from_text(doc.page_content)
            if inline:
                doc.metadata["ni_item"] = inline[0]
                doc.metadata["section_title"] = inline[1]
            else:
                doc.metadata.setdefault("ni_item", 0)
                doc.metadata.setdefault("section_title", "")
    return docs


def chapter_index_path(extracted_dir: Path, pdf_name: str) -> Path:
    return extracted_dir / f"{Path(pdf_name).stem}_chapters.json"


def save_chapter_index(extracted_dir: Path, pdf_name: str, chapters: List[ChapterEntry]) -> Path:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    path = chapter_index_path(extracted_dir, pdf_name)
    path.write_text(
        json.dumps([c.to_dict() for c in chapters], indent=2),
        encoding="utf-8",
    )
    return path


def load_chapter_index(extracted_dir: Path, pdf_name: str) -> List[ChapterEntry]:
    path = chapter_index_path(extracted_dir, pdf_name)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [ChapterEntry(**e) for e in raw]


def reindex_chapters_for_pdf(
    pdf_path: Path,
    chunk_size: int,
    parse_fn,
    extracted_dir: Path,
) -> Tuple[List[ChapterEntry], List[Document]]:
    """Re-parse PDF, rebuild chapter index and tagged documents."""
    docs = parse_fn(pdf_path, chunk_size)
    chapters = build_chapter_index_from_documents(docs)
    docs = tag_documents_with_items(docs, chapters)
    save_chapter_index(extracted_dir, pdf_path.name, chapters)
    return chapters, docs
