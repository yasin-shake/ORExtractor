"""DoclingDocument to canonical element conversion."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from ingestion.models import ElementRecord


_CATEGORY_MAP = {
    "section_header": "Title",
    "title": "Title",
    "text": "NarrativeText",
    "paragraph": "NarrativeText",
    "list_item": "ListItem",
    "table": "Table",
    "picture": "Image",
    "formula": "Formula",
    "code": "CodeSnippet",
    "caption": "Caption",
    "footnote": "Footnote",
    "page_header": "Header",
    "page_footer": "Footer",
}


def _value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").lower()


def _provenance(item: Any) -> tuple[int, dict[str, Any] | None]:
    prov = getattr(item, "prov", None) or []
    first = prov[0] if prov else None
    page = int(getattr(first, "page_no", 1) or 1)
    bbox = getattr(first, "bbox", None)
    if bbox is None:
        return page, None
    if hasattr(bbox, "model_dump"):
        return page, bbox.model_dump()
    keys = ("l", "t", "r", "b", "coord_origin")
    return page, {
        key: getattr(bbox, key)
        for key in keys
        if getattr(bbox, key, None) is not None
    }


def _page_dimensions(
    document: Any,
    page_number: int,
) -> tuple[float, float] | None:
    pages = getattr(document, "pages", None) or {}
    if isinstance(pages, dict):
        page = pages.get(page_number) or pages.get(str(page_number))
    else:
        try:
            page = pages[page_number - 1]
        except (IndexError, KeyError, TypeError):
            page = None
    size = getattr(page, "size", None)
    if size is None and isinstance(page, dict):
        size = page.get("size")
    if isinstance(size, dict):
        width = size.get("width")
        height = size.get("height")
    else:
        width = getattr(size, "width", None)
        height = getattr(size, "height", None)
    try:
        parsed_width = float(width)
        parsed_height = float(height)
    except (TypeError, ValueError):
        return None
    if parsed_width <= 0 or parsed_height <= 0:
        return None
    return parsed_width, parsed_height


def _iter_items(document: Any) -> Iterable[tuple[Any, int]]:
    iterator = getattr(document, "iterate_items", None)
    if callable(iterator):
        for entry in iterator():
            if isinstance(entry, tuple):
                yield entry[0], int(entry[1] or 0)
            else:
                yield entry, 0
        return
    for collection_name in ("texts", "tables", "pictures", "groups"):
        for item in getattr(document, collection_name, None) or []:
            yield item, 0


def _stable_id(
    source_file: str,
    parser_id: str,
    category: str,
    page: int,
    text: str,
) -> str:
    seed = f"{source_file}|{parser_id}|{category}|{page}|{text[:256]}"
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:24]


def _export(item: Any, method: str, document: Any) -> str:
    fn = getattr(item, method, None)
    if not callable(fn):
        return ""
    for args in ((document,), ()):
        try:
            value = fn(*args)
            if value is not None:
                return str(value)
        except (TypeError, ValueError, RuntimeError):
            continue
    return ""


_INLINE_IMAGE_MARKDOWN = re.compile(
    r"!\[[^\]]*\]\(data:image/[^)]*\)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_embedded_image_markup(value: str) -> str:
    return _INLINE_IMAGE_MARKDOWN.sub("", value).strip()


def _persist_image(item: Any, document: Any, image_dir: Path, element_id: str) -> str | None:
    get_image = getattr(item, "get_image", None)
    if not callable(get_image):
        return None
    try:
        image = get_image(document)
    except (TypeError, ValueError, RuntimeError, AttributeError):
        return None
    if image is None:
        return None
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{element_id}.png"
    try:
        image.save(target, format="PNG")
    except (AttributeError, OSError, ValueError):
        return None
    return str(target)


def normalize_docling_document(
    document: Any,
    *,
    source_file: str,
    artifact_dir: Path,
    parser_version: str,
) -> list[ElementRecord]:
    """Convert Docling items while preserving hierarchy and provenance."""
    records: list[ElementRecord] = []
    image_dir = artifact_dir / "images"

    for ordinal, (item, level) in enumerate(_iter_items(document)):
        label = _value(getattr(item, "label", item.__class__.__name__))
        category = _CATEGORY_MAP.get(label, label.replace("_", " ").title() or "Unknown")
        text = str(getattr(item, "text", "") or "")
        page, coordinates = _provenance(item)
        raw_id = str(
            getattr(item, "self_ref", None)
            or getattr(item, "id", None)
            or f"{label}-{ordinal}"
        )
        element_id = _stable_id(source_file, raw_id, category, page, text)
        markdown = _export(item, "export_to_markdown", document)
        html = _export(item, "export_to_html", document)
        # Docling exports pictures as inline base64 data URIs. The crop is
        # persisted separately below, so retaining the URI would inflate parser
        # caches and accidentally send megabytes of image bytes to a text
        # embedding model.
        if category == "Image":
            markdown = _strip_embedded_image_markup(markdown)
            if "data:image/" in html.lower():
                html = ""
        image_path = None
        if category in {"Image", "Table"}:
            image_path = _persist_image(item, document, image_dir, element_id)

        parent = getattr(item, "parent", None)
        parent_id = str(
            getattr(parent, "cref", None)
            or getattr(parent, "self_ref", None)
            or ""
        ) or None
        confidence = getattr(item, "confidence", None)
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None

        metadata = {"docling_label": label, "docling_ordinal": ordinal}
        page_dimensions = _page_dimensions(document, page)
        if page_dimensions is not None:
            metadata.update(
                {
                    "page_width": page_dimensions[0],
                    "page_height": page_dimensions[1],
                }
            )

        records.append(
            ElementRecord(
                element_id=element_id,
                source_file=source_file,
                category=category,
                text=text or markdown,
                text_as_html=html,
                text_as_markdown=markdown,
                page_number=max(1, page),
                parent_id=parent_id,
                category_depth=level,
                coordinates=coordinates,
                image_path=image_path,
                parser="docling",
                parser_version=parser_version,
                parser_element_id=raw_id,
                parser_confidence=confidence,
                metadata=metadata,
            )
        )

    return records
