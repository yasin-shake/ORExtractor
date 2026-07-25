"""MinerU content-list to canonical element conversion."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from ingestion.models import ElementRecord


_CATEGORY_MAP = {
    "text": "NarrativeText",
    "title": "Title",
    "header": "Header",
    "footer": "Footer",
    "list": "ListItem",
    "list_item": "ListItem",
    "table": "Table",
    "image": "Image",
    "figure": "Image",
    "equation": "Formula",
    "interline_equation": "Formula",
}


def _stable_id(source_file: str, ordinal: int, item: dict[str, Any]) -> str:
    seed = (
        f"{source_file}|{ordinal}|{item.get('type', '')}|"
        f"{item.get('page_idx', item.get('page_number', 0))}|"
        f"{item.get('text', '')[:256]}"
    )
    return hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:24]


def normalize_mineru_content(
    content: Iterable[dict[str, Any]],
    *,
    source_file: str,
    artifact_dir: Path,
    parser_version: str,
) -> list[ElementRecord]:
    """Normalize MinerU's JSON content list; artifacts remain in its output tree."""
    records: list[ElementRecord] = []
    for ordinal, item in enumerate(content):
        item_type = str(item.get("type") or item.get("category") or "text").lower()
        category = _CATEGORY_MAP.get(item_type, item_type.replace("_", " ").title())
        text = str(
            item.get("text")
            or item.get("content")
            or item.get("table_caption")
            or item.get("image_caption")
            or ""
        )
        markdown = str(item.get("markdown") or item.get("table_body") or "")
        html = str(item.get("html") or item.get("table_html") or "")
        page = int(item.get("page_number", int(item.get("page_idx", 0)) + 1) or 1)
        element_id = _stable_id(source_file, ordinal, item)
        relative_image = item.get("img_path") or item.get("image_path")
        image_path = None
        if relative_image:
            candidate = Path(str(relative_image))
            if not candidate.is_absolute():
                candidate = artifact_dir / candidate
            image_path = str(candidate)

        confidence = item.get("score", item.get("confidence"))
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        coordinates = item.get("bbox")
        if isinstance(coordinates, (list, tuple)):
            coordinates = {"bbox": list(coordinates)}
        elif coordinates is not None and not isinstance(coordinates, dict):
            coordinates = {"repr": str(coordinates)}

        records.append(
            ElementRecord(
                element_id=element_id,
                source_file=source_file,
                category=category,
                text=text or markdown,
                text_as_html=html,
                text_as_markdown=markdown,
                page_number=max(1, page),
                category_depth=item.get("level"),
                coordinates=coordinates,
                image_path=image_path,
                parser="mineru",
                parser_version=parser_version,
                parser_element_id=str(item.get("id") or ordinal),
                parser_confidence=confidence,
                metadata={
                    "mineru_type": item_type,
                    "mineru_ordinal": ordinal,
                },
            )
        )
    return records
