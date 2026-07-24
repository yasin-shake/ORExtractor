"""Normalize raw Unstructured elements into ElementRecord and persist artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional

from ingestion.models import ElementRecord

_CATEGORY_MAP = {
    "title": "Title",
    "narrativetext": "NarrativeText",
    "narrative_text": "NarrativeText",
    "listitem": "ListItem",
    "list_item": "ListItem",
    "table": "Table",
    "image": "Image",
    "figure": "Image",
    "figurecaption": "Caption",
    "figure_caption": "Caption",
    "caption": "Caption",
    "pagebreak": "PageBreak",
    "page_break": "PageBreak",
    "header": "Header",
    "footer": "Footer",
    "formulalike": "Formula",
    "formula": "Formula",
}

_DECORATIVE_CATEGORIES = frozenset({"Header", "Footer"})


def normalize_category(raw: str) -> str:
    key = re.sub(r"[\s\-]+", "", (raw or "").strip().lower())
    return _CATEGORY_MAP.get(key, raw or "Unknown")


def stable_element_id(source_file: str, page: int, category: str, index: int, text: str = "") -> str:
    digest = hashlib.md5(f"{source_file}|{page}|{category}|{index}|{text[:80]}".encode("utf-8")).hexdigest()[:10]
    cat = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-") or "el"
    return f"{cat}-{page}-{index}-{digest}"


def _element_attr(el: Any, name: str, default: Any = None) -> Any:
    if hasattr(el, name):
        return getattr(el, name)
    if isinstance(el, dict):
        if name in el:
            return el.get(name, default)
        meta = el.get("metadata")
    else:
        meta = getattr(el, "metadata", None)
    if meta is not None and hasattr(meta, name):
        return getattr(meta, name)
    if isinstance(meta, dict):
        return meta.get(name, default)
    return default


def _page_number(el: Any) -> int:
    page = _element_attr(el, "page_number", None)
    try:
        return int(page) if page is not None else 1
    except (TypeError, ValueError):
        return 1


def _coordinates(el: Any) -> Optional[dict]:
    meta = el.get("metadata") if isinstance(el, dict) else getattr(el, "metadata", None)
    coords = None
    if meta is not None:
        coords = getattr(meta, "coordinates", None) if not isinstance(meta, dict) else meta.get("coordinates")
    if coords is None:
        return None
    if hasattr(coords, "to_dict"):
        try:
            return coords.to_dict()
        except Exception:
            pass
    if isinstance(coords, dict):
        return coords
    try:
        return {"repr": str(coords)}
    except Exception:
        return None


def _text_as_html(el: Any) -> str:
    meta = el.get("metadata") if isinstance(el, dict) else getattr(el, "metadata", None)
    if meta is None:
        return ""
    if isinstance(meta, dict):
        return str(meta.get("text_as_html") or "")
    return str(getattr(meta, "text_as_html", "") or "")


def _extract_image_bytes(el: Any) -> Optional[bytes]:
    meta = el.get("metadata") if isinstance(el, dict) else getattr(el, "metadata", None)
    if meta is None:
        return None
    payload = None
    if isinstance(meta, dict):
        payload = meta.get("image_base64") or meta.get("image_bytes")
    else:
        payload = getattr(meta, "image_base64", None) or getattr(meta, "image_bytes", None)
    if payload is None:
        return None
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            return base64.b64decode(payload)
        except Exception:
            return None
    return None


def _image_size(data: bytes) -> tuple[Optional[int], Optional[int]]:
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _persist_image(data: bytes, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            image.convert("RGB").save(dest, format="PNG")
    except Exception:
        dest.write_bytes(data)
    return dest


def _text_fingerprint(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.md5(collapsed.encode("utf-8")).hexdigest()


def normalize_elements(
    raw_elements: Iterable[Any],
    source_file: str,
    artifact_dir: Path,
) -> List[ElementRecord]:
    """Convert raw Unstructured elements into ElementRecord list with artifacts on disk."""
    figures_dir = artifact_dir / "figures"
    tables_dir = artifact_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    records: List[ElementRecord] = []
    seen_text_fps: dict[str, int] = {}
    seen_image_hashes: set[str] = set()
    category_counters: dict[str, int] = {}

    for el in raw_elements:
        category_raw = getattr(el, "category", None) or _element_attr(el, "type", "Unknown")
        category = normalize_category(str(category_raw))
        text = (getattr(el, "text", None) or "").strip() if not isinstance(el, dict) else str(el.get("text") or "").strip()
        html = _text_as_html(el)
        page = _page_number(el)

        if category in _DECORATIVE_CATEGORIES and len(text) < 80:
            # Keep a lightweight record but mark skip for enrichment.
            pass
        elif not text and not html and category not in {"Image", "PageBreak", "Table"}:
            continue

        category_counters[category] = category_counters.get(category, 0) + 1
        idx = category_counters[category]
        element_id = stable_element_id(source_file, page, category, idx, text)

        image_path = None
        width = height = None
        image_bytes = _extract_image_bytes(el)
        image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
        if image_bytes:
            width, height = _image_size(image_bytes)
            image_dir = tables_dir if category == "Table" else figures_dir
            image_path = str(_persist_image(image_bytes, image_dir / f"{element_id}.png"))

        if category == "Table" and html:
            html_path = tables_dir / f"{element_id}.html"
            html_path.write_text(html, encoding="utf-8")
            if image_bytes is None and text:
                # Persist text snapshot for audit.
                (tables_dir / f"{element_id}.txt").write_text(text, encoding="utf-8")

        fp = _text_fingerprint(text) if text else ""
        is_duplicate = False
        skip_reason = None
        if fp and category in {"Header", "Footer", "Image"} and fp in seen_text_fps:
            is_duplicate = True
            skip_reason = "duplicate_header_footer_or_logo"
        if category == "Image" and image_hash and image_hash in seen_image_hashes:
            is_duplicate = True
            skip_reason = "duplicate_image"
        if fp:
            seen_text_fps[fp] = seen_text_fps.get(fp, 0) + 1
        if image_hash:
            seen_image_hashes.add(image_hash)

        if category in {"Header", "Footer"}:
            skip_reason = skip_reason or "decorative"

        parent_id = _element_attr(el, "parent_id")
        depth = _element_attr(el, "category_depth")

        records.append(
            ElementRecord(
                element_id=element_id,
                source_file=source_file,
                category=category,
                text=text,
                text_as_html=html,
                page_number=page,
                parent_id=str(parent_id) if parent_id is not None else None,
                category_depth=int(depth) if depth is not None else None,
                coordinates=_coordinates(el),
                image_path=image_path,
                image_width=width,
                image_height=height,
                is_duplicate=is_duplicate,
                skip_reason=skip_reason,
                metadata={"raw_category": str(category_raw)},
            )
        )

    manifest_path = artifact_dir / "partition_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_file": source_file,
                "element_count": len(records),
                "categories": category_counters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return records
