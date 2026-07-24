"""Content-addressed caches for partition and enrichment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional  # noqa: F401 — Any used in should_skip_pdf

from ingestion.models import (
    PARTITIONER_VERSION,
    PIPELINE_VERSION,
    VISUAL_PROMPT_VERSION,
    VISUAL_SCHEMA_VERSION,
    ElementRecord,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(*parts: str) -> str:
    return _sha256_text("|".join(parts))


class EnrichmentCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.visual_dir = self.root / "visual"
        self.table_dir = self.root / "table"
        self.visual_dir.mkdir(exist_ok=True)
        self.table_dir.mkdir(exist_ok=True)

    def _visual_key(self, el: ElementRecord, context: dict, model_id: str, image_bytes: bytes) -> str:
        return cache_key(
            el.source_file,
            file_sha256(Path(el.image_path)) if el.image_path and Path(el.image_path).exists() else _sha256_bytes(image_bytes),
            _sha256_text(json.dumps(context, sort_keys=True, ensure_ascii=True)),
            model_id,
            VISUAL_PROMPT_VERSION,
            VISUAL_SCHEMA_VERSION,
            PARTITIONER_VERSION,
            PIPELINE_VERSION,
        )

    def get_visual(self, el: ElementRecord, context: dict, model_id: str, image_bytes: bytes) -> Optional[dict]:
        key = self._visual_key(el, context, model_id, image_bytes)
        path = self.visual_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put_visual(self, el: ElementRecord, context: dict, model_id: str, image_bytes: bytes, payload: dict) -> None:
        key = self._visual_key(el, context, model_id, image_bytes)
        path = self.visual_dir / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _table_key(self, el: ElementRecord, context: dict, model_id: str, image_bytes: bytes) -> str:
        html = el.text_as_html or el.text or ""
        return cache_key(
            el.source_file,
            _sha256_text(html),
            _sha256_bytes(image_bytes) if image_bytes else "",
            _sha256_text(json.dumps(context, sort_keys=True, ensure_ascii=True)),
            model_id,
            VISUAL_PROMPT_VERSION,
            VISUAL_SCHEMA_VERSION,
            PIPELINE_VERSION,
        )

    def get_table(self, el: ElementRecord, context: dict, model_id: str, image_bytes: bytes) -> Optional[dict]:
        key = self._table_key(el, context, model_id, image_bytes)
        path = self.table_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def put_table(self, el: ElementRecord, context: dict, model_id: str, image_bytes: bytes, payload: dict) -> None:
        key = self._table_key(el, context, model_id, image_bytes)
        path = self.table_dir / f"{key}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_ingest_manifest(chroma_dir: Path) -> dict:
    path = chroma_dir / "ingest_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_ingest_manifest(chroma_dir: Path, manifest: dict) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / "ingest_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def fingerprint_legacy(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def should_skip_pdf(entry: Any, pdf_path: Path, settings) -> bool:
    """Return True if this PDF can be skipped given current pipeline versions."""
    if entry is None:
        return False
    # Legacy string fingerprint
    if isinstance(entry, str):
        return entry == fingerprint_legacy(pdf_path)
    if not isinstance(entry, dict):
        return False
    if entry.get("source_sha256") != file_sha256(pdf_path):
        return False
    if entry.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if entry.get("partitioner_version") != PARTITIONER_VERSION:
        return False
    if entry.get("visual_prompt_version") != VISUAL_PROMPT_VERSION:
        return False
    if entry.get("visual_schema_version") != VISUAL_SCHEMA_VERSION:
        return False
    if entry.get("visual_model_id") != settings.bedrock_visual_model_id:
        return False
    if entry.get("chunk_size") != settings.chunk_size:
        return False
    if entry.get("chunk_overlap") != settings.chunk_overlap:
        return False
    if entry.get("embedding_model") != settings.embed_model:
        return False
    return True


def build_manifest_entry(
    pdf_path: Path,
    settings,
    *,
    element_count: int,
    visual_count: int,
    table_count: int,
    indexed_chunk_count: int,
    failed_element_ids: list[str],
    partitioner: str,
    partition_strategy: str,
) -> dict:
    return {
        "source_sha256": file_sha256(pdf_path),
        "fingerprint": fingerprint_legacy(pdf_path),
        "pipeline_version": PIPELINE_VERSION,
        "partitioner": partitioner,
        "partitioner_version": PARTITIONER_VERSION,
        "partition_strategy": partition_strategy,
        "visual_model_id": settings.bedrock_visual_model_id,
        "visual_prompt_version": VISUAL_PROMPT_VERSION,
        "visual_schema_version": VISUAL_SCHEMA_VERSION,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embed_model,
        "element_count": element_count,
        "visual_count": visual_count,
        "table_count": table_count,
        "indexed_chunk_count": indexed_chunk_count,
        "failed_element_ids": failed_element_ids,
    }
