"""Content-addressed caches for partition and enrichment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
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


def runtime_partitioner_version() -> str:
    """Include the installed parser version in cache and manifest invalidation."""
    try:
        package_version = importlib.metadata.version("unstructured")
    except importlib.metadata.PackageNotFoundError:
        package_version = "not-installed"
    return f"{PARTITIONER_VERSION}:unstructured={package_version}"


def _partition_signature(pdf_path: Path, settings, partitioner) -> dict:
    return {
        "source_sha256": file_sha256(pdf_path),
        "partitioner": getattr(partitioner, "provider_name", "unstructured-local"),
        "partitioner_version": getattr(
            partitioner, "version", runtime_partitioner_version()
        ),
        "partition_strategy": getattr(
            partitioner,
            "strategy",
            getattr(settings, "unstructured_strategy", "hi_res"),
        ),
    }


def load_partition_cache(
    artifact_dir: Path,
    pdf_path: Path,
    settings,
    partitioner,
) -> Optional[list[ElementRecord]]:
    """Load normalized elements when source and partition configuration match."""
    meta_path = artifact_dir / "partition_cache.json"
    elements_path = artifact_dir / "normalized_elements.json"
    if not meta_path.exists() or not elements_path.exists():
        return None
    try:
        if json.loads(meta_path.read_text(encoding="utf-8")) != _partition_signature(
            pdf_path, settings, partitioner
        ):
            return None
        payload = json.loads(elements_path.read_text(encoding="utf-8"))
        elements = [ElementRecord.model_validate(item) for item in payload]
        for element in elements:
            if element.image_path and not Path(element.image_path).exists():
                return None
        return elements
    except Exception:
        return None


def save_partition_cache(
    artifact_dir: Path,
    pdf_path: Path,
    settings,
    partitioner,
    elements: list[ElementRecord],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "normalized_elements.json").write_text(
        json.dumps(
            [element.model_dump(mode="json") for element in elements],
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "partition_cache.json").write_text(
        json.dumps(
            _partition_signature(pdf_path, settings, partitioner),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


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
            str(el.metadata.get("source_sha256") or el.source_file),
            file_sha256(Path(el.image_path)) if el.image_path and Path(el.image_path).exists() else _sha256_bytes(image_bytes),
            _sha256_text(json.dumps(context, sort_keys=True, ensure_ascii=True)),
            model_id,
            VISUAL_PROMPT_VERSION,
            VISUAL_SCHEMA_VERSION,
            runtime_partitioner_version(),
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
            str(el.metadata.get("source_sha256") or el.source_file),
            _sha256_text(html),
            _sha256_bytes(image_bytes) if image_bytes else "",
            _sha256_text(json.dumps(context, sort_keys=True, ensure_ascii=True)),
            model_id,
            VISUAL_PROMPT_VERSION,
            VISUAL_SCHEMA_VERSION,
            runtime_partitioner_version(),
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
    expected_provider = (
        f"unstructured-{getattr(settings, 'unstructured_provider', 'local')}"
    )
    if entry.get("partitioner") != expected_provider:
        return False
    if entry.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if entry.get("partitioner_version") != runtime_partitioner_version():
        return False
    if entry.get("visual_prompt_version") != VISUAL_PROMPT_VERSION:
        return False
    if entry.get("visual_schema_version") != VISUAL_SCHEMA_VERSION:
        return False
    if entry.get("visual_model_id") != settings.bedrock_visual_model_id:
        return False
    if entry.get("partition_strategy") != getattr(
        settings, "unstructured_strategy", "hi_res"
    ):
        return False
    if entry.get("visual_enrichment_enabled", True) is not True:
        return False
    if entry.get("visual_confidence_threshold") != getattr(
        settings, "bedrock_visual_confidence_threshold", 0.85
    ):
        return False
    if entry.get("visual_reconstruct_charts") != getattr(
        settings, "visual_reconstruct_charts", True
    ):
        return False
    if entry.get("visual_reconstruct_diagrams") != getattr(
        settings, "visual_reconstruct_diagrams", True
    ):
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
    visual_enrichment_enabled: bool = True,
) -> dict:
    return {
        "source_sha256": file_sha256(pdf_path),
        "fingerprint": fingerprint_legacy(pdf_path),
        "pipeline_version": PIPELINE_VERSION,
        "partitioner": partitioner,
        "partitioner_version": runtime_partitioner_version(),
        "partition_strategy": partition_strategy,
        "visual_model_id": settings.bedrock_visual_model_id,
        "visual_prompt_version": VISUAL_PROMPT_VERSION,
        "visual_schema_version": VISUAL_SCHEMA_VERSION,
        "visual_enrichment_enabled": visual_enrichment_enabled,
        "visual_confidence_threshold": getattr(
            settings, "bedrock_visual_confidence_threshold", 0.85
        ),
        "visual_reconstruct_charts": getattr(
            settings, "visual_reconstruct_charts", True
        ),
        "visual_reconstruct_diagrams": getattr(
            settings, "visual_reconstruct_diagrams", True
        ),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embed_model,
        "element_count": element_count,
        "visual_count": visual_count,
        "table_count": table_count,
        "indexed_chunk_count": indexed_chunk_count,
        "failed_element_ids": failed_element_ids,
    }
