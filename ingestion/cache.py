"""Content-addressed caches for partition and enrichment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Optional  # noqa: F401 — Any used in should_skip_pdf

from ingestion.models import (
    NORMALIZER_VERSION,
    PIPELINE_VERSION,
    VISUAL_PROMPT_VERSION,
    VISUAL_SCHEMA_VERSION,
    ElementRecord,
    ParserResult,
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


def runtime_package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parser_policy_signature(settings) -> dict:
    """Configuration and package revisions that affect parser selection."""
    primary = str(
        getattr(settings, "force_parser", "")
        or getattr(settings, "parser_primary", "")
        or getattr(settings, "ingestion_backend", "docling")
    ).lower()
    return {
        "primary": primary,
        "fallback": str(getattr(settings, "parser_fallback", "mineru") or ""),
        "fallback_enabled": bool(
            getattr(settings, "parser_fallback_enabled", True)
        ),
        "docling_version": runtime_package_version("docling"),
        "mineru_version": runtime_package_version("mineru"),
        "normalizer_version": NORMALIZER_VERSION,
        "docling_options": {
            "do_ocr": getattr(settings, "docling_do_ocr", True),
            "do_table_structure": getattr(
                settings, "docling_do_table_structure", True
            ),
            "table_mode": getattr(settings, "docling_table_mode", "accurate"),
            "generate_page_images": getattr(
                settings, "docling_generate_page_images", True
            ),
            "generate_picture_images": getattr(
                settings, "docling_generate_picture_images", True
            ),
            "model_artifact_revision": getattr(
                settings, "docling_model_artifact_revision", ""
            ),
        },
        "quality_policy": {
            "min_text_page_coverage": getattr(
                settings, "parser_min_text_page_coverage", 0.90
            ),
            "max_empty_page_ratio": getattr(
                settings, "parser_max_empty_page_ratio", 0.10
            ),
            "max_replacement_char_ratio": getattr(
                settings, "parser_max_replacement_char_ratio", 0.01
            ),
            "min_table_valid_ratio": getattr(
                settings, "parser_min_table_valid_ratio", 0.80
            ),
            "require_picture_crops": getattr(
                settings, "parser_require_picture_crops", False
            ),
        },
    }


def _parser_signature(pdf_path: Path, parser) -> dict:
    signature = {
        "source_sha256": file_sha256(pdf_path),
    }
    cache_signature = getattr(parser, "cache_signature", None)
    if callable(cache_signature):
        candidate = cache_signature()
        if isinstance(candidate, dict):
            signature["parser_signature"] = candidate
    return signature


def load_parser_cache(
    artifact_dir: Path,
    pdf_path: Path,
    settings,
    parser,
) -> Optional[ParserResult]:
    """Load a full routed parser result when its source/configuration matches."""
    meta_path = artifact_dir / "parser_cache.json"
    result_path = artifact_dir / "parser_result.json"
    if not meta_path.exists() or not result_path.exists():
        return None
    try:
        if json.loads(meta_path.read_text(encoding="utf-8")) != _parser_signature(
            pdf_path, parser
        ):
            return None
        result = ParserResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        for element in result.elements:
            if element.image_path and not Path(element.image_path).exists():
                return None
        return result
    except Exception:
        return None


def save_parser_cache(
    artifact_dir: Path,
    pdf_path: Path,
    settings,
    parser,
    result: ParserResult,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "parser_result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    (artifact_dir / "parser_cache.json").write_text(
        json.dumps(
            _parser_signature(pdf_path, parser),
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
            f"{el.parser or 'unknown'}:{el.parser_version}",
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
            f"{el.parser or 'unknown'}:{el.parser_version}",
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
    # Legacy fingerprints do not identify their vector space, so they cannot
    # safely skip ingestion after embedding-provider changes.
    if isinstance(entry, str):
        return False
    if not isinstance(entry, dict):
        return False
    if entry.get("source_sha256") != file_sha256(pdf_path):
        return False
    if entry.get("parser_policy") != parser_policy_signature(settings):
        return False
    if entry.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if entry.get("visual_prompt_version") != VISUAL_PROMPT_VERSION:
        return False
    if entry.get("visual_schema_version") != VISUAL_SCHEMA_VERSION:
        return False
    if entry.get("visual_model_id") != settings.bedrock_visual_model_id:
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
    resolved_signature = getattr(
        settings, "resolved_embedding_signature", None
    )
    if not isinstance(resolved_signature, dict):
        return False
    if entry.get("embedding_signature") != resolved_signature:
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
    visual_enrichment_enabled: bool = True,
    parser_result: ParserResult,
) -> dict:
    entry = {
        "source_sha256": file_sha256(pdf_path),
        "fingerprint": fingerprint_legacy(pdf_path),
        "pipeline_version": PIPELINE_VERSION,
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
        "embedding_signature": dict(
            getattr(settings, "resolved_embedding_signature", {})
        ),
        "element_count": element_count,
        "visual_count": visual_count,
        "table_count": table_count,
        "indexed_chunk_count": indexed_chunk_count,
        "failed_element_ids": failed_element_ids,
    }
    entry.update(
        {
            "parser_policy": parser_policy_signature(settings),
            "primary_parser": parser_result.metadata.get(
                "primary_parser", parser_result.parser
            ),
            "primary_parser_version": parser_result.metadata.get(
                "primary_parser_version", parser_result.parser_version
            ),
            "selected_parser": parser_result.parser,
            "selected_parser_version": parser_result.parser_version,
            "fallback_enabled": bool(
                getattr(settings, "parser_fallback_enabled", True)
            ),
            "fallback_used": parser_result.fallback.used,
            "fallback_reason_codes": parser_result.fallback.reasons,
            "parser_quality": parser_result.quality.model_dump(mode="json"),
        }
    )
    return entry
