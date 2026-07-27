"""Shared API dependencies: settings accessors, upload helpers, ingest lock."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import api_state
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from rag_app import get_vectorstore, ingest

_INGEST_LOCK = threading.Lock()
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024


class IngestBusyError(RuntimeError):
    pass


class IngestionFailedError(RuntimeError):
    pass


def settings_or_503():
    if api_state.settings is None:
        raise HTTPException(status_code=503, detail="API startup has not completed.")
    return api_state.settings


def vectorstore_or_503():
    if api_state.vectorstore is None:
        raise HTTPException(status_code=503, detail="API startup has not completed.")
    return api_state.vectorstore


def llm_or_503():
    if api_state.llm is None:
        raise HTTPException(status_code=503, detail="API startup has not completed.")
    return api_state.llm


def embedder_or_503():
    if api_state.embedder is None:
        raise HTTPException(status_code=503, detail="API startup has not completed.")
    return api_state.embedder


def safe_pdf_name(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    if not name or name in {".", ".."} or not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are accepted: {filename!r}")
    return name


async def save_upload(upload: UploadFile, destination: Path) -> int:
    temp_path = destination.with_suffix(destination.suffix + ".uploading")
    size = 0
    try:
        with temp_path.open("wb") as output:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{destination.name!r} exceeds MAX_UPLOAD_MB="
                            f"{_MAX_UPLOAD_BYTES // (1024 * 1024)}."
                        ),
                    )
                output.write(chunk)
        temp_path.replace(destination)
        return size
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def try_acquire_ingest_lock() -> None:
    if not _INGEST_LOCK.acquire(blocking=False):
        raise IngestBusyError("Another ingestion or export is already running.")


def release_ingest_lock() -> None:
    _INGEST_LOCK.release()


def _run_ingest_unlocked(
    rebuild: bool = False,
    *,
    only_file: Optional[str] = None,
    parser: Optional[str] = None,
    fallback_enabled: Optional[bool] = None,
) -> Any:
    """Run ingest while the caller holds the process lock."""
    from copy import copy

    base_settings = settings_or_503()
    settings = copy(base_settings)
    if parser:
        settings.parser_primary = parser
        settings.ingestion_backend = parser
    if fallback_enabled is not None:
        settings.parser_fallback_enabled = fallback_enabled
    result = ingest(
        settings,
        rebuild=rebuild,
        only_file=only_file,
        backend=parser,
    )
    api_state.vectorstore = get_vectorstore(base_settings, embedder_or_503())
    return result


def run_ingest(
    rebuild: bool = False,
    *,
    only_file: Optional[str] = None,
    parser: Optional[str] = None,
    fallback_enabled: Optional[bool] = None,
) -> Any:
    """Run ingest under the process lock; refresh the live vectorstore handle."""
    try_acquire_ingest_lock()
    try:
        return _run_ingest_unlocked(
            rebuild=rebuild,
            only_file=only_file,
            parser=parser,
            fallback_enabled=fallback_enabled,
        )
    finally:
        release_ingest_lock()


def create_upload_staging_dir() -> Path:
    settings = settings_or_503()
    staging_root = settings.knowledge_dir.resolve().parent
    staging_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix="orextractor-upload-",
            dir=staging_root,
        )
    )


def _remove_upload_staging_dir(staging_dir: Path) -> None:
    settings = settings_or_503()
    resolved = staging_dir.resolve()
    expected_parent = settings.knowledge_dir.resolve().parent
    if (
        resolved.parent != expected_parent
        or not resolved.name.startswith("orextractor-upload-")
    ):
        raise ValueError(f"Refusing to remove unmanaged staging directory: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def _rollback_installed_uploads(
    installed: list[tuple[Path, Path | None]],
) -> None:
    for destination, backup in reversed(installed):
        destination.unlink(missing_ok=True)
        if backup is not None and backup.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup.replace(destination)


def _install_staged_uploads(
    staging_dir: Path,
    source_files: List[str],
) -> list[tuple[Path, Path | None]]:
    settings = settings_or_503()
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = staging_dir / ".backups"
    installed: list[tuple[Path, Path | None]] = []
    try:
        for source_file in source_files:
            name = safe_pdf_name(source_file)
            staged = staging_dir / name
            if not staged.is_file():
                raise FileNotFoundError(f"Staged upload is missing: {name}")
            destination = settings.knowledge_dir / name
            backup = None
            if destination.exists():
                if not destination.is_file():
                    raise IsADirectoryError(destination)
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup = backup_dir / name
                destination.replace(backup)
            installed.append((destination, backup))
            staged.replace(destination)
        return installed
    except Exception:
        _rollback_installed_uploads(installed)
        raise


def _run_with_staged_uploads(
    staging_dir: Path,
    source_files: List[str],
    operation: Callable[[], Any],
    finalize: Callable[[Any], Any] | None = None,
) -> Any:
    try:
        try_acquire_ingest_lock()
        try:
            installed = _install_staged_uploads(staging_dir, source_files)
            try:
                result = operation()
            except Exception:
                _rollback_installed_uploads(installed)
                raise

            failure = _ingestion_failure_message(result)
            if failure:
                successful = {
                    str(source).replace("\\", "/").casefold()
                    for source in (getattr(result, "files", None) or [])
                }
                failed_uploads = [
                    record
                    for record in installed
                    if record[0].name.casefold() not in successful
                ]
                _rollback_installed_uploads(failed_uploads)

            if finalize is not None:
                return finalize(result)
            return result
        finally:
            release_ingest_lock()
    finally:
        _remove_upload_staging_dir(staging_dir)


def run_staged_ingest(
    staging_dir: Path,
    source_files: List[str],
    *,
    parser: Optional[str] = None,
    fallback_enabled: Optional[bool] = None,
) -> Any:
    return _run_with_staged_uploads(
        staging_dir,
        source_files,
        lambda: _run_ingest_unlocked(
            rebuild=False,
            parser=parser,
            fallback_enabled=fallback_enabled,
        ),
    )


def _ingestion_failure_message(result: Any) -> str | None:
    status = getattr(result, "status", None)
    errors = getattr(result, "errors", None) or []
    if status not in {"failed", "completed_with_errors"} and not errors:
        return None
    messages = [
        str(getattr(error, "message", error))
        for error in errors
    ]
    return "; ".join(messages) or f"Ingestion ended with status {status!r}."


def _ensure_ingestion_succeeded(result: Any) -> None:
    failure = _ingestion_failure_message(result)
    if failure:
        raise IngestionFailedError(failure)


def _indexed_source_files() -> List[str]:
    collection = getattr(vectorstore_or_503(), "_collection", None)
    if collection is None:
        raise RuntimeError("Vector store does not expose its Chroma collection.")
    sources: set[str] = set()
    offset = 0
    page_size = 1000
    while True:
        page = collection.get(
            include=["metadatas"],
            limit=page_size,
            offset=offset,
        )
        metadatas = page.get("metadatas") or []
        for metadata in metadatas:
            source = (metadata or {}).get("source")
            if source:
                sources.add(str(source))
        if len(metadatas) < page_size:
            break
        offset += len(metadatas)
    return sorted(sources)


def archive_chroma(_source_files: Optional[List[str]] = None) -> Path:
    settings = settings_or_503()
    if not settings.chroma_dir.exists():
        raise FileNotFoundError("Chroma directory does not exist. Ingest at least one document first.")
    indexed_sources = _indexed_source_files()

    staging_dir = Path(tempfile.mkdtemp(prefix="orextractor-chroma-stage-"))
    archive_base = Path(tempfile.gettempdir()) / f"orextractor-chroma-{uuid.uuid4().hex}"
    archive_path = archive_base.with_suffix(".zip")

    try:
        shutil.copytree(settings.chroma_dir, staging_dir / "chroma_db")
        count = 0
        try:
            count = int(api_state.vectorstore._collection.count())
        except Exception:
            pass

        manifest = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "collection_name": settings.collection_name,
            "embedding_provider": getattr(
                settings,
                "resolved_embedding_provider",
                settings.embedding_provider,
            ),
            "embedding_model": getattr(
                settings, "resolved_embedding_model", settings.embed_model
            ),
            "embedding_signature": getattr(
                settings, "resolved_embedding_signature", None
            ),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "document_count": len(indexed_sources),
            "documents": indexed_sources,
            "vector_count": count,
            "restore_directory_name": "chroma_db",
            "notes": (
                "Use the same ChromaDB/LangChain versions and the same embedding model "
                "when opening this persistent index."
            ),
        }
        (staging_dir / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        shutil.make_archive(str(archive_base), "zip", root_dir=staging_dir)
        return archive_path
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def remove_archive(path: Path) -> None:
    path.unlink(missing_ok=True)


def run_ingest_and_archive(rebuild: bool, _source_files: List[str]) -> Path:
    try_acquire_ingest_lock()
    try:
        result = _run_ingest_unlocked(rebuild=rebuild)
        _ensure_ingestion_succeeded(result)
        return archive_chroma()
    finally:
        release_ingest_lock()


def run_staged_ingest_and_archive(
    staging_dir: Path,
    source_files: List[str],
    *,
    rebuild: bool,
) -> Path:
    def finalize(result: Any) -> Path:
        _ensure_ingestion_succeeded(result)
        return archive_chroma()

    return _run_with_staged_uploads(
        staging_dir,
        source_files,
        lambda: _run_ingest_unlocked(rebuild=rebuild),
        finalize,
    )


def delete_document_and_rebuild(relative_path: Path) -> Any:
    settings = settings_or_503()
    knowledge_root = settings.knowledge_dir.resolve()
    target = (knowledge_root / relative_path).resolve()
    try:
        target.relative_to(knowledge_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe document path: {relative_path}") from exc

    quarantine = create_upload_staging_dir()
    try:
        try_acquire_ingest_lock()
        try:
            if not target.is_file():
                raise FileNotFoundError(target)
            quarantined = quarantine / relative_path
            quarantined.parent.mkdir(parents=True, exist_ok=True)
            target.replace(quarantined)
            try:
                result = _run_ingest_unlocked(rebuild=True)
                _ensure_ingestion_succeeded(result)
                return result
            except Exception:
                target.parent.mkdir(parents=True, exist_ok=True)
                quarantined.replace(target)
                raise
        finally:
            release_ingest_lock()
    finally:
        _remove_upload_staging_dir(quarantine)


def run_archive_only() -> Path:
    try_acquire_ingest_lock()
    try:
        return archive_chroma()
    finally:
        release_ingest_lock()


def zip_response(archive_path: Path, ingested_count: Optional[int] = None) -> FileResponse:
    settings = settings_or_503()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    headers = {"X-Chroma-Collection": settings.collection_name}
    if ingested_count is not None:
        headers["X-Ingested-Files"] = str(ingested_count)
    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=f"orextractor-chromadb-{timestamp}.zip",
        headers=headers,
        background=BackgroundTask(remove_archive, archive_path),
    )
