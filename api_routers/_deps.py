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
from typing import Any, List, Optional

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


def run_ingest(rebuild: bool = False) -> Any:
    """Run ingest under the process lock; refresh the live vectorstore handle."""
    try_acquire_ingest_lock()
    try:
        settings = settings_or_503()
        result = ingest(settings, rebuild=rebuild)
        api_state.vectorstore = get_vectorstore(settings, embedder_or_503())
        return result
    finally:
        release_ingest_lock()


def archive_chroma(source_files: List[str]) -> Path:
    settings = settings_or_503()
    if not settings.chroma_dir.exists():
        raise FileNotFoundError("Chroma directory does not exist. Ingest at least one document first.")

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
            "embedding_model": settings.embed_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "document_count": len(source_files),
            "documents": source_files,
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


def run_ingest_and_archive(rebuild: bool, source_files: List[str]) -> Path:
    try_acquire_ingest_lock()
    try:
        settings = settings_or_503()
        ingest(settings, rebuild=rebuild)
        api_state.vectorstore = get_vectorstore(settings, embedder_or_503())
        return archive_chroma(source_files)
    finally:
        release_ingest_lock()


def run_archive_only() -> Path:
    try_acquire_ingest_lock()
    try:
        settings = settings_or_503()
        source_files = sorted(p.name for p in settings.knowledge_dir.glob("*.pdf"))
        return archive_chroma(source_files)
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
