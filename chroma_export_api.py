from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import anyio
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

import api as base_api
from rag_app import get_vectorstore, ingest


app = base_api.app

_INGEST_LOCK = threading.Lock()
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "2048")) * 1024 * 1024


class IngestBusyError(RuntimeError):
    pass


def _settings_or_503():
    if base_api._settings is None:
        raise HTTPException(status_code=503, detail="API startup has not completed.")
    return base_api._settings


def _safe_pdf_name(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    if not name or name in {".", ".."} or not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are accepted: {filename!r}")
    return name


async def _save_upload(upload: UploadFile, destination: Path) -> int:
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


def _archive_chroma(source_files: List[str]) -> Path:
    settings = _settings_or_503()
    if not settings.chroma_dir.exists():
        raise FileNotFoundError("Chroma directory does not exist. Ingest at least one document first.")

    staging_dir = Path(tempfile.mkdtemp(prefix="orextractor-chroma-stage-"))
    archive_base = Path(tempfile.gettempdir()) / f"orextractor-chroma-{uuid.uuid4().hex}"
    archive_path = archive_base.with_suffix(".zip")

    try:
        shutil.copytree(settings.chroma_dir, staging_dir / "chroma_db")
        count = 0
        try:
            count = int(base_api._vectorstore._collection.count())
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


def _remove_archive(path: Path) -> None:
    path.unlink(missing_ok=True)


def _run_ingest_and_archive(rebuild: bool, source_files: List[str]) -> Path:
    if not _INGEST_LOCK.acquire(blocking=False):
        raise IngestBusyError("Another ingestion or export is already running.")
    try:
        settings = _settings_or_503()
        ingest(settings, rebuild=rebuild)
        base_api._vectorstore = get_vectorstore(settings, base_api._embedder)
        return _archive_chroma(source_files)
    finally:
        _INGEST_LOCK.release()


def _run_archive_only() -> Path:
    if not _INGEST_LOCK.acquire(blocking=False):
        raise IngestBusyError("Another ingestion or export is already running.")
    try:
        settings = _settings_or_503()
        source_files = sorted(p.name for p in settings.knowledge_dir.glob("*.pdf"))
        return _archive_chroma(source_files)
    finally:
        _INGEST_LOCK.release()


def _zip_response(archive_path: Path, ingested_count: int | None = None) -> FileResponse:
    settings = _settings_or_503()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    headers = {"X-Chroma-Collection": settings.collection_name}
    if ingested_count is not None:
        headers["X-Ingested-Files"] = str(ingested_count)
    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=f"orextractor-chromadb-{timestamp}.zip",
        headers=headers,
        background=BackgroundTask(_remove_archive, archive_path),
    )


@app.get("/api/chroma/info", summary="Describe the persistent Chroma index")
def chroma_info():
    settings = _settings_or_503()
    count = 0
    try:
        count = int(base_api._vectorstore._collection.count())
    except Exception:
        pass
    return {
        "collection_name": settings.collection_name,
        "persist_directory": str(settings.chroma_dir),
        "embedding_model": settings.embed_model,
        "vector_count": count,
    }


@app.get("/api/chroma/export", summary="Download the current Chroma index as ZIP")
async def export_chroma():
    try:
        archive_path = await anyio.to_thread.run_sync(_run_archive_only)
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _zip_response(archive_path)


@app.post(
    "/api/ingest/export",
    summary="Upload PDFs, ingest them, and download the resulting Chroma index",
)
async def ingest_and_export(
    files: List[UploadFile] = File(...),
    rebuild: bool = Form(False),
):
    settings = _settings_or_503()
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for upload in files:
        name = _safe_pdf_name(upload.filename)
        destination = settings.knowledge_dir / name
        await _save_upload(upload, destination)
        saved.append(name)

    try:
        archive_path = await anyio.to_thread.run_sync(
            _run_ingest_and_archive, rebuild, saved
        )
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _zip_response(archive_path, ingested_count=len(saved))
