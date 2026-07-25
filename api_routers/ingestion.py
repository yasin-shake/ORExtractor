"""Ingestion upload, rebuild, and ingest+export endpoints."""

from __future__ import annotations

from typing import Any, List

import anyio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api_routers._deps import (
    IngestBusyError,
    run_ingest,
    run_ingest_and_archive,
    safe_pdf_name,
    save_upload,
    settings_or_503,
    zip_response,
)

router = APIRouter(tags=["ingestion"])


def _ingest_payload(saved: List[str], result: Any, status: str) -> dict:
    payload: dict = {"status": status, "files": saved}
    if result is not None and hasattr(result, "model_dump"):
        data = result.model_dump()
        for field in ("reports", "errors", "metrics"):
            if field in data:
                payload[field] = data[field]
        payload["status"] = data.get("status", payload["status"])
        if not saved and data.get("files"):
            payload["files"] = data["files"]
    elif isinstance(result, dict):
        payload.update({k: v for k, v in result.items() if k != "files"})
    return payload


@router.post("/api/ingest", status_code=202, summary="Upload and ingest PDF files")
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    parser: str | None = Form(None),
    fallback_enabled: bool | None = Form(None),
):
    """
    Upload one or more PDF files, save them to the knowledge directory,
    and upsert their chunks into the vector store.
    """
    settings = settings_or_503()
    if parser not in {None, "docling", "mineru"}:
        raise HTTPException(status_code=422, detail="Unsupported parser")
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for upload in files:
        name = safe_pdf_name(upload.filename)
        destination = settings.knowledge_dir / name
        await save_upload(upload, destination)
        saved.append(name)

    try:
        result = await anyio.to_thread.run_sync(
            lambda: run_ingest(
                rebuild=False,
                parser=parser,
                fallback_enabled=fallback_enabled,
            )
        )
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _ingest_payload(saved, result, status="ingested")


@router.post("/api/ingest/rebuild", status_code=202, summary="Rebuild the entire vector index")
async def rebuild_index(
    parser: str | None = None,
    fallback_enabled: bool | None = None,
):
    """
    Delete the existing vector index and rebuild it from scratch using all PDFs
    currently in the knowledge directory. Use after changing chunk or embedding settings.
    """
    try:
        if parser not in {None, "docling", "mineru"}:
            raise HTTPException(status_code=422, detail="Unsupported parser")
        result = await anyio.to_thread.run_sync(
            lambda: run_ingest(
                rebuild=True,
                parser=parser,
                fallback_enabled=fallback_enabled,
            )
        )
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _ingest_payload([], result, status="rebuilt")


@router.post(
    "/api/ingest/export",
    summary="Upload PDFs, ingest them, and download the resulting Chroma index",
)
async def ingest_and_export(
    files: List[UploadFile] = File(...),
    rebuild: bool = Form(False),
):
    settings = settings_or_503()
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for upload in files:
        name = safe_pdf_name(upload.filename)
        destination = settings.knowledge_dir / name
        await save_upload(upload, destination)
        saved.append(name)

    try:
        archive_path = await anyio.to_thread.run_sync(
            run_ingest_and_archive, rebuild, saved
        )
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return zip_response(archive_path, ingested_count=len(saved))
