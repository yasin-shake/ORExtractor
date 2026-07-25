"""Chroma index info and export endpoints."""

from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException

from api_routers._deps import (
    IngestBusyError,
    run_archive_only,
    settings_or_503,
    vectorstore_or_503,
    zip_response,
)

router = APIRouter(tags=["chroma"])


@router.get("/api/chroma/info", summary="Describe the persistent Chroma index")
def chroma_info():
    settings = settings_or_503()
    count = 0
    try:
        count = int(vectorstore_or_503()._collection.count())
    except Exception:
        pass
    return {
        "collection_name": settings.collection_name,
        "persist_directory": str(settings.chroma_dir),
        "embedding_provider": getattr(
            settings, "resolved_embedding_provider", settings.embedding_provider
        ),
        "embedding_model": getattr(
            settings, "resolved_embedding_model", settings.embed_model
        ),
        "embedding_signature": getattr(
            settings, "resolved_embedding_signature", None
        ),
        "vector_count": count,
    }


@router.get("/api/chroma/export", summary="Download the current Chroma index as ZIP")
async def export_chroma():
    try:
        archive_path = await anyio.to_thread.run_sync(run_archive_only)
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return zip_response(archive_path)
