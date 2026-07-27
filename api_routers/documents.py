"""Document list and delete endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api_routers._deps import (
    IngestBusyError,
    IngestionFailedError,
    delete_document_and_rebuild,
    safe_pdf_name,
    settings_or_503,
)
from rag_app import iter_pdf_paths, pdf_source_id

router = APIRouter(tags=["documents"])


@router.get("/api/documents", summary="List ingested documents")
def list_documents():
    """Return the filenames of all PDFs currently in the knowledge directory."""
    settings = settings_or_503()
    try:
        docs = sorted(
            pdf_source_id(
                path,
                settings.knowledge_dir,
                settings.extra_pdf_dirs,
            )
            for path in iter_pdf_paths(
                settings.knowledge_dir,
                settings.extra_pdf_dirs,
            )
        )
    except FileNotFoundError:
        docs = []
    return {"documents": docs}


@router.delete("/api/documents/{filename:path}", summary="Delete a document and rebuild index")
def delete_document(filename: str):
    """Remove a PDF from the knowledge directory and rebuild the vector index."""
    settings = settings_or_503()
    normalized = filename.replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(status_code=400, detail="Unsafe document path")
    safe_name = safe_pdf_name(relative.name)
    path = settings.knowledge_dir / relative.parent / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {filename!r}")
    try:
        delete_document_and_rebuild(relative.parent / safe_name)
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IngestionFailedError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {filename!r}",
        ) from exc
    return {"status": "deleted", "file": normalized}
