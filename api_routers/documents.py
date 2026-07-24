"""Document list and delete endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api_routers._deps import run_ingest, settings_or_503
from rag_app import iter_pdf_paths

router = APIRouter(tags=["documents"])


@router.get("/api/documents", summary="List ingested documents")
def list_documents():
    """Return the filenames of all PDFs currently in the knowledge directory."""
    settings = settings_or_503()
    try:
        docs = sorted(p.name for p in iter_pdf_paths(settings.knowledge_dir, settings.extra_pdf_dirs))
    except FileNotFoundError:
        docs = []
    return {"documents": docs}


@router.delete("/api/documents/{filename}", summary="Delete a document and rebuild index")
def delete_document(filename: str):
    """Remove a PDF from the knowledge directory and rebuild the vector index."""
    from api_routers._deps import IngestBusyError

    settings = settings_or_503()
    path = settings.knowledge_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {filename!r}")
    path.unlink()
    try:
        run_ingest(rebuild=True)
    except IngestBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "deleted", "file": filename}
