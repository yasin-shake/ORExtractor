"""Structured extraction and spatial model endpoints."""

from __future__ import annotations

from typing import List

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api_routers._deps import llm_or_503, settings_or_503, vectorstore_or_503
from extractor import extract_report, list_extractions, load_extraction
from rag_app import _index_is_empty, save_extraction
from schemas import NI43101Report

router = APIRouter(tags=["reports"])


class ExtractRequest(BaseModel):
    filename: str


@router.get("/api/spatial-models", summary="List available 3D geological model files")
def list_spatial_models():
    """Return id/label/file metadata for every HTML model in the spatial_data directory."""
    import re as _re

    models = []
    settings = settings_or_503()
    spatial_dir = settings.spatial_dir
    if spatial_dir.exists():
        for p in sorted(spatial_dir.glob("*.html")):
            stem = p.stem
            label = _re.sub(r"[-_]+", " ", stem).strip()
            model_id = _re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
            models.append({"id": model_id, "label": label, "file": f"spatial_data/{p.name}"})
    return {"models": models}


@router.get("/api/reports", summary="List structured report extractions")
def list_reports():
    """Return all saved structured extractions from the extracted_data directory."""
    return {"reports": list_extractions(settings_or_503())}


@router.get(
    "/api/reports/{filename}",
    response_model=NI43101Report,
    summary="Get a structured report extraction",
)
def get_report(filename: str):
    """Return the structured extraction for a single report by source filename."""
    data = load_extraction(settings_or_503(), filename)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No extraction found for {filename!r}. Run extraction first.",
        )
    return NI43101Report(**data)


@router.post(
    "/api/extract",
    response_model=NI43101Report,
    summary="Run structured extraction on an ingested report",
)
async def extract_endpoint(req: ExtractRequest):
    """
    Run structured NI 43-101 extraction on an already-ingested PDF and persist
    the result to the extracted_data directory.
    """
    settings = settings_or_503()
    vectorstore = vectorstore_or_503()
    llm = llm_or_503()
    if _index_is_empty(vectorstore):
        raise HTTPException(status_code=400, detail="Vector index is empty. Ingest first.")
    report = await anyio.to_thread.run_sync(
        lambda: extract_report(settings, vectorstore, llm, req.filename)
    )
    save_extraction(settings, report)
    return report


@router.post("/api/extract/all", summary="Run structured extraction on all reports")
async def extract_all_endpoint():
    """Run structured extraction across every ingested PDF and persist the results."""
    settings = settings_or_503()
    vectorstore = vectorstore_or_503()
    llm = llm_or_503()
    if _index_is_empty(vectorstore):
        raise HTTPException(status_code=400, detail="Vector index is empty. Ingest first.")

    def _run() -> List[str]:
        from extractor import extract_all

        processed: List[str] = []
        for filename, report in extract_all(settings, vectorstore, llm):
            save_extraction(settings, report)
            processed.append(filename)
        return processed

    processed = await anyio.to_thread.run_sync(_run)
    return {"status": "extracted", "files": processed}


@router.post("/api/reindex-chapters", status_code=202, summary="Rebuild NI Item chapter tags")
def reindex_chapters_endpoint():
    """Re-parse PDFs and patch chunk metadata with NI Item numbers without re-embedding."""
    from rag_app import reindex_chapters

    settings = settings_or_503()
    vectorstore = vectorstore_or_503()
    reindex_chapters(settings, vectorstore)
    return {"status": "reindexed"}
