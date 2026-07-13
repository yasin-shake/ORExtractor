import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

_HERE = Path(__file__).parent

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent_chat import AgentChatResult, ChatTurn, agent_chat_enabled, run_agent_chat, stream_agent_chat, _llm_chunk_text
from extractor import extract_report, list_extractions, load_extraction
from rag_app import (
    Settings,
    _index_is_empty,
    _is_short_greeting_or_thanks,
    build_chat_prompt,
    get_chat_model,
    get_embedder,
    get_vectorstore,
    ingest,
    iter_pdf_paths,
    load_settings,
    query_context,
    reindex_chapters,
    save_extraction,
)
from schemas import NI43101Report

_settings: Optional[Settings] = None
_embedder = None
_vectorstore = None
_llm = None

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(api_key: str = Security(_api_key_header)) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settings, _embedder, _vectorstore, _llm
    _settings = load_settings()
    _embedder = get_embedder(_settings)
    _vectorstore = get_vectorstore(_settings, _embedder)
    _llm = get_chat_model(_settings)
    yield


app = FastAPI(
    title="NI 43-101 RAG & Extraction API",
    version="1.0.0",
    description=(
        "REST API for ingesting NI 43-101 mineral project reports, answering questions "
        "via RAG, and extracting structured project data."
    ),
    lifespan=lifespan,
    dependencies=[Depends(_verify_api_key)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/dashboard", include_in_schema=False, dependencies=[])
def serve_dashboard():
    return FileResponse(_HERE / "dashboard.html")


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


@app.get("/api/documents", summary="List ingested documents")
def list_documents():
    """Return the filenames of all PDFs currently in the knowledge directory."""
    try:
        docs = sorted(p.name for p in iter_pdf_paths(_settings.knowledge_dir, _settings.extra_pdf_dirs))
    except FileNotFoundError:
        docs = []
    return {"documents": docs}


@app.post("/api/ingest", status_code=202, summary="Upload and ingest PDF files")
async def upload_and_ingest(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files, save them to the knowledge directory,
    and upsert their chunks into the vector store.
    """
    global _vectorstore
    _settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for file in files:
        name = file.filename or ""
        if not name.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are accepted: {name!r}",
            )
        dest = _settings.knowledge_dir / name
        dest.write_bytes(await file.read())
        saved.append(name)

    await anyio.to_thread.run_sync(lambda: ingest(_settings, rebuild=False))
    _vectorstore = get_vectorstore(_settings, _embedder)
    return {"status": "ingested", "files": saved}


@app.post("/api/ingest/rebuild", status_code=202, summary="Rebuild the entire vector index")
def rebuild_index():
    """
    Delete the existing vector index and rebuild it from scratch using all PDFs
    currently in the knowledge directory. Use after changing chunk or embedding settings.
    """
    global _vectorstore
    ingest(_settings, rebuild=True)
    _vectorstore = get_vectorstore(_settings, _embedder)
    return {"status": "rebuilt"}


@app.delete("/api/documents/{filename}", summary="Delete a document and rebuild index")
def delete_document(filename: str):
    """Remove a PDF from the knowledge directory and rebuild the vector index."""
    global _vectorstore
    path = _settings.knowledge_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {filename!r}")
    path.unlink()
    ingest(_settings, rebuild=True)
    _vectorstore = get_vectorstore(_settings, _embedder)
    return {"status": "deleted", "file": filename}


# ---------------------------------------------------------------------------
# Structured extraction endpoints
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    filename: str


@app.get("/api/spatial-models", summary="List available 3D geological model files")
def list_spatial_models():
    """Return id/label/file metadata for every HTML model in the spatial_data directory."""
    import re as _re
    models = []
    spatial_dir = _settings.spatial_dir
    if spatial_dir.exists():
        for p in sorted(spatial_dir.glob("*.html")):
            stem = p.stem
            label = _re.sub(r"[-_]+", " ", stem).strip()
            model_id = _re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
            models.append({"id": model_id, "label": label, "file": f"spatial_data/{p.name}"})
    return {"models": models}


@app.get("/api/reports", summary="List structured report extractions")
def list_reports():
    """Return all saved structured extractions from the extracted_data directory."""
    return {"reports": list_extractions(_settings)}


@app.get(
    "/api/reports/{filename}",
    response_model=NI43101Report,
    summary="Get a structured report extraction",
)
def get_report(filename: str):
    """Return the structured extraction for a single report by source filename."""
    data = load_extraction(_settings, filename)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No extraction found for {filename!r}. Run extraction first.",
        )
    return NI43101Report(**data)


@app.post(
    "/api/extract",
    response_model=NI43101Report,
    summary="Run structured extraction on an ingested report",
)
async def extract_endpoint(req: ExtractRequest):
    """
    Run structured NI 43-101 extraction on an already-ingested PDF and persist
    the result to the extracted_data directory.
    """
    if _index_is_empty(_vectorstore):
        raise HTTPException(status_code=400, detail="Vector index is empty. Ingest first.")
    report = await anyio.to_thread.run_sync(
        lambda: extract_report(_settings, _vectorstore, _llm, req.filename)
    )
    save_extraction(_settings, report)
    return report


@app.post("/api/extract/all", summary="Run structured extraction on all reports")
async def extract_all_endpoint():
    """Run structured extraction across every ingested PDF and persist the results."""
    if _index_is_empty(_vectorstore):
        raise HTTPException(status_code=400, detail="Vector index is empty. Ingest first.")

    def _run() -> List[str]:
        from extractor import extract_all

        processed: List[str] = []
        for filename, report in extract_all(_settings, _vectorstore, _llm):
            save_extraction(_settings, report)
            processed.append(filename)
        return processed

    processed = await anyio.to_thread.run_sync(_run)
    return {"status": "extracted", "files": processed}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    pdf_filter: Optional[List[str]] = None
    history: Optional[List[ChatTurn]] = None


class SourceRef(BaseModel):
    file: str
    page: Any
    chunk: Any
    ni_item: Any = 0
    section_title: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]
    routed_items: List[int] = []
    cross_check_items: List[int] = []
    flags: List[str] = []
    peer_summary: Optional[str] = None
    assessment: Optional[str] = None
    tool_calls: List[str] = []


def _chat_response_from_agent(result: AgentChatResult) -> ChatResponse:
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceRef(
                file=s.get("file", "unknown"),
                page=s.get("page", "?"),
                chunk=s.get("chunk", "?"),
                ni_item=s.get("ni_item", 0),
                section_title=s.get("section_title", ""),
            )
            for s in result.sources
        ],
        routed_items=result.routed_items,
        cross_check_items=result.cross_check_items,
        flags=result.flags,
        peer_summary=result.peer_summary,
        assessment=result.assessment,
        tool_calls=result.tool_calls,
    )


@app.post("/api/chat", response_model=ChatResponse, summary="Ask a question")
def chat_endpoint(req: ChatRequest):
    """
    Answer a question using chapter-directed agentic retrieval from NI 43-101 reports.
    Optionally restrict retrieval to a subset of files via `pdf_filter`.
    Send `history` for multi-turn context (client-side memory).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if _is_short_greeting_or_thanks(req.question):
        return ChatResponse(
            answer=(
                "I answer NI 43-101 due diligence questions using chapter-directed retrieval. "
                "Ask about resources, QAQC, cut-off grades, economics, red flags, or peer benchmarks."
            ),
            sources=[],
        )

    if agent_chat_enabled():
        result = run_agent_chat(
            _settings,
            _vectorstore,
            _llm,
            req.question,
            pdf_filter=req.pdf_filter,
            history=req.history,
        )
        if not result.answer:
            return ChatResponse(
                answer="I could not find relevant context in the indexed reports.",
                sources=[],
            )
        return _chat_response_from_agent(result)

    history_pairs = None
    if req.history:
        from agent_chat import _history_pairs
        history_pairs = _history_pairs(req.history)

    context, metadatas = query_context(
        _vectorstore, req.question, _settings.top_k, req.pdf_filter
    )
    if not context:
        return ChatResponse(
            answer="I could not find relevant context in the indexed reports.",
            sources=[],
        )

    prompt = build_chat_prompt(req.question, context, history=history_pairs)
    response = _llm.invoke([HumanMessage(content=prompt)])
    answer = str(response.content)

    seen: set = set()
    sources: List[SourceRef] = []
    for meta in metadatas:
        key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceRef(
                file=meta.get("source", "unknown"),
                page=meta.get("page", "?"),
                chunk=meta.get("chunk", "?"),
                ni_item=meta.get("ni_item", 0),
                section_title=meta.get("section_title", ""),
            )
        )

    return ChatResponse(answer=answer, sources=sources)


def _stream_chat_events(req: ChatRequest):
    import json

    if not req.question.strip():
        yield f"data: {json.dumps({'type': 'error', 'message': 'Question cannot be empty.'})}\n\n"
        return

    if _is_short_greeting_or_thanks(req.question):
        answer = (
            "I answer NI 43-101 due diligence questions using chapter-directed retrieval. "
            "Ask about resources, QAQC, cut-off grades, economics, red flags, or peer benchmarks."
        )
        yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'flags': [], 'assessment': None})}\n\n"
        return

    if agent_chat_enabled():
        for event in stream_agent_chat(
            _settings,
            _vectorstore,
            _llm,
            req.question,
            pdf_filter=req.pdf_filter,
            history=req.history,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
        return

    # Non-agent fallback: stream simple RAG response in one chunk
    from agent_chat import _history_pairs

    history_pairs = _history_pairs(req.history) if req.history else None
    context, metadatas = query_context(
        _vectorstore, req.question, _settings.top_k, req.pdf_filter
    )
    if not context:
        msg = "I could not find relevant context in the indexed reports."
        yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'answer': msg})}\n\n"
        return

    prompt = build_chat_prompt(req.question, context, history=history_pairs)
    parts: List[str] = []
    for chunk in _llm.stream([HumanMessage(content=prompt)]):
        token = _llm_chunk_text(chunk.content)
        if token:
            parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=True)}\n\n"

    answer = "".join(parts)
    seen: set = set()
    sources: List[dict] = []
    for meta in metadatas:
        key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "file": meta.get("source", "unknown"),
            "page": meta.get("page", "?"),
            "chunk": meta.get("chunk", "?"),
            "ni_item": meta.get("ni_item", 0),
            "section_title": meta.get("section_title", ""),
        })
    yield f"data: {json.dumps({'type': 'meta', 'sources': sources}, ensure_ascii=True)}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'answer': answer}, ensure_ascii=True)}\n\n"


@app.post("/api/chat/stream", summary="Ask a question (streaming SSE)")
def chat_stream_endpoint(req: ChatRequest):
    """
    Stream a chat response as Server-Sent Events.
    Events: status, meta, token, done, error.
    """
    return StreamingResponse(
        _stream_chat_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reindex-chapters", status_code=202, summary="Rebuild NI Item chapter tags")
def reindex_chapters_endpoint():
    """Re-parse PDFs and patch chunk metadata with NI Item numbers without re-embedding."""
    global _vectorstore
    reindex_chapters(_settings, _vectorstore)
    return {"status": "reindexed"}
