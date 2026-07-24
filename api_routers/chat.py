"""Chat and streaming chat endpoints."""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent_chat import (
    AgentChatResult,
    ChatTurn,
    _llm_chunk_text,
    agent_chat_enabled,
    run_agent_chat,
    stream_agent_chat,
)
from api_routers._deps import llm_or_503, settings_or_503, vectorstore_or_503
from rag_app import _is_short_greeting_or_thanks, build_chat_prompt, query_context

router = APIRouter(tags=["chat"])


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


@router.post("/api/chat", response_model=ChatResponse, summary="Ask a question")
def chat_endpoint(req: ChatRequest):
    """
    Answer a question using chapter-directed agentic retrieval from NI 43-101 reports.
    Optionally restrict retrieval to a subset of files via `pdf_filter`.
    Send `history` for multi-turn context (client-side memory).
    """
    settings = settings_or_503()
    vectorstore = vectorstore_or_503()
    llm = llm_or_503()

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
            settings,
            vectorstore,
            llm,
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
        vectorstore, req.question, settings.top_k, req.pdf_filter
    )
    if not context:
        return ChatResponse(
            answer="I could not find relevant context in the indexed reports.",
            sources=[],
        )

    prompt = build_chat_prompt(req.question, context, history=history_pairs)
    response = llm.invoke([HumanMessage(content=prompt)])
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

    settings = settings_or_503()
    vectorstore = vectorstore_or_503()
    llm = llm_or_503()

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
            settings,
            vectorstore,
            llm,
            req.question,
            pdf_filter=req.pdf_filter,
            history=req.history,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
        return

    from agent_chat import _history_pairs

    history_pairs = _history_pairs(req.history) if req.history else None
    context, metadatas = query_context(
        vectorstore, req.question, settings.top_k, req.pdf_filter
    )
    if not context:
        msg = "I could not find relevant context in the indexed reports."
        yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'answer': msg})}\n\n"
        return

    prompt = build_chat_prompt(req.question, context, history=history_pairs)
    parts: List[str] = []
    for chunk in llm.stream([HumanMessage(content=prompt)]):
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


@router.post("/api/chat/stream", summary="Ask a question (streaming SSE)")
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
