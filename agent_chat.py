"""Agentic chapter-directed chat for NI 43-101 due diligence (LangGraph ReAct agent)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_aws import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from benchmark import _coerce_metadata_str, benchmark_field, find_peer_reports, infer_benchmark_field
from extractor import load_extraction
from rag_app import (
    Settings,
    _MAX_HISTORY_TURNS,
    _history_to_text,
    query_by_items,
    query_context,
    SYSTEM_INSTRUCTION,
)
from routing_guide import get_playbook, resolve_items_for_question

_MAX_TOOL_ROUNDS = 5
_MAX_CONTEXT_CHARS = 14000


class ChatTurn(BaseModel):
    role: str
    content: str


class AgentChatResult(BaseModel):
    answer: str
    sources: List[dict] = Field(default_factory=list)
    routed_items: List[int] = Field(default_factory=list)
    cross_check_items: List[int] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    peer_summary: Optional[str] = None
    assessment: Optional[str] = None
    tool_calls: List[str] = Field(default_factory=list)


@dataclass
class AgentRunContext:
    settings: Settings
    vectorstore: Chroma
    question: str
    pdf_filter: Optional[List[str]] = None
    history: Optional[List[ChatTurn]] = None
    routed_primary: List[int] = field(default_factory=list)
    routed_cross: List[int] = field(default_factory=list)
    benchmark_template: Optional[str] = None
    source_metas: List[dict] = field(default_factory=list)
    peer_summary: Optional[str] = None
    benchmark_flags: List[str] = field(default_factory=list)
    playbook_text: str = ""
    tool_calls: List[str] = field(default_factory=list)
    context_blocks: List[str] = field(default_factory=list)


AGENT_SYSTEM_PROMPT = (
    f"{SYSTEM_INSTRUCTION}\n\n"
    "You are an agentic NI 43-101 due diligence assistant. Follow this workflow:\n"
    "1. Call route_question first to map the query to NI 43-101 Items.\n"
    "2. Call get_routing_playbook for Extract/Compare/Flag checklists.\n"
    "3. Call search_by_items for primary Items, then cross-check Items if any.\n"
    "4. When a target report is scoped (pdf_filter), call get_extraction.\n"
    "5. For peer/comparison/benchmark questions, call find_peer_reports then benchmark_field.\n"
    "6. Synthesize a final answer using ONLY tool outputs.\n\n"
    "Rules:\n"
    "- Cite NI Item numbers and page numbers from search results.\n"
    "- Quote figures exactly with units.\n"
    "- List red flags where playbook evidence supports them.\n"
    "- For Go/No-Go questions, end with: Assessment: [Go|Conditional Go|Further Work|No-Go].\n\n"
    "Format the final answer in GitHub-flavored Markdown:\n"
    "- Use ## and ### for section headings\n"
    "- Use horizontal rules (---) between major sections\n"
    "- Use markdown tables for numeric/tabular data\n"
    "- Use bullet or numbered lists for enumerations\n"
    "- Use **bold** for key terms and figures with units"
)


def _history_pairs(history: Optional[List[ChatTurn]]) -> List[Tuple[str, str]]:
    if not history:
        return []
    pairs: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for turn in history[-(_MAX_HISTORY_TURNS * 2):]:
        if turn.role == "user":
            pending_user = turn.content
        elif turn.role == "assistant" and pending_user is not None:
            pairs.append((pending_user, turn.content))
            pending_user = None
    return pairs[-_MAX_HISTORY_TURNS:]


def _routing_text(ctx: AgentRunContext) -> str:
    parts = [ctx.question]
    pairs = _history_pairs(ctx.history)
    if pairs:
        parts.insert(0, _history_to_text(pairs))
    return "\n\n".join(parts)


def _parse_item_list(items: str) -> List[int]:
    nums: List[int] = []
    for part in re.split(r"[,;\s]+", items.strip()):
        if part.isdigit():
            n = int(part)
            if 1 <= n <= 27:
                nums.append(n)
    return nums


def _record_tool(ctx: AgentRunContext, name: str) -> None:
    ctx.tool_calls.append(name)



def _append_sources(ctx: AgentRunContext, metadatas: List[dict]) -> None:
    ctx.source_metas.extend(metadatas)


def _dedupe_sources(metadatas: List[dict]) -> List[dict]:
    seen: set = set()
    out: List[dict] = []
    for meta in metadatas:
        src = meta.get("source") or meta.get("file") or "unknown"
        page = meta.get("page", "?")
        chunk = meta.get("chunk", "?")
        key = (src, page, chunk)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "file": src,
            "page": page,
            "chunk": chunk,
            "ni_item": meta.get("ni_item", 0),
            "section_title": meta.get("section_title", ""),
        })
    return out


def _playbook_flags(playbook: str, answer: str) -> List[str]:
    flags: List[str] = []
    for line in playbook.splitlines():
        if line.startswith("Flag:"):
            for item in line.replace("Flag:", "").split(";"):
                item = item.strip().rstrip(".")
                if item:
                    flags.append(item)
    answer_lower = answer.lower()
    triggered = []
    for f in flags:
        key = f.split(".")[0].lower()[:30]
        if any(word in answer_lower for word in ("flag", "concern", "risk", "issue")) and key[:8] in answer_lower:
            triggered.append(f)
    return triggered[:8] if triggered else []


def _infer_assessment(question: str, answer: str, flags: List[str]) -> Optional[str]:
    q = question.lower()
    if not any(w in q for w in ("go", "no-go", "no go", "invest", "decision")):
        if "should the project be classified" not in q:
            return None
    if re.search(r"\bno-?go\b", answer, re.I):
        return "No-Go"
    if re.search(r"conditional\s+go", answer, re.I):
        return "Conditional Go"
    if re.search(r"\bfurther\s+work\b", answer, re.I):
        return "Further Work"
    if re.search(r"\bgo\b", answer, re.I) and not re.search(r"no-?go", answer, re.I):
        return "Go"
    if len(flags) >= 4:
        return "No-Go"
    if len(flags) >= 2:
        return "Conditional Go"
    if len(flags) == 1:
        return "Further Work"
    return "Go"


def _tool_route_question(ctx: AgentRunContext) -> str:
    _record_tool(ctx, "route_question")
    routing = resolve_items_for_question(_routing_text(ctx))
    ctx.routed_primary = routing.primary_items
    ctx.routed_cross = routing.cross_check_items
    ctx.benchmark_template = routing.benchmark_template
    return json.dumps({
        "primary_items": routing.primary_items,
        "cross_check_items": routing.cross_check_items,
        "matched_patterns": routing.matched_patterns,
        "benchmark_template": routing.benchmark_template,
        "needs_peer_benchmark": routing.needs_peer_benchmark,
    }, ensure_ascii=True)


def _tool_search_by_items(ctx: AgentRunContext, items: str, query: str = "") -> str:
    _record_tool(ctx, "search_by_items")
    item_nums = _parse_item_list(items)
    if not item_nums:
        item_nums = ctx.routed_primary or [1, 25]
    search_q = (query or ctx.question).strip()
    top_k = max(ctx.settings.top_k, 10)
    context, metadatas = query_by_items(
        ctx.vectorstore, search_q, item_nums, top_k, ctx.pdf_filter
    )
    if not context:
        context, metadatas = query_context(
            ctx.vectorstore, search_q, top_k, ctx.pdf_filter
        )
    _append_sources(ctx, metadatas)
    if context and len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n… [truncated]"
    if context:
        label = f"Items {item_nums}" + (f" ({ctx.pdf_filter[0]})" if ctx.pdf_filter else "")
        ctx.context_blocks.append(f"### Retrieved: {label}\n{context}")
    return json.dumps({
        "items_searched": item_nums,
        "context": context or "No matching chunks found.",
        "source_count": len(metadatas),
    }, ensure_ascii=True)


def _tool_get_extraction(ctx: AgentRunContext, filename: str = "") -> str:
    _record_tool(ctx, "get_extraction")
    fname = filename.strip()
    if not fname and ctx.pdf_filter and len(ctx.pdf_filter) == 1:
        fname = ctx.pdf_filter[0]
    if not fname:
        return json.dumps({"error": "No target report specified. Set pdf_filter or pass filename."})
    data = load_extraction(ctx.settings, fname)
    if not data:
        return json.dumps({"error": f"No extraction found for {fname}. Run extraction first."})
    slim = {k: v for k, v in data.items() if v is not None and v != [] and v != {}}
    text = json.dumps(slim, ensure_ascii=True, indent=2)
    if len(text) > _MAX_CONTEXT_CHARS:
        text = text[:_MAX_CONTEXT_CHARS] + "\n… [truncated]"
    ctx.context_blocks.append(f"### Structured extraction: {fname}\n{text}")
    return text


def _tool_find_peer_reports(
    ctx: AgentRunContext,
    commodity: str = "",
    country: str = "",
    deposit_type: str = "",
    mining_method: str = "",
    study_stage: str = "",
    limit: int = 8,
) -> str:
    _record_tool(ctx, "find_peer_reports")
    target_fn = ctx.pdf_filter[0] if ctx.pdf_filter and len(ctx.pdf_filter) == 1 else None
    target = load_extraction(ctx.settings, target_fn) if target_fn else None

    def _pick(*args: Any, fallback: Optional[Any] = None) -> Optional[str]:
        for a in args:
            picked = _coerce_metadata_str(a)
            if picked:
                return picked
        return _coerce_metadata_str(fallback)

    peers = find_peer_reports(
        ctx.settings,
        target_filename=target_fn,
        commodity=_pick(commodity, (target or {}).get("primary_commodity")),
        country=_pick(country, ((target or {}).get("property_info") or {}).get("country")),
        deposit_type=_pick(
            deposit_type,
            (target or {}).get("deposit_type"),
            ((target or {}).get("geology") or {}).get("deposit_type"),
        ),
        mining_method=_pick(mining_method, (target or {}).get("primary_mining_method"), (target or {}).get("mining_method")),
        study_stage=_pick(study_stage, (target or {}).get("study_stage")),
        limit=min(max(limit, 1), 12),
    )
    summary_lines = []
    for p in peers:
        pi = p.get("property_info") or {}
        summary_lines.append(
            f"{p.get('source_file')}: {pi.get('project_name', '?')}, "
            f"{pi.get('country', '?')}, commodity={p.get('primary_commodity') or pi.get('commodities')}, "
            f"stage={p.get('study_stage') or pi.get('project_stage', '?')}"
        )
    ctx.peer_summary = "\n".join(summary_lines) if summary_lines else "No peer reports matched."
    return json.dumps({
        "peer_count": len(peers),
        "peers": summary_lines,
        "source_files": [p.get("source_file") for p in peers],
    }, ensure_ascii=True)


def _tool_benchmark_field(ctx: AgentRunContext, field: str = "", peer_files: str = "") -> str:
    _record_tool(ctx, "benchmark_field")
    field_name = (field or infer_benchmark_field(ctx.question) or "cutoff_grade").strip()
    target_fn = ctx.pdf_filter[0] if ctx.pdf_filter and len(ctx.pdf_filter) == 1 else None
    target = load_extraction(ctx.settings, target_fn) if target_fn else None

    if peer_files.strip():
        files = [f.strip() for f in peer_files.split(",") if f.strip()]
        all_reports = []
        for fn in files:
            r = load_extraction(ctx.settings, fn)
            if r:
                all_reports.append(r)
        peers = all_reports
    else:
        peers = find_peer_reports(ctx.settings, target_filename=target_fn, limit=8)

    result = benchmark_field(field_name, peers, target)
    ctx.benchmark_flags.extend(result.get("outliers") or [])
    summary = result.get("summary", "")
    ctx.peer_summary = (ctx.peer_summary or "") + "\n" + summary if ctx.peer_summary else summary
    return json.dumps(result, ensure_ascii=True)


def _tool_get_routing_playbook(ctx: AgentRunContext, items: str = "") -> str:
    _record_tool(ctx, "get_routing_playbook")
    item_nums = _parse_item_list(items) if items.strip() else (ctx.routed_primary + ctx.routed_cross)
    if not item_nums:
        item_nums = [1, 25]
    ctx.playbook_text = get_playbook(item_nums)
    return ctx.playbook_text or "No playbook found for the requested Items."


def build_agent_tools(ctx: AgentRunContext) -> List[StructuredTool]:
    """Create LangChain tools bound to a per-request context."""
    return [
        StructuredTool.from_function(
            name="route_question",
            description=(
                "Route the user question to NI 43-101 Form Items (1-27). "
                "Returns primary_items, cross_check_items, and benchmark_template. Call this first."
            ),
            func=lambda: _tool_route_question(ctx),
        ),
        StructuredTool.from_function(
            name="search_by_items",
            description=(
                "Retrieve report text from the vector index scoped to NI Items. "
                "Args: items (comma-separated Item numbers e.g. '10,14'), "
                "query (optional search phrase; defaults to user question)."
            ),
            func=lambda items, query="": _tool_search_by_items(ctx, items, query),
        ),
        StructuredTool.from_function(
            name="get_extraction",
            description=(
                "Load structured JSON extraction for a report (resources, NPV, metadata). "
                "Args: filename (optional; uses scoped target report if omitted)."
            ),
            func=lambda filename="": _tool_get_extraction(ctx, filename),
        ),
        StructuredTool.from_function(
            name="find_peer_reports",
            description=(
                "Find comparable peer reports in the portfolio by commodity, country, deposit type, etc. "
                "Optional args: commodity, country, deposit_type, mining_method, study_stage, limit."
            ),
            func=lambda commodity="", country="", deposit_type="", mining_method="", study_stage="", limit=8: (
                _tool_find_peer_reports(ctx, commodity, country, deposit_type, mining_method, study_stage, limit)
            ),
        ),
        StructuredTool.from_function(
            name="benchmark_field",
            description=(
                "Compare a numeric field across peers (cutoff_grade, post_tax_npv, irr, initial_capex, opex, "
                "recovery_rate, dilution, mining_recovery). "
                "Optional args: field, peer_files (comma-separated source filenames)."
            ),
            func=lambda field="", peer_files="": _tool_benchmark_field(ctx, field, peer_files),
        ),
        StructuredTool.from_function(
            name="get_routing_playbook",
            description=(
                "Return due diligence Extract/Compare/Flag checklist for NI Items. "
                "Args: items (comma-separated; defaults to routed items)."
            ),
            func=lambda items="": _tool_get_routing_playbook(ctx, items),
        ),
    ]


def _build_user_message(ctx: AgentRunContext) -> str:
    scope = ""
    if ctx.pdf_filter:
        scope = f"\n\nScope reports to: {', '.join(ctx.pdf_filter)}"
    hist = _history_pairs(ctx.history)
    if hist:
        return (
            f"Conversation history:\n{_history_to_text(hist)}\n\n"
            f"Current question: {ctx.question}{scope}"
        )
    return f"{ctx.question}{scope}"


def _extract_final_answer(messages: List[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if text.strip():
                return text.strip()
    return "I could not produce an answer from the available context."


def _offline_answer_from_ctx(ctx: AgentRunContext, error: str) -> str:
    """Build a readable summary from tool outputs when the LLM is unavailable."""
    lines = [
        f"*LLM unavailable ({error}). Summary from agent tools:*",
        "",
        f"**Routed Items:** {ctx.routed_primary or '—'}",
        f"**Cross-check Items:** {ctx.routed_cross or '—'}",
    ]
    if ctx.tool_calls:
        lines.append(f"**Tools called:** {' -> '.join(ctx.tool_calls)}")
    if ctx.peer_summary:
        lines.append(f"\n**Peer benchmark:**\n{ctx.peer_summary}")
    if ctx.playbook_text:
        snippet = ctx.playbook_text[:2000]
        if len(ctx.playbook_text) > 2000:
            snippet += "\n…"
        lines.append(f"\n**Playbook excerpt:**\n{snippet}")
    if ctx.source_metas:
        lines.append(f"\n**Sources retrieved:** {len(ctx.source_metas)} chunk(s)")
    lines.append("\nRefresh AWS credentials and retry for a full synthesized answer.")
    return "\n".join(lines)


def prepare_agent_context(
    settings: Settings,
    vectorstore: Chroma,
    question: str,
    pdf_filter: Optional[List[str]] = None,
    history: Optional[List[ChatTurn]] = None,
) -> AgentRunContext:
    """Run the tool chain and return populated context (no LLM call)."""
    ctx = AgentRunContext(
        settings=settings,
        vectorstore=vectorstore,
        question=question,
        pdf_filter=pdf_filter,
        history=history,
    )
    _tool_route_question(ctx)
    _tool_get_routing_playbook(ctx, ",".join(str(i) for i in ctx.routed_primary))
    _tool_search_by_items(ctx, ",".join(str(i) for i in ctx.routed_primary))
    if ctx.routed_cross:
        _tool_search_by_items(ctx, ",".join(str(i) for i in ctx.routed_cross))
    if pdf_filter and len(pdf_filter) == 1:
        _tool_get_extraction(ctx)
    peer_routing = resolve_items_for_question(question)
    if peer_routing.needs_peer_benchmark or peer_routing.benchmark_template:
        _tool_find_peer_reports(ctx)
        _tool_benchmark_field(ctx)
    return ctx


def _llm_chunk_text(content: Any) -> str:
    """Extract plain text from Bedrock/LangChain message chunk content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            val = content.get(key)
            if isinstance(val, str) and val:
                return val
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _build_synthesis_messages(
    ctx: AgentRunContext,
    question: str,
    history: Optional[List[ChatTurn]],
) -> List[Any]:
    """Build LLM messages for streamed synthesis after tools complete."""
    from langchain_core.messages import BaseMessage

    sections: List[str] = []
    if ctx.context_blocks:
        sections.append("\n\n".join(ctx.context_blocks))
    if ctx.playbook_text:
        sections.append(f"### Due diligence playbook\n{ctx.playbook_text}")
    if ctx.peer_summary:
        sections.append(f"### Peer benchmark\n{ctx.peer_summary}")

    body_parts: List[str] = []
    hist = _history_pairs(history)
    if hist:
        body_parts.append(f"Conversation history:\n{_history_to_text(hist)}")
    body_parts.append(f"Current question: {question}")
    if ctx.pdf_filter:
        body_parts.append(f"Report scope: {', '.join(ctx.pdf_filter)}")
    if sections:
        body_parts.append("Retrieved context:\n" + "\n\n".join(sections))
    else:
        body_parts.append("No retrieved context was found.")

    messages: List[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    messages.append(HumanMessage(content="\n\n".join(body_parts)))
    return messages


def _result_from_ctx(ctx: AgentRunContext, question: str, answer: str) -> AgentChatResult:
    flags = _playbook_flags(ctx.playbook_text, answer) + ctx.benchmark_flags
    assessment = _infer_assessment(question, answer, flags)
    if assessment and "Assessment:" not in answer:
        answer = answer.rstrip() + f"\n\n**Assessment:** {assessment}"
    return AgentChatResult(
        answer=answer,
        sources=_dedupe_sources(ctx.source_metas),
        routed_items=ctx.routed_primary,
        cross_check_items=ctx.routed_cross,
        flags=flags[:12],
        peer_summary=ctx.peer_summary,
        assessment=assessment,
        tool_calls=ctx.tool_calls,
    )


def stream_agent_chat(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    question: str,
    pdf_filter: Optional[List[str]] = None,
    history: Optional[List[ChatTurn]] = None,
):
    """Yield SSE-friendly event dicts: status, meta, token, done, error."""
    if os.getenv("AGENT_DRY_RUN", "").strip().lower() in ("1", "true", "yes"):
        result = run_agent_tools_only(settings, vectorstore, question, pdf_filter, history)
        yield {
            "type": "meta",
            "routed_items": result.routed_items,
            "cross_check_items": result.cross_check_items,
            "sources": result.sources,
            "tool_calls": result.tool_calls,
        }
        yield {"type": "token", "content": result.answer}
        yield {
            "type": "done",
            "answer": result.answer,
            "flags": result.flags,
            "peer_summary": result.peer_summary,
            "assessment": result.assessment,
        }
        return

    yield {"type": "status", "message": "Routing question to NI 43-101 Items…"}
    try:
        ctx = prepare_agent_context(settings, vectorstore, question, pdf_filter, history)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    yield {
        "type": "status",
        "message": f"Retrieved context for Items {ctx.routed_primary or '—'}…",
    }
    yield {
        "type": "meta",
        "routed_items": ctx.routed_primary,
        "cross_check_items": ctx.routed_cross,
        "sources": _dedupe_sources(ctx.source_metas),
        "tool_calls": ctx.tool_calls,
        "peer_summary": ctx.peer_summary,
    }

    messages = _build_synthesis_messages(ctx, question, history)
    parts: List[str] = []
    try:
        for chunk in llm.stream(messages):
            token = _llm_chunk_text(chunk.content)
            if not token:
                continue
            parts.append(token)
            yield {"type": "token", "content": token}
    except Exception as exc:
        if parts:
            answer = "".join(parts)
            result = _result_from_ctx(ctx, question, answer)
            yield {
                "type": "done",
                "answer": result.answer,
                "flags": result.flags,
                "peer_summary": result.peer_summary,
                "assessment": result.assessment,
            }
            return
        if ctx.tool_calls:
            answer = _offline_answer_from_ctx(ctx, str(exc))
            yield {"type": "token", "content": answer}
            yield {
                "type": "done",
                "answer": answer,
                "flags": ctx.benchmark_flags[:12],
                "peer_summary": ctx.peer_summary,
                "assessment": None,
            }
            return
        yield {"type": "error", "message": str(exc)}
        return

    answer = "".join(parts)
    result = _result_from_ctx(ctx, question, answer)
    yield {
        "type": "done",
        "answer": result.answer,
        "flags": result.flags,
        "peer_summary": result.peer_summary,
        "assessment": result.assessment,
    }


def run_agent_tools_only(
    settings: Settings,
    vectorstore: Chroma,
    question: str,
    pdf_filter: Optional[List[str]] = None,
    history: Optional[List[ChatTurn]] = None,
) -> AgentChatResult:
    """Run the tool chain without LLM (for testing routing/retrieval/benchmark)."""
    ctx = prepare_agent_context(settings, vectorstore, question, pdf_filter, history)
    answer = _offline_answer_from_ctx(ctx, "dry-run — no LLM call")
    return AgentChatResult(
        answer=answer,
        sources=_dedupe_sources(ctx.source_metas),
        routed_items=ctx.routed_primary,
        cross_check_items=ctx.routed_cross,
        flags=ctx.benchmark_flags[:12],
        peer_summary=ctx.peer_summary,
        assessment=None,
        tool_calls=ctx.tool_calls,
    )


def run_agent_chat(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    question: str,
    pdf_filter: Optional[List[str]] = None,
    history: Optional[List[ChatTurn]] = None,
) -> AgentChatResult:
    """Run LangGraph ReAct agent with chapter-directed tools."""
    if os.getenv("AGENT_DRY_RUN", "").strip().lower() in ("1", "true", "yes"):
        return run_agent_tools_only(settings, vectorstore, question, pdf_filter, history)

    mode = os.getenv("AGENT_MODE", "langgraph").strip().lower()
    if mode == "pipeline":
        return _run_pipeline_chat(settings, vectorstore, llm, question, pdf_filter, history)

    ctx = AgentRunContext(
        settings=settings,
        vectorstore=vectorstore,
        question=question,
        pdf_filter=pdf_filter,
        history=history,
    )
    tools = build_agent_tools(ctx)
    agent = create_react_agent(llm, tools, prompt=SystemMessage(content=AGENT_SYSTEM_PROMPT))

    # recursion_limit: model + tool pairs; ~2 steps per tool call
    recursion_limit = 2 + _MAX_TOOL_ROUNDS * 2
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=_build_user_message(ctx))]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as exc:
        err = str(exc)
        if ctx.tool_calls:
            answer = _offline_answer_from_ctx(ctx, err)
            flags = ctx.benchmark_flags
            return AgentChatResult(
                answer=answer,
                sources=_dedupe_sources(ctx.source_metas),
                routed_items=ctx.routed_primary,
                cross_check_items=ctx.routed_cross,
                flags=flags[:12],
                peer_summary=ctx.peer_summary,
                assessment=None,
                tool_calls=ctx.tool_calls,
            )
        fallback = _run_pipeline_chat(settings, vectorstore, llm, question, pdf_filter, history)
        fallback.answer = (
            f"*Note: LangGraph agent error ({err}); used pipeline fallback.*\n\n{fallback.answer}"
        )
        return fallback

    messages = result.get("messages", [])
    answer = _extract_final_answer(messages)

    flags = _playbook_flags(ctx.playbook_text, answer) + ctx.benchmark_flags
    assessment = _infer_assessment(question, answer, flags)
    if assessment and "Assessment:" not in answer:
        answer = answer.rstrip() + f"\n\n**Assessment:** {assessment}"

    return AgentChatResult(
        answer=answer,
        sources=_dedupe_sources(ctx.source_metas),
        routed_items=ctx.routed_primary,
        cross_check_items=ctx.routed_cross,
        flags=flags[:12],
        peer_summary=ctx.peer_summary,
        assessment=assessment,
        tool_calls=ctx.tool_calls,
    )


def _run_pipeline_chat(
    settings: Settings,
    vectorstore: Chroma,
    llm: ChatBedrockConverse,
    question: str,
    pdf_filter: Optional[List[str]] = None,
    history: Optional[List[ChatTurn]] = None,
) -> AgentChatResult:
    """Deterministic fallback pipeline (pre-LangGraph behaviour)."""
    ctx = prepare_agent_context(settings, vectorstore, question, pdf_filter, history)
    playbook = ctx.playbook_text
    parts = [
        f"=== Routed Items {ctx.routed_primary} / cross-check {ctx.routed_cross} ===",
        f"=== Playbook ===\n{playbook}",
    ]
    if ctx.peer_summary:
        parts.append(f"=== Peer benchmark ===\n{ctx.peer_summary}")

    hist_pairs = _history_pairs(history)
    prompt_payload = {
        "instruction": AGENT_SYSTEM_PROMPT,
        "context": "\n\n".join(parts),
        "question": question,
    }
    if hist_pairs:
        prompt_payload["history"] = _history_to_text(hist_pairs)

    prompt = "Follow the instruction and answer clearly.\n\n" + json.dumps(
        prompt_payload, ensure_ascii=True, indent=2
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = str(response.content)
    flags = _playbook_flags(playbook, answer) + ctx.benchmark_flags
    assessment = _infer_assessment(question, answer, flags)
    if assessment and "Assessment:" not in answer:
        answer = answer.rstrip() + f"\n\n**Assessment:** {assessment}"

    return AgentChatResult(
        answer=answer,
        sources=_dedupe_sources(ctx.source_metas),
        routed_items=ctx.routed_primary,
        cross_check_items=ctx.routed_cross,
        flags=flags[:12],
        peer_summary=ctx.peer_summary,
        assessment=assessment,
        tool_calls=ctx.tool_calls,
    )


def agent_chat_enabled() -> bool:
    return os.getenv("AGENT_CHAT", "1").strip().lower() in ("1", "true", "yes")


def main() -> int:
    """CLI test harness: python agent_chat.py \"Your question\" [--file report.pdf]"""
    import argparse
    from rag_app import get_chat_model, get_embedder, get_vectorstore, load_settings

    parser = argparse.ArgumentParser(description="Test NI 43-101 agent chat")
    parser.add_argument("question", help="Due diligence question")
    parser.add_argument("--file", default=None, help="Scope to one PDF filename")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run tools only (no Bedrock LLM call)",
    )
    args = parser.parse_args()

    settings = load_settings()
    embedder = get_embedder(settings)
    vectorstore = get_vectorstore(settings, embedder)
    llm = get_chat_model(settings)
    pdf_filter = [args.file] if args.file else None

    print(f"AGENT_MODE={os.getenv('AGENT_MODE', 'langgraph')}")
    if args.dry_run:
        os.environ["AGENT_DRY_RUN"] = "1"
    result = run_agent_chat(settings, vectorstore, llm, args.question, pdf_filter=pdf_filter)
    print("\n--- Routed Items ---")
    print(result.routed_items, "cross-check:", result.cross_check_items)
    print("\n--- Tools called ---")
    print(" -> ".join(result.tool_calls) or "(none recorded)")
    if result.assessment:
        print("\n--- Assessment ---")
        print(result.assessment)
    print("\n--- Answer ---\n")
    print(result.answer)
    if result.sources:
        print("\n--- Sources ---")
        for s in result.sources[:8]:
            print(f"  {s['file']} p.{s['page']} Item {s.get('ni_item', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
