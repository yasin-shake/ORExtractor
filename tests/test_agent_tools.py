"""Unit tests for LangGraph agent tools (no LLM required)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_chat import (
    AgentRunContext,
    _parse_item_list,
    _tool_find_peer_reports,
    _tool_get_routing_playbook,
    _tool_route_question,
    build_agent_tools,
)
from benchmark import _coerce_metadata_str, find_peer_reports
from rag_app import Settings


def _settings() -> Settings:
    return Settings(
        openai_api_key="x",
        openai_base_url=None,
        embed_model="text-embedding-3-small",
        aws_region="us-east-1",
        bedrock_model_id="test",
        knowledge_dir=Path("knowledge"),
        extra_pdf_dirs=[],
        chroma_dir=Path(".chroma_db"),
        collection_name="test",
        chunk_size=1400,
        chunk_overlap=150,
        embed_batch_size=64,
        upsert_batch_size=24,
        top_k=8,
        extracted_dir=Path("extracted_data"),
        extract_top_k=12,
    )


def test_parse_item_list():
    assert _parse_item_list("10, 14, 12") == [10, 14, 12]
    assert _parse_item_list("invalid") == []


def test_route_question_tool():
    ctx = AgentRunContext(
        settings=_settings(),
        vectorstore=MagicMock(),
        question="Are QAQC results acceptable?",
    )
    out = json.loads(_tool_route_question(ctx))
    assert 11 in out["primary_items"]
    assert 12 in out["primary_items"]
    assert ctx.routed_primary == out["primary_items"]


def test_routing_playbook_tool():
    ctx = AgentRunContext(
        settings=_settings(),
        vectorstore=MagicMock(),
        question="Is drill spacing adequate?",
    )
    _tool_route_question(ctx)
    text = _tool_get_routing_playbook(ctx, "10,14")
    assert "Item 10" in text or "Item 14" in text
    assert ctx.playbook_text


def test_build_agent_tools_count():
    ctx = AgentRunContext(
        settings=_settings(),
        vectorstore=MagicMock(),
        question="test",
    )
    tools = build_agent_tools(ctx)
    names = {t.name for t in tools}
    assert names == {
        "route_question",
        "search_by_items",
        "get_extraction",
        "find_peer_reports",
        "benchmark_field",
        "get_routing_playbook",
    }


def test_coerce_metadata_str_from_mining_method_dict():
    assert _coerce_metadata_str({"method": "Underground"}) == "Underground"
    assert _coerce_metadata_str({"method": None}) is None
    assert _coerce_metadata_str("  Open Pit  ") == "Open Pit"


def test_find_peer_reports_with_dict_mining_method(monkeypatch):
    settings = _settings()
    target = {
        "source_file": "target.pdf",
        "mining_method": {"method": "Underground", "strip_ratio": "2:1"},
        "property_info": {"country": "Canada", "commodities": ["Cu"]},
    }
    peer = {
        "source_file": "peer.pdf",
        "mining_method": {"method": "Underground"},
        "property_info": {"country": "Canada", "commodities": ["Cu"]},
    }

    monkeypatch.setattr(
        "benchmark.list_extractions",
        lambda _settings: [target, peer],
    )
    monkeypatch.setattr(
        "benchmark.load_extraction",
        lambda _settings, filename: target if filename == "target.pdf" else None,
    )

    peers = find_peer_reports(settings, target_filename="target.pdf", limit=4)
    assert peers
    assert peers[0]["source_file"] == "peer.pdf"


def test_find_peer_reports_tool_with_dict_mining_method(monkeypatch):
    settings = _settings()
    target = {
        "source_file": "target.pdf",
        "mining_method": {"method": "Underground"},
        "property_info": {"country": "Canada", "commodities": ["Cu"]},
    }
    peer = {
        "source_file": "peer.pdf",
        "mining_method": {"method": "Underground"},
        "property_info": {"country": "Canada", "commodities": ["Cu"]},
    }

    monkeypatch.setattr(
        "benchmark.list_extractions",
        lambda _settings: [target, peer],
    )
    monkeypatch.setattr(
        "extractor.load_extraction",
        lambda _settings, filename: target if filename == "target.pdf" else None,
    )

    ctx = AgentRunContext(
        settings=settings,
        vectorstore=MagicMock(),
        question="benchmark vs peers",
        pdf_filter=["target.pdf"],
    )
    out = json.loads(_tool_find_peer_reports(ctx))
    assert out["peer_count"] >= 1
