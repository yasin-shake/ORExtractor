from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import agent_chat
from agent_chat import AgentRunContext
from rag_app import query_by_items


class _Collection:
    def count(self):
        return 1


class _Store:
    _collection = _Collection()

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.filters = []

    def similarity_search_with_score(self, question, k, filter=None):
        self.filters.append(filter)
        if self.error:
            raise self.error
        return self.results


def test_item_query_does_not_retry_without_item_filter_after_error():
    store = _Store(error=RuntimeError("filtered query failed"))

    with pytest.raises(RuntimeError, match="filtered query failed"):
        query_by_items(store, "resource estimate", [14], 3, ["r.pdf"])

    assert store.filters == [
        {"$and": [{"source": "r.pdf"}, {"ni_item": 14}]}
    ]


def test_item_query_rejects_out_of_scope_metadata():
    store = _Store(
        results=[
            (
                Document(
                    page_content="Economic analysis only",
                    metadata={
                        "source": "r.pdf",
                        "page": 9,
                        "chunk": 1,
                        "ni_item": 22,
                    },
                ),
                0.1,
            )
        ]
    )

    context, metadata = query_by_items(
        store,
        "resource estimate",
        [14],
        3,
        ["r.pdf"],
    )

    assert context == ""
    assert metadata == []


def test_agent_item_tool_does_not_fall_back_to_unscoped_context(monkeypatch):
    monkeypatch.setattr(agent_chat, "query_by_items", lambda *args, **kwargs: ("", []))
    monkeypatch.setattr(
        agent_chat,
        "query_context",
        lambda *args, **kwargs: pytest.fail("unscoped fallback must not run"),
    )
    context = AgentRunContext(
        settings=SimpleNamespace(top_k=8),
        vectorstore=object(),
        question="What is the resource estimate?",
        pdf_filter=["r.pdf"],
    )

    payload = agent_chat._tool_search_by_items(context, "14")

    assert "No matching chunks found." in payload
    assert context.context_blocks == []
