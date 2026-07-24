"""LangSmith tracing helpers for ingestion stages."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional


def configure_langsmith(settings) -> None:
    """Enable or disable LangSmith based on settings (content tracing off by default)."""
    if getattr(settings, "langsmith_tracing", False):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = getattr(settings, "langsmith_project", "orextractor-ingestion")
        if not getattr(settings, "langsmith_trace_content", False):
            os.environ["LANGCHAIN_HIDE_INPUTS"] = "true"
            os.environ["LANGCHAIN_HIDE_OUTPUTS"] = "true"
            os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
            os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


@contextmanager
def stage_span(name: str, metadata: Optional[dict] = None, enabled: bool = False) -> Iterator[None]:
    """Optional LangSmith run span; no-ops when tracing disabled or langsmith missing."""
    if not enabled:
        yield
        return
    try:
        from langsmith import trace
    except ImportError:
        yield
        return
    with trace(
        name,
        run_type="chain",
        inputs={},
        metadata=metadata or {},
    ):
        yield
