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
            # Prefer hiding inputs/outputs when supported
            os.environ.setdefault("LANGCHAIN_HIDE_INPUTS", "true")
            os.environ.setdefault("LANGCHAIN_HIDE_OUTPUTS", "true")
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
        from langsmith import traceable

        @traceable(name=name, metadata=metadata or {})
        def _run():
            return None

        # Use RunTree if available for proper context manager semantics
        try:
            from langsmith.run_helpers import tracing_context
            from langsmith import RunTree

            rt = RunTree(name=name, run_type="chain", extra={"metadata": metadata or {}})
            rt.post()
            try:
                yield
                rt.end()
                rt.patch()
            except Exception as exc:
                rt.end(error=str(exc))
                rt.patch()
                raise
            return
        except Exception:
            _run()
            yield
            return
    except Exception:
        yield
