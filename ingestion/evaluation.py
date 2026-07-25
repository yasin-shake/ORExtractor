"""Offline parser evaluation records and optional LangSmith experiment logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion.models import ParserResult


def evaluation_record(
    result: ParserResult,
    *,
    expected_parser: str | None = None,
    expected_fallback: bool | None = None,
) -> dict[str, Any]:
    """Produce a stable JSON-serializable row for regression datasets."""
    return {
        "source_file": result.source_file,
        "selected_parser": result.parser,
        "parser_version": result.parser_version,
        "status": result.status,
        "page_count": result.page_count,
        "duration_ms": result.duration_ms,
        "element_count": len(result.elements),
        "quality": result.quality.model_dump(mode="json"),
        "fallback": result.fallback.model_dump(mode="json"),
        "expected_parser": expected_parser,
        "expected_fallback": expected_fallback,
        "parser_selection_correct": (
            None if expected_parser is None else result.parser == expected_parser
        ),
        "fallback_decision_correct": (
            None
            if expected_fallback is None
            else result.fallback.used == expected_fallback
        ),
    }


def append_evaluation_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one local benchmark row; no document content is included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_langsmith_evaluation(
    record: dict[str, Any],
    *,
    project_name: str = "orextractor-ingestion-evaluation",
) -> str | None:
    """Create a content-free LangSmith run only when explicitly invoked."""
    try:
        from langsmith import Client
    except ImportError:
        return None
    client = Client()
    run = client.create_run(
        name="parser-quality-evaluation",
        run_type="chain",
        inputs={},
        outputs={
            "selected_parser": record.get("selected_parser"),
            "quality": record.get("quality"),
            "fallback": record.get("fallback"),
        },
        project_name=project_name,
    )
    return str(getattr(run, "id", "") or "") or None
