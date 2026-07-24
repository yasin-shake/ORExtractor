"""Shared mutable API runtime state (avoids circular imports with routers)."""

from __future__ import annotations

from typing import Any, Optional

from rag_app import Settings

settings: Optional[Settings] = None
embedder: Any = None
vectorstore: Any = None
llm: Any = None
