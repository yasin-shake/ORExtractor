"""Dependencies supplied by the application composition root."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class IngestionRuntime:
    """Small interface the ingestion module needs from its host application."""

    get_embedder: Callable[[Any], Any]
    get_vectorstore: Callable[[Any, Any], Any]

