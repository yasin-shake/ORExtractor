"""Common parser protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from ingestion.models import ParserResult


@runtime_checkable
class DocumentParser(Protocol):
    parser_name: str
    parser_version: str

    def parse(
        self,
        pdf_path: Path,
        *,
        source_file: Optional[str] = None,
        artifact_dir: Optional[Path] = None,
    ) -> ParserResult:
        ...
