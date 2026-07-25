"""LangChain integration points for parser output and benchmark experiments."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ingestion.chunking import elements_to_documents
from ingestion.models import ParserResult


def parser_result_to_documents(
    result: ParserResult,
    *,
    chunk_size: int = 1400,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Create the same LangChain document contract used by production Chroma writes."""
    return elements_to_documents(
        result.elements,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_with_langchain_docling(pdf_path: Path) -> list[Document]:
    """Official DoclingLoader baseline for parser/chunker benchmark comparisons."""
    try:
        from langchain_docling import DoclingLoader
    except ImportError as exc:
        raise RuntimeError(
            "langchain-docling is required for the DoclingLoader benchmark."
        ) from exc
    return DoclingLoader(file_path=str(pdf_path)).load()
