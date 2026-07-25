"""Normalize parser-specific output into canonical element records."""

from ingestion.normalizers.docling import normalize_docling_document
from ingestion.normalizers.mineru import normalize_mineru_content

__all__ = ["normalize_docling_document", "normalize_mineru_content"]
