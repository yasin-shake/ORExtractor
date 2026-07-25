"""Parser adapters and deterministic parser selection."""

from ingestion.parsers.base import DocumentParser
from ingestion.parsers.router import ParserRouter, get_parser_router

__all__ = ["DocumentParser", "ParserRouter", "get_parser_router"]
