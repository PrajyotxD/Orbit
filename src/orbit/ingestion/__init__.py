"""Ingestion: discovery, safety filters, decoding and metadata extraction."""

from .loader import discover_files, is_secret_filename, iter_source_docs, load_file
from .parsers import ParsedMeta, detect_language, enrich_text, parse_metadata

__all__ = [
    "discover_files",
    "load_file",
    "iter_source_docs",
    "is_secret_filename",
    "ParsedMeta",
    "parse_metadata",
    "detect_language",
    "enrich_text",
]