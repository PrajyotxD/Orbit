"""Evidence: validation, relationships, freshness and conflict candidates."""

from .freshness import (
    ConflictRule,
    detect_negation_conflicts,
    detect_numeric_conflicts,
    find_conflicts,
    freshness_summary,
    stale_evidence,
)
from .graph import add_edge, ensure_document_edges, neighbors, traverse
from .validation import chunk_matches_source, validate_citation, validate_evidence

__all__ = [
    "validate_evidence",
    "validate_citation",
    "chunk_matches_source",
    "add_edge",
    "ensure_document_edges",
    "neighbors",
    "traverse",
    "find_conflicts",
    "stale_evidence",
    "freshness_summary",
    "detect_numeric_conflicts",
    "detect_negation_conflicts",
    "ConflictRule",
]