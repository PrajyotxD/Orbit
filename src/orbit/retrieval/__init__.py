"""Retrieval: lexical, vector, fusion and orchestration."""

from .fusion import FusedHit, fuse, min_max_normalize
from .lexical import LexicalHit, build_match_query, search_lexical
from .retriever import Retriever
from .vector import VectorHit, cosine_scores, search_vector

__all__ = [
    "Retriever",
    "FusedHit",
    "fuse",
    "min_max_normalize",
    "LexicalHit",
    "build_match_query",
    "search_lexical",
    "VectorHit",
    "cosine_scores",
    "search_vector",
]