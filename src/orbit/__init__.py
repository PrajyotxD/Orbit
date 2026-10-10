"""Orbit: an evidence-driven, local-first retrieval engine.

Orbit indexes a corpus of documents, retrieves relevant passages with a hybrid
lexical+vector pipeline, and returns every result with a verifiable pointer back
to the exact source lines it came from. It is designed for developers,
knowledge-intensive applications and AI agents.

Quick start::

    from orbit import OrbitEngine, OrbitConfig

    engine = OrbitEngine(OrbitConfig(db_path="kb.db"))
    engine.index(["docs/"])
    result = engine.search("how is ranking fused?")
    for hit in result:
        print(hit.citation)

Orbit is useful with no LLM at all; answer generation is an optional layer.
"""

from .config import (
    ChunkingConfig,
    ContextBudgetConfig,
    EmbeddingConfig,
    GenerationConfig,
    IngestionConfig,
    OrbitConfig,
    RetrievalConfig,
    load_config,
)
from .engine import OrbitEngine
from .errors import (
    EmbeddingUnavailableError,
    OrbitError,
    RetrievalError,
)
from .models import (
    AssembledContext,
    ConflictCandidate,
    ContextBlock,
    DocumentRecord,
    DocumentStatus,
    Evidence,
    EvidenceEdge,
    IndexReport,
    IndexStats,
    RetrievalMethod,
    SearchHit,
    SearchResult,
    ValidationResult,
    ValidationState,
)

__version__ = "0.1.0"

__all__ = [
    "OrbitEngine",
    "OrbitConfig",
    "ChunkingConfig",
    "IngestionConfig",
    "RetrievalConfig",
    "EmbeddingConfig",
    "GenerationConfig",
    "ContextBudgetConfig",
    "load_config",
    "OrbitError",
    "RetrievalError",
    "EmbeddingUnavailableError",
    "DocumentRecord",
    "DocumentStatus",
    "ChunkRecord",
    "Evidence",
    "ValidationResult",
    "ValidationState",
    "SearchHit",
    "SearchResult",
    "IndexReport",
    "IndexStats",
    "ConflictCandidate",
    "EvidenceEdge",
    "AssembledContext",
    "ContextBlock",
    "RetrievalMethod",
    "__version__",
]

from .models import ChunkRecord  # noqa: E402  (re-export for convenience)