"""Orbit exception hierarchy.

Every error carries a machine-readable ``code`` so CLI consumers, the MCP server
and the Python API can react without string matching on messages.
"""

from __future__ import annotations

__all__ = [
    "OrbitError",
    "ConfigurationError",
    "PathSafetyError",
    "SchemaError",
    "StorageError",
    "IndexCorruptError",
    "IngestionError",
    "ChunkingError",
    "EmbeddingError",
    "EmbeddingUnavailableError",
    "RetrievalError",
    "EvidenceError",
    "InvalidCitationError",
    "GenerationError",
    "IndexNotFoundError",
]


class OrbitError(Exception):
    """Base class for all Orbit errors."""

    code = "orbit_error"

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} ({self.detail})"
        return self.message

    def as_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message, "detail": self.detail}


class ConfigurationError(OrbitError):
    code = "configuration_error"


class PathSafetyError(OrbitError):
    code = "path_safety_error"


class SchemaError(OrbitError):
    code = "schema_error"


class StorageError(OrbitError):
    code = "storage_error"


class IndexCorruptError(StorageError):
    code = "index_corrupt"


class IngestionError(OrbitError):
    code = "ingestion_error"


class ChunkingError(IngestionError):
    code = "chunking_error"


class EmbeddingError(OrbitError):
    code = "embedding_error"


class EmbeddingUnavailableError(EmbeddingError):
    """Raised when no embedding provider can be constructed or reached.

    Retrieval intentionally degrades to lexical-only rather than failing.
    """

    code = "embedding_unavailable"


class RetrievalError(OrbitError):
    code = "retrieval_error"


class EvidenceError(OrbitError):
    code = "evidence_error"


class InvalidCitationError(EvidenceError):
    code = "invalid_citation"


class GenerationError(OrbitError):
    code = "generation_error"


class IndexNotFoundError(OrbitError):
    """Raised when an operation needs an index that has not been created."""

    code = "index_not_found"
