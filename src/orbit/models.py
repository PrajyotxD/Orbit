"""Core data model.

Plain frozen dataclasses with no behaviour, so every subsystem can build and
consume them without importing each other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "DocumentStatus",
    "RetrievalMethod",
    "ValidationState",
    "RelationKind",
    "LoadFailureReason",
    "DocumentRecord",
    "ChunkRecord",
    "SourceDoc",
    "SourceChunk",
    "Evidence",
    "ValidationResult",
    "SearchHit",
    "SearchResult",
    "IndexReport",
    "IndexStats",
    "ConflictCandidate",
    "EvidenceEdge",
    "TraversalStep",
    "sha256_text",
    "sha256_bytes",
]

SCHEMA_VERSION = 1


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 encoded string, hex encoded."""
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DocumentStatus(str, Enum):
    """Lifecycle state of a document record in the index."""

    CURRENT = "current"
    DELETED = "deleted"


class RetrievalMethod(str, Enum):
    """Which retrieval channels produced a hit."""

    LEXICAL = "lexical"
    VECTOR = "vector"
    HYBRID = "hybrid"


class ValidationState(str, Enum):
    """Outcome of re-validating an evidence reference against the live source."""

    VALID = "valid"
    PATH_MISSING = "path_missing"
    CONTENT_CHANGED = "content_changed"
    CHUNK_OUT_OF_RANGE = "chunk_out_of_range"
    DOCUMENT_DELETED = "document_deleted"
    NOT_VALIDATED = "not_validated"


class RelationKind(str, Enum):
    """Class of an evidence relationship.

    ``EXPLICIT`` relationships are derived from structural facts that Orbit can
    prove (same repository, a real symbol reference, a direct hyperlink).
    ``INFERRED`` relationships are produced by a heuristic and must always be
    reported as such; they are never presented as fact.
    """

    EXPLICIT = "explicit"
    INFERRED = "inferred"


class LoadFailureReason(str, Enum):
    """Why a candidate file could not be indexed."""

    UNREADABLE = "unreadable"
    DECODE_ERROR = "decode_error"
    PARSE_ERROR = "parse_error"
    TOO_LARGE = "too_large"
    UNSUPPORTED = "unsupported"
    EXCLUDED = "excluded"
    SECRET_PATTERN = "secret_pattern"
    EMPTY = "empty"


@dataclass(frozen=True)
class DocumentRecord:
    """A persisted document (one source file), content-addressed."""

    doc_id: str
    path: str
    content_hash: str
    revision: int
    size_bytes: int
    mtime: float
    language: str
    status: DocumentStatus = DocumentStatus.CURRENT
    indexed_at: float = 0.0
    abs_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        return self.status is DocumentStatus.CURRENT


@dataclass(frozen=True)
class ChunkRecord:
    """A deterministic slice of a document with full source provenance."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    content_hash: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    token_estimate: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def line_range(self) -> str:
        if self.start_line == self.end_line:
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"


@dataclass
class SourceDoc:
    """A document as loaded from disk, before chunking."""

    path: str
    abs_path: str
    text: str
    language: str
    content_hash: str
    size_bytes: int
    mtime: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceChunk:
    """A chunk produced by the chunker, prior to persistence."""

    ordinal: int
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    token_estimate: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """Everything needed to re-verify a retrieved passage against its source.

    An :class:`Evidence` record is *not* a claim that the passage is true. It is
    a verifiable pointer back to the indexed bytes it came from.
    """

    chunk_id: str
    doc_id: str
    source_path: str
    start_line: int
    end_line: int
    content_hash: str
    retrieval_method: RetrievalMethod
    score: float
    rank: int
    lexical_score: float | None = None
    vector_score: float | None = None
    normalized_score: float = 0.0
    revision: int | None = None
    validation: ValidationState = ValidationState.NOT_VALIDATED
    validation_detail: str = ""

    @property
    def line_range(self) -> str:
        if self.start_line == self.end_line:
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"

    @property
    def is_valid(self) -> bool:
        return self.validation is ValidationState.VALID


@dataclass(frozen=True)
class ValidationResult:
    """Result of checking an evidence reference against the current index state."""

    state: ValidationState
    detail: str = ""
    current_hash: str | None = None
    current_revision: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.state is ValidationState.VALID


@dataclass(frozen=True)
class SearchHit:
    """A single ranked retrieval result."""

    evidence: Evidence
    text: str
    snippet: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """A compact human-readable citation string."""
        return f"{self.evidence.source_path}:{self.evidence.line_range}"


@dataclass(frozen=True)
class SearchResult:
    """The result set of a single query, with provenance for every hit."""

    query: str
    hits: tuple[SearchHit, ...]
    retrieval_method: RetrievalMethod
    total_candidates: int = 0
    duration_ms: float = 0.0
    channels_used: tuple[str, ...] = ()
    degraded_reason: str = ""

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)


@dataclass(frozen=True)
class IndexReport:
    """Summary of one index synchronisation pass."""

    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    renamed: int = 0
    failed: int = 0
    skipped: int = 0
    chunks_written: int = 0
    embeddings_written: int = 0
    duration_ms: float = 0.0
    failures: tuple[tuple[str, str, str], ...] = ()
    skips: tuple[tuple[str, str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "added": self.added,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "deleted": self.deleted,
            "renamed": self.renamed,
            "failed": self.failed,
            "skipped": self.skipped,
            "chunks_written": self.chunks_written,
            "embeddings_written": self.embeddings_written,
            "duration_ms": round(self.duration_ms, 3),
            "failures": [
                {"path": p, "reason": r, "detail": d} for p, r, d in self.failures
            ],
            "skips": [{"path": p, "reason": r, "detail": d} for p, r, d in self.skips],
        }


@dataclass(frozen=True)
class IndexStats:
    """Read-only snapshot of index state."""

    schema_version: int
    documents_current: int
    documents_deleted: int
    chunks: int
    embeddings: int
    embedding_provider: str
    embedding_dim: int
    index_version: str
    created_at: float
    updated_at: float
    db_path: str
    languages: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "documents_current": self.documents_current,
            "documents_deleted": self.documents_deleted,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "embedding_provider": self.embedding_provider,
            "embedding_dim": self.embedding_dim,
            "index_version": self.index_version,
            "languages": dict(self.languages),
            "db_path": self.db_path,
        }


@dataclass(frozen=True)
class ConflictCandidate:
    """Two passages that a heuristic flagged as *possibly* conflicting.

    ``ConflictCandidate`` is a research lead, not a finding. The rule that raised
    it is named in ``rule`` so a human or an agent can judge it.
    """

    chunk_id_a: str
    chunk_id_b: str
    path_a: str
    path_b: str
    line_range_a: str
    line_range_b: str
    rule: str
    score: float
    detail: str = ""
    kind: RelationKind = RelationKind.INFERRED


@dataclass(frozen=True)
class EvidenceEdge:
    """A relationship between two indexed chunks (or documents)."""

    source_id: str
    target_id: str
    kind: RelationKind
    relation: str
    weight: float = 1.0
    detail: str = ""


@dataclass(frozen=True)
class TraversalStep:
    """One hop of a bounded evidence traversal."""

    from_id: str
    to_id: str
    relation: str
    kind: RelationKind
    depth: int
    detail: str = ""


@dataclass(frozen=True)
class ContextBlock:
    """One passage selected for context assembly."""

    text: str
    source_path: str
    start_line: int
    end_line: int
    content_hash: str
    chunk_id: str
    token_estimate: int
    score: float
    rank: int


@dataclass(frozen=True)
class AssembledContext:
    """Output of the context assembler."""

    blocks: tuple[ContextBlock, ...]
    total_token_estimate: int
    token_budget: int
    truncated: bool
    dropped_duplicates: int = 0
    dropped_irrelevant: int = 0

    def render(self) -> str:
        """Render the assembled context as text with inline source markers."""
        parts: list[str] = []
        for i, block in enumerate(self.blocks, start=1):
            parts.append(
                f"[{i}] Source: {block.source_path}:{block.start_line}-{block.end_line}\n"
                f"    Hash: {block.content_hash[:12]}\n"
                f"{block.text}"
            )
        return "\n\n".join(parts)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Answer plus the evidence it was given.

    The presence of citations does not imply the answer is correct. It records
    which retrieved records were supplied to the generator.
    """

    text: str
    citations: tuple[Evidence, ...]
    model: str
    insufficient_evidence: bool = False
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass(frozen=True)
class LoadFailure:
    """A file that could not be indexed."""

    path: str
    reason: LoadFailureReason
    detail: str = ""

