"""Orbit engine — the public facade.

:class:`OrbitEngine` ties the storage, indexing and retrieval layers together and
is the stable entry point for the CLI, the MCP server and library users.

Typical use::

    from orbit import OrbitEngine, OrbitConfig

    engine = OrbitEngine(OrbitConfig(db_path="kb/orbit.db"))
    engine.index(["docs/"])                 # build/refresh the index
    result = engine.search("how does ranking work?", limit=5)
    for hit in result:
        print(hit.citation, hit.evidence.score)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .config import OrbitConfig
from .embedding import EmbeddingProvider
from .errors import IndexNotFoundError
from .indexing import IndexManager
from .models import (
    ConflictCandidate,
    ContextBlock,
    IndexReport,
    IndexStats,
    SearchHit,
    SearchResult,
    ValidationResult,
)
from .retrieval import Retriever
from .storage.db import OrbitDB

__all__ = ["OrbitEngine"]


class OrbitEngine:
    """High-level API over a single Orbit index."""

    def __init__(self, config: OrbitConfig | None = None, *, db: OrbitDB | None = None) -> None:
        self.config = config or OrbitConfig()
        self._owns_db = db is None
        self.db = db or OrbitDB(self.config.db_path)
        if self._owns_db:
            self.db.open()
        self.indexer = IndexManager(self.db, self.config)
        self.retriever = Retriever(self.db, self.config)

    # ------------------------------------------------------------------ context
    def __enter__(self) -> OrbitEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    # ----------------------------------------------------------------- indexing
    def init(self) -> IndexStats:
        """Create/migrate the index schema without indexing any files."""
        self.db.migrate()
        return self.indexer.stats()

    def index(
        self,
        paths: Sequence[str | Path] | None = None,
        *,
        prune: bool = True,
    ) -> IndexReport:
        """Reconcile ``paths`` (or the configured paths) into the index."""
        targets = [str(p) for p in (paths if paths is not None else self.config.paths)]
        if not targets:
            raise IndexNotFoundError(
                "no paths to index; pass paths explicitly or set config.paths"
            )
        report = self.indexer.sync(
            targets, prune=prune, exclude=self.config.exclude_paths
        )
        # Record the active embedding provider for status reporting. Only write
        # it when an embedder was actually constructed during this run; otherwise
        # (e.g. every document unchanged in a fresh process) we would overwrite a
        # correct persisted value with the "none" default.
        if self.indexer._embedding_provider != "none":
            self.db.set_meta("embedding_provider", self.indexer._embedding_provider)
        if self.indexer._embedding_dim:
            self.db.set_meta("embedding_dim", str(self.indexer._embedding_dim))
        return report

    def stats(self) -> IndexStats:
        return self.indexer.stats()

    def rebuild(self, paths: Sequence[str | Path] | None = None) -> IndexReport:
        """Drop all index state and re-index from scratch."""
        with self.db.transaction():
            for table in ("embeddings", "chunks_fts", "chunks", "chunk_symbols", "documents"):
                try:
                    if table == "chunks_fts":
                        self.db.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
                    else:
                        self.db.conn.execute(f"DELETE FROM {table}")
                except Exception:
                    pass
        return self.index(paths)

    # ---------------------------------------------------------------- retrieval
    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        language: str | None = None,
        path_glob: str | None = None,
        tags: Sequence[str] | None = None,
        modes: Sequence[str] = ("lexical", "vector"),
        validate: bool = True,
    ) -> SearchResult:
        return self.retriever.search(
            query,
            limit=limit,
            language=language,
            path_glob=path_glob,
            tags=tags,
            modes=modes,
            validate=validate,
        )

    def get_chunk(self, chunk_id: str) -> SearchHit | None:
        """Fetch a single chunk by id as a SearchHit (source-passage tool)."""
        row = self.db.query_one(
            "SELECT c.chunk_id, c.doc_id, c.text, c.content_hash, c.start_line, c.end_line, "
            "c.ordinal, d.path, d.language, d.revision FROM chunks c "
            "JOIN documents d ON d.doc_id=c.doc_id WHERE c.chunk_id=?",
            (chunk_id,),
        )
        if row is None:
            return None
        from .models import Evidence, RetrievalMethod, ValidationState

        ev = Evidence(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source_path=row["path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            content_hash=row["content_hash"],
            retrieval_method=RetrievalMethod.LEXICAL,
            score=0.0,
            rank=1,
            revision=row["revision"],
            validation=ValidationState.NOT_VALIDATED,
        )
        return SearchHit(
            evidence=ev,
            text=row["text"],
            metadata={"language": row["language"], "ordinal": row["ordinal"]},
        )

    def document(self, doc_id: str) -> dict | None:
        row = self.db.query_one(
            "SELECT doc_id, path, content_hash, revision, size_bytes, mtime, language, "
            "status, indexed_at, metadata FROM documents WHERE doc_id=?",
            (doc_id,),
        )
        return dict(row) if row else None

    def documents(self, *, status: str | None = "current") -> list[dict]:
        if status:
            rows = self.db.query(
                "SELECT doc_id, path, content_hash, revision, size_bytes, language, status "
                "FROM documents WHERE status=? ORDER BY path",
                (status,),
            )
        else:
            rows = self.db.query(
                "SELECT doc_id, path, content_hash, revision, size_bytes, language, status "
                "FROM documents ORDER BY path"
            )
        return [dict(r) for r in rows]

    def close_all(self) -> None:  # pragma: no cover - convenience alias
        self.close()

    # ------------------------------------------------------------------ evidence
    def validate_citation(self, chunk_id: str) -> ValidationResult:
        from .evidence import validate_evidence

        return validate_evidence(self.db, chunk_id)

    def conflicts(self, *, limit: int = 50) -> list[ConflictCandidate]:
        rows = self.db.query(
            "SELECT chunk_id_a, chunk_id_b, rule, score, kind, detail FROM conflicts "
            "ORDER BY score DESC LIMIT ?",
            (limit,),
        )
        path_map = self._chunk_paths([r["chunk_id_a"] for r in rows] + [r["chunk_id_b"] for r in rows])
        return [
            ConflictCandidate(
                chunk_id_a=r["chunk_id_a"],
                chunk_id_b=r["chunk_id_b"],
                path_a=path_map.get(r["chunk_id_a"], ("?", "?"))[0],
                path_b=path_map.get(r["chunk_id_b"], ("?", "?"))[0],
                line_range_a=path_map.get(r["chunk_id_a"], ("?", "?"))[1],
                line_range_b=path_map.get(r["chunk_id_b"], ("?", "?"))[1],
                rule=r["rule"],
                score=r["score"],
                detail=r["detail"],
            )
            for r in rows
        ]

    def _chunk_paths(self, chunk_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
        if not chunk_ids:
            return {}
        ph = ",".join("?" for _ in chunk_ids)
        rows = self.db.query(
            "SELECT c.chunk_id, d.path, c.start_line, c.end_line FROM chunks c "
            f"JOIN documents d ON d.doc_id=c.doc_id WHERE c.chunk_id IN ({ph})",
            list(chunk_ids),
        )
        return {
            r["chunk_id"]: (r["path"], f"{r['start_line']}-{r['end_line']}") for r in rows
        }

    # -------------------------------------------------------------------- context
    def assemble_context(self, query: str, *, limit: int = 10, **kwargs) -> object:
        from .context import assemble_context

        result = self.search(query, limit=limit)
        return assemble_context(result, self.config.context, **kwargs)