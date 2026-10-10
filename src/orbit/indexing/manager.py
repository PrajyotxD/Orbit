"""Incremental index management.

Each document is written inside its own transaction, so an interrupted run never
leaves a half-written document behind. Embeddings are computed before the
transaction opens to keep slow providers off the write lock.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..chunking import chunk_document, estimate_tokens
from ..config import OrbitConfig
from ..embedding import EmbeddingProvider, build_embedder
from ..errors import EmbeddingUnavailableError, StorageError
from ..ingestion import iter_source_docs, parse_metadata
from ..models import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    IndexReport,
    IndexStats,
    LoadFailure,
    LoadFailureReason,
    SourceDoc,
    sha256_text,
)
from ..storage.db import OrbitDB

__all__ = ["IndexManager"]


@dataclass
class _FileState:
    path: str
    abs_path: str
    content_hash: str
    size_bytes: int
    mtime: float


class IndexManager:
    """Owns every write to the index."""

    def __init__(self, db: OrbitDB, config: OrbitConfig, embedder: EmbeddingProvider | None = None):
        self.db = db
        self.config = config
        self._embedder = embedder
        self._embedder_failed = False
        self._embedding_dim = 0
        self._embedding_provider = "none"

    # ---------------------------------------------------------------- embedder
    @property
    def embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = build_embedder(self.config.embedding)
            self._embedding_provider = getattr(self._embedder, "name", "unknown")
            self._embedding_dim = int(getattr(self._embedder, "dimension", 0) or 0)
        return self._embedder

    def _ensure_embedder(self) -> EmbeddingProvider | None:
        """Return a usable embedder, or ``None`` to degrade to lexical-only."""
        if self.config.embedding.provider == "none":
            return None
        if self._embedder_failed:
            return None
        try:
            emb = self.embedder
            if getattr(emb, "name", "") == "none":
                return None
            ok, _detail = emb.health()
            if not ok:
                raise EmbeddingUnavailableError("embedding provider health check failed")
            self._embedding_dim = int(getattr(emb, "dimension", 0) or self._embedding_dim)
            self._embedding_provider = getattr(emb, "name", "unknown")
            return emb
        except Exception:
            self._embedder_failed = True
            return None

    # ------------------------------------------------------------------ helpers
    def _existing_documents(self) -> dict[str, DocumentRecord]:
        rows = self.db.query(
            "SELECT doc_id, path, abs_path, content_hash, revision, size_bytes, mtime, language, "
            "status, indexed_at, metadata FROM documents"
        )
        import json

        out: dict[str, DocumentRecord] = {}
        for r in rows:
            try:
                meta = json.loads(r["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                meta = {}
            out[r["path"]] = DocumentRecord(
                doc_id=r["doc_id"],
                path=r["path"],
                content_hash=r["content_hash"],
                revision=r["revision"],
                size_bytes=r["size_bytes"],
                mtime=r["mtime"],
                language=r["language"],
                status=DocumentStatus(r["status"]),
                indexed_at=r["indexed_at"],
                abs_path=r["abs_path"] or "",
                metadata=meta,
            )
        return out

    @staticmethod
    def _doc_id(path: str) -> str:
        """Stable, human-debuggable id derived from the repo-relative path."""
        import hashlib

        return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]

    def _owning_root(self, rec: DocumentRecord) -> str | None:
        """The root a document was indexed under, recorded in its metadata."""
        return rec.metadata.get("root") if rec.metadata else None

    @staticmethod
    def _root_among(root_str: str, scannable_roots: Sequence[Path]) -> bool:
        try:
            target = Path(root_str).expanduser().resolve()
        except OSError:
            return False
        for r in scannable_roots:
            try:
                if r.resolve() == target:
                    return True
            except OSError:
                continue
        return False

    # -------------------------------------------------------------- main entry
    def sync(
        self,
        roots: Sequence[str | Path],
        *,
        prune: bool = True,
        exclude: Sequence[str] = (),
    ) -> IndexReport:
        """Reconcile ``roots`` into the index. Returns a summary of the sweep.

        ``exclude`` holds the operator's ``exclude_paths`` -- paths and globs that
        must never be indexed.

        Pruning is deliberately conservative: a document is only considered
        deleted if it lived under a root that was successfully scanned in this
        sweep. A root that failed to open (typo, unmounted volume, wrong cwd)
        must never be interpreted as "the whole corpus vanished".
        """
        start = time.perf_counter()
        self.db.migrate()

        existing = self._existing_documents()
        existing_by_path = existing
        hash_to_path = {d.content_hash: d.path for d in existing.values()}

        added = updated = unchanged = deleted = renamed = failed = 0
        skipped = 0
        chunks_written = embeddings_written = 0
        failures: list[tuple[str, str, str]] = []
        skips: list[tuple[str, str, str]] = []
        renamed_paths: set[str] = set()

        seen_paths: set[str] = set()
        renamed_sources: set[str] = set()

        # Determine which roots are actually openable before walking them. A root
        # that cannot be opened (typo, unmounted volume, wrong cwd) must not be
        # allowed to delete anything, so it is excluded from prune scope.
        scannable_roots: list[Path] = []
        for raw in roots:
            rp = Path(raw).expanduser()
            try:
                if rp.exists():
                    scannable_roots.append(rp)
            except OSError:
                continue

        items = iter_source_docs([str(r) for r in roots], self.config.ingestion, exclude)
        scanned = 0

        for item in items:
            if isinstance(item, LoadFailure):
                # Security exclusions and empty files are expected skips, not
                # errors. Only genuine load/parse problems count as failures.
                if item.reason in (
                    LoadFailureReason.SECRET_PATTERN,
                    LoadFailureReason.EMPTY,
                    LoadFailureReason.EXCLUDED,
                ):
                    skipped += 1
                    skips.append((item.path, item.reason.value, item.detail))
                else:
                    failed += 1
                    failures.append((item.path, item.reason.value, item.detail))
                continue
            doc: SourceDoc = item
            scanned += 1
            seen_paths.add(doc.path)
            owning_root = self._root_for_path(doc.abs_path, roots)

            prior = existing_by_path.get(doc.path)
            if (
                prior is not None
                and prior.content_hash == doc.content_hash
                and prior.metadata.get("chunk_signature") == self._chunk_signature()
            ):
                unchanged += 1
                continue

            # Detect a rename: same content hash at a different (previously
            # deleted) path. Only applies within this sweep's deletions, which
            # are applied after processing, so we compare against deleted docs.
            rename_from = None
            if prior is None and doc.content_hash in hash_to_path:
                candidate = hash_to_path[doc.content_hash]
                if candidate not in seen_paths:
                    rename_from = candidate

            try:
                res = self._index_document(doc, prior=prior, rename_from=rename_from, root=owning_root)
            except StorageError:
                failed += 1
                failures.append((doc.path, "storage_error", "transaction failed; document skipped"))
                continue
            except Exception as exc:  # noqa: BLE001 - index must not abort
                failed += 1
                failures.append((doc.path, "ingest_error", str(exc)[:160]))
                continue

            if rename_from:
                renamed += 1
                renamed_sources.add(rename_from)
            elif prior is None:
                added += 1
            else:
                updated += 1
            chunks_written += res.chunks
            embeddings_written += res.embeddings

        # Deletions: only prune documents that lived under a root we actually
        # opened and scanned. Documents whose path was never seen during this
        # sweep, but whose owning root failed to open, are left untouched.
        if prune and scannable_roots:
            prune_candidates: list[DocumentRecord] = []
            for path, rec in existing.items():
                if rec.status is not DocumentStatus.CURRENT:
                    continue
                if path in seen_paths:
                    continue
                owner = self._owning_root(rec)
                if owner and not self._root_among(owner, scannable_roots):
                    continue
                prune_candidates.append(rec)
            for rec in prune_candidates:
                self._mark_deleted(rec)
                deleted += 1

        self.db.set_meta("index_version", str(int(time.time())))
        self.db.set_meta("updated_at", str(time.time()))
        self.db.set_meta("last_sync", str(time.time()))

        duration_ms = (time.perf_counter() - start) * 1000.0
        return IndexReport(
            scanned=scanned,
            added=added,
            updated=updated,
            unchanged=unchanged,
            deleted=deleted,
            renamed=renamed,
            failed=failed,
            skipped=skipped,
            chunks_written=chunks_written,
            embeddings_written=embeddings_written,
            duration_ms=duration_ms,
            failures=tuple(failures),
            skips=tuple(skips),
        )

    # ------------------------------------------------------------- per-document
    @dataclass
    class _WriteResult:
        chunks: int
        embeddings: int

    def _chunk_signature(self) -> str:
        """Fingerprint of every setting that changes how a file is chunked.

        Stored per document so that changing the chunk size (or the embedder)
        invalidates existing chunks instead of leaving a stale index behind.
        Without this, re-indexing after a config change silently kept the old
        chunk boundaries and the report claimed everything was unchanged.
        """
        c = self.config.chunking
        e = self.config.embedding
        return sha256_text(
            "|".join(
                str(x)
                for x in (
                    c.max_tokens,
                    c.overlap_tokens,
                    c.max_chars,
                    c.tokens_per_char,
                    c.min_chunk_tokens,
                    c.respect_blank_lines,
                    c.respect_symbol_boundaries,
                    e.provider,
                    e.dimension,
                    e.model,
                    e.model_path,
                )
            )
        )[:16]

    def _index_document(
        self,
        doc: SourceDoc,
        *,
        prior: DocumentRecord | None,
        rename_from: str | None = None,
        root: str = "",
    ) -> _WriteResult:
        meta = parse_metadata(doc.text, doc.language)
        chunks = chunk_document(doc.text, self.config.chunking)
        doc_id = self._doc_id(doc.path)
        revision = (prior.revision + 1) if prior is not None else 1
        now = time.time()

        # Compute embeddings *before* opening the write transaction so a slow
        # or failing provider never holds a write lock.
        embedder = self._ensure_embedder()
        vectors: list[list[float]] | None = None
        if embedder is not None and chunks:
            try:
                vectors = embedder.embed([c.text for c in chunks])
            except Exception:
                # Embedding failure must not block lexical indexing.
                vectors = None
                self._embedder_failed = True

        with self.db.transaction():
            # If this is a rename, retire the old row (and its chunks) first.
            if rename_from and rename_from != doc.path:
                old = self._existing_documents().get(rename_from)
                if old is not None:
                    self._delete_document_rows(old.doc_id)

            # If updating, purge the previous chunks for this doc.
            if prior is not None:
                self._delete_document_rows(prior.doc_id)

            import json

            doc_meta = {
                "title": meta.title,
                "symbols": list(meta.symbols[:200]),
                "suffix": doc.metadata.get("suffix", ""),
                "root": root or self._root_for_doc(doc),
                "chunk_signature": self._chunk_signature(),
            }
            self.db.execute(
                "INSERT OR REPLACE INTO documents(doc_id, path, abs_path, content_hash, revision, "
                "size_bytes, mtime, language, status, indexed_at, metadata) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    doc.path,
                    doc.abs_path,
                    doc.content_hash,
                    revision,
                    doc.size_bytes,
                    doc.mtime,
                    doc.language,
                    DocumentStatus.CURRENT.value,
                    now,
                    json.dumps(doc_meta),
                ),
            )

            for ch in chunks:
                chunk_id = f"{doc_id}:{ch.ordinal}"
                chash = sha256_text(f"{ch.start_line}:{ch.text}")
                tok = estimate_tokens(ch.text, self.config.chunking.tokens_per_char)
                self.db.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, ordinal, text, content_hash, "
                    "start_line, end_line, start_offset, end_offset, token_estimate, metadata) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk_id,
                        doc_id,
                        ch.ordinal,
                        ch.text,
                        chash,
                        ch.start_line,
                        ch.end_line,
                        ch.start_offset,
                        ch.end_offset,
                        tok,
                        json.dumps({}),
                    ),
                )
                # Mirror into FTS (external content): insert the same rowid.
                rowid = self.db.scalar("SELECT id FROM chunks WHERE chunk_id=?", (chunk_id,))
                self.db.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)", (rowid, ch.text)
                )
                if vectors is not None:
                    vec = vectors[ch.ordinal]
                    self._write_embedding(chunk_id, vec)
                if meta.symbols:
                    pass  # symbols are document-level; stored below

            if meta.symbols:
                for ch in chunks:
                    chunk_id = f"{doc_id}:{ch.ordinal}"
                    for sym in meta.symbols:
                        self.db.execute(
                            "INSERT OR IGNORE INTO chunk_symbols(chunk_id, symbol, kind) "
                            "VALUES(?,?,?)",
                            (chunk_id, sym, doc.language),
                        )

        # Build evidence edges for this document's chunk graph (explicit
        # structural relations only: contains + adjacent + shared symbols).
        try:
            from ..evidence.graph import ensure_document_edges, add_shared_symbol_edges

            ensure_document_edges(self.db, doc_id)
            if meta.symbols:
                add_shared_symbol_edges(self.db, doc_id)
        except Exception:
            # Edge building is best-effort auxiliary structure; never block indexing.
            pass

        return self._WriteResult(chunks=len(chunks), embeddings=len(vectors or []))

    @staticmethod
    def _root_for_doc(doc: SourceDoc) -> str:
        """Deprecated shim retained for callers; the real root is passed in."""
        return str(Path(doc.abs_path).parent)

    @staticmethod
    def _root_for_path(abs_path: str, roots: Sequence[str | Path]) -> str:
        """Which of ``roots`` contains ``abs_path`` (longest match wins)."""
        try:
            target = Path(abs_path).resolve()
        except OSError:
            return str(Path(abs_path).parent)
        best = ""
        for raw in roots:
            rp = Path(raw).expanduser()
            try:
                resolved = rp.resolve()
            except OSError:
                continue
            try:
                target.relative_to(resolved)
            except ValueError:
                continue
            if len(str(resolved)) > len(best):
                best = str(resolved)
        return best or str(Path(abs_path).parent)

    def _write_embedding(self, chunk_id: str, vector: Sequence[float]) -> None:
        import array

        arr = array.array("f", [float(x) for x in vector])
        dim = len(arr)
        if self._embedding_dim and dim != self._embedding_dim:
            raise StorageError(
                f"embedding dimension mismatch: index expects {self._embedding_dim}, "
                f"provider returned {dim}; rebuild the index or change the provider"
            )
        self._embedding_dim = dim
        self.db.execute(
            "INSERT OR REPLACE INTO embeddings(chunk_id, dim, vector) VALUES(?,?,?)",
            (chunk_id, dim, arr.tobytes()),
        )

    def _delete_document_rows(self, doc_id: str) -> None:
        # Remove FTS rows first (they reference chunks by rowid), then chunks;
        # embeddings and chunk_symbols cascade via FK.
        rowids = self.db.query("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))
        chunk_ids = self.db.query("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))
        ids = [r["chunk_id"] for r in chunk_ids]
        for r in rowids:
            self.db.execute("DELETE FROM chunks_fts WHERE rowid=?", (r["id"],))
        # evidence_edges and conflicts have no foreign key to chunks, so orphaned
        # rows would otherwise accumulate forever and point at dead nodes.
        if ids:
            self._purge_graph_references(ids)
        self.db.execute("DELETE FROM evidence_edges WHERE source_id=? OR target_id=?",
                        (f"doc:{doc_id}", f"doc:{doc_id}"))
        self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    def _purge_graph_references(self, chunk_ids: Sequence[str]) -> None:
        """Delete evidence edges and conflict rows that reference ``chunk_ids``."""
        for cid in chunk_ids:
            self.db.execute(
                "DELETE FROM evidence_edges WHERE source_id=? OR target_id=?", (cid, cid)
            )
            self.db.execute("DELETE FROM conflicts WHERE chunk_id_a=? OR chunk_id_b=?", (cid, cid))
        # Document-level edges for this doc's chunks vanish with the chunks.

    def _mark_deleted(self, rec: DocumentRecord) -> None:
        """Soft-delete: purge chunks (so search won't return them) but retain the
        document row so a later rename can be recognised."""
        with self.db.transaction():
            self._delete_document_rows(rec.doc_id)
            # Re-insert a tombstone row so the hash remains discoverable.
            import json

            self.db.execute(
                "INSERT OR REPLACE INTO documents(doc_id, path, abs_path, content_hash, revision, "
                "size_bytes, mtime, language, status, indexed_at, metadata) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.doc_id,
                    rec.path,
                    rec.abs_path,
                    rec.content_hash,
                    rec.revision,
                    rec.size_bytes,
                    rec.mtime,
                    rec.language,
                    DocumentStatus.DELETED.value,
                    time.time(),
                    json.dumps({"tombstone": True}),
                ),
            )

    # ------------------------------------------------------------------- stats
    def stats(self) -> IndexStats:
        docs_current = int(self.db.scalar("SELECT count(*) FROM documents WHERE status='current'") or 0)
        docs_deleted = int(self.db.scalar("SELECT count(*) FROM documents WHERE status='deleted'") or 0)
        chunks = int(self.db.scalar("SELECT count(*) FROM chunks") or 0)
        emb = int(self.db.scalar("SELECT count(*) FROM embeddings") or 0)
        languages = {
            r["language"]: r["n"]
            for r in self.db.query(
                "SELECT language, count(*) AS n FROM documents WHERE status='current' GROUP BY language"
            )
        }
        provider = self.db.get_meta("embedding_provider") or self._embedding_provider
        # A fresh process has no in-memory embedder, so fall back to the value
        # recorded at index time and finally to the stored vectors' dimension.
        dim = self._embedding_dim
        if not dim:
            stored = self.db.get_meta("embedding_dim")
            if stored:
                dim = int(stored)
            else:
                dim = int(self.db.scalar("SELECT dim FROM embeddings LIMIT 1") or 0)
        return IndexStats(
            schema_version=self.db.schema_version(),
            documents_current=docs_current,
            documents_deleted=docs_deleted,
            chunks=chunks,
            embeddings=emb,
            embedding_provider=provider,
            embedding_dim=dim,
            index_version=self.db.get_meta("index_version") or "",
            created_at=float(self.db.get_meta("created_at") or 0.0),
            updated_at=float(self.db.get_meta("updated_at") or 0.0),
            db_path=str(self.db.path),
            languages=languages,
        )