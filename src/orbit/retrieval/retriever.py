"""Retrieval orchestration.

Runs the lexical and vector channels, fuses them, applies filters, and returns
hits that each carry a fully-populated :class:`~orbit.models.Evidence`.

A channel that fails to contribute is reported in ``degraded_reason`` and left
out of ``channels_used`` rather than quietly counted. A lexical-only result is a
normal result, not an error.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

from ..config import OrbitConfig
from ..embedding import EmbeddingProvider
from ..errors import RetrievalError
from ..models import (
    DocumentStatus,
    Evidence,
    RetrievalMethod,
    SearchHit,
    SearchResult,
    ValidationState,
)
from ..storage.db import OrbitDB
from . import fusion as fusion_mod
from . import lexical as lexical_mod
from . import vector as vector_mod

__all__ = ["Retriever"]


class Retriever:
    def __init__(self, db: OrbitDB, config: OrbitConfig, embedder: EmbeddingProvider | None = None):
        self.db = db
        self.config = config
        self._embedder = embedder
        self._embedder_unavailable_reason = ""

    # ------------------------------------------------------------------ helpers
    def _chunk_rows(self, chunk_ids: Sequence[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.db.query(
            "SELECT c.chunk_id, c.doc_id, c.text, c.content_hash, c.start_line, "
            "c.end_line, c.ordinal, d.path, d.language, d.status AS doc_status, "
            "d.revision, d.content_hash AS doc_hash, d.metadata AS doc_meta "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE c.chunk_id IN ({placeholders})",
            list(chunk_ids),
        )
        return {r["chunk_id"]: dict(r) for r in rows}

    def _apply_filters(
        self, chunk_ids: Sequence[str], *, language: str | None, doc_id: str | None,
        path_glob: str | None, tags: Sequence[str] | None,
    ) -> set[str]:
        allowed: set[str] | None = None
        if not any([language, doc_id, path_glob, tags]):
            return set(chunk_ids)
        rows = self.db.query(
            "SELECT c.chunk_id, d.language, d.doc_id, d.path, d.metadata "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id"
        )
        import fnmatch

        allowed = set()
        for r in rows:
            if r["chunk_id"] not in chunk_ids:
                continue
            if language and r["language"] != language:
                continue
            if doc_id and r["doc_id"] != doc_id:
                continue
            if path_glob and not fnmatch.fnmatch(r["path"], path_glob):
                continue
            if tags:
                import json

                try:
                    meta = json.loads(r["metadata"] or "{}")
                except json.JSONDecodeError:
                    meta = {}
                doc_tags = set(meta.get("tags", []))
                if not set(tags) & doc_tags:
                    continue
            allowed.add(r["chunk_id"])
        return allowed

    # -------------------------------------------------------------------- main
    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        language: str | None = None,
        doc_id: str | None = None,
        path_glob: str | None = None,
        tags: Sequence[str] | None = None,
        modes: Sequence[str] = ("lexical", "vector"),
        validate: bool = True,
    ) -> SearchResult:
        """Run a hybrid (or degraded) search and return ranked hits with evidence."""
        start = time.perf_counter()
        rcfg = self.config.retrieval
        limit = limit or rcfg.limit
        cand = limit * max(1, rcfg.candidate_multiplier)

        if query is None or not query.strip():
            return SearchResult(
                query=query or "",
                hits=(),
                retrieval_method=RetrievalMethod.LEXICAL,
                channels_used=(),
                degraded_reason="empty query",
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )

        modes = [m for m in modes if m in ("lexical", "vector")]
        use_lexical = "lexical" in modes and not rcfg.lexical_only
        use_vector = "vector" in modes and not rcfg.lexical_only

        lex_pairs: list[tuple[str, float]] = []
        vec_pairs: list[tuple[str, float]] = []
        channels_used: list[str] = []
        degraded = ""

        # Lexical channel
        if use_lexical:
            lex_hits = lexical_mod.search_lexical(
                self.db, query, limit=cand, mode="or", doc_id=doc_id
            )
            lex_pairs = [(h.chunk_id, h.score) for h in lex_hits]
            if lex_pairs:
                channels_used.append("lexical")

        # Vector channel (best-effort). A query with no searchable terms cannot
        # produce a meaningful vector match, so the channel is skipped entirely
        # rather than returning every chunk with zero similarity.
        if use_vector and _has_terms(query):
            try:
                qv = self._embed_query(query)
                vec_hits = vector_mod.search_vector(
                    self.db, qv, limit=cand, min_score=rcg_min_vector(rcfg)
                )
                vec_pairs = [(h.chunk_id, h.score) for h in vec_hits]
                # Only claim the channel ran if it actually contributed candidates.
                if vec_pairs:
                    channels_used.append("vector")
                else:
                    degraded = degraded or (
                        "vector channel produced no matches "
                        "(no embeddings stored for this index?)"
                    )
            except Exception as exc:
                degraded = f"vector channel unavailable: {exc}"[:200]
        elif use_vector:
            degraded = degraded or "query has no searchable terms; vector channel skipped"

        # If nothing produced candidates, try lexical even if only vector was requested.
        if not lex_pairs and not vec_pairs and "lexical" in modes:
            lex_hits = lexical_mod.search_lexical(self.db, query, limit=cand, mode="or")
            lex_pairs = [(h.chunk_id, h.score) for h in lex_hits]
            if lex_pairs:
                channels_used.append("lexical")
            else:
                degraded = degraded or "no channel produced results"

        # Fuse
        if lex_pairs and vec_pairs:
            method = rcfg.fusion if rcfg.fusion != "lexical_only" else "rrf"
            fused = fusion_mod.fuse(
                lex_pairs,
                vec_pairs,
                method=method,
                lexical_weight=rcfg.lexical_weight,
                vector_weight=rcfg.vector_weight,
                rrf_k=rcfg.rrf_k,
                min_normalized_score=rcfg.min_normalized_score,
            )
            retrieval_method = RetrievalMethod.HYBRID
        elif lex_pairs:
            fused = fusion_mod.fuse(lex_pairs, [], method="rrf", lexical_weight=1.0)
            retrieval_method = RetrievalMethod.LEXICAL
        elif vec_pairs:
            fused = fusion_mod.fuse([], vec_pairs, method="rrf", vector_weight=1.0)
            retrieval_method = RetrievalMethod.VECTOR
            # Only a degradation if the caller actually asked for both channels;
            # a vector-only request legitimately has no lexical results.
            if "lexical" in modes and not degraded:
                degraded = "lexical channel returned no matches"
        else:
            fused = []
            retrieval_method = RetrievalMethod.LEXICAL
            degraded = degraded or "no results"

        # Apply filters, then dedupe, then cap.
        fused_ids = [f.chunk_id for f in fused]
        filtered_ids = self._apply_filters(
            fused_ids, language=language, doc_id=doc_id, path_glob=path_glob, tags=tags
        )
        fused = [f for f in fused if f.chunk_id in filtered_ids]

        if rcfg.dedupe:
            fused = _dedupe(fused, self.db)

        fused = fused[:limit]
        rows = self._chunk_rows([f.chunk_id for f in fused])

        hits: list[SearchHit] = []
        for rank, f in enumerate(fused, start=1):
            row = rows.get(f.chunk_id)
            if row is None:
                continue
            if row["doc_status"] != DocumentStatus.CURRENT.value:
                continue
            ev = Evidence(
                chunk_id=f.chunk_id,
                doc_id=row["doc_id"],
                source_path=row["path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                content_hash=row["content_hash"],
                retrieval_method=retrieval_method,
                score=f.fused_score,
                rank=rank,
                lexical_score=f.lexical_score,
                vector_score=f.vector_score,
                normalized_score=f.normalized_score,
                revision=row["revision"],
                validation=ValidationState.NOT_VALIDATED,
            )
            if validate:
                from ..evidence import validate_evidence

                res = validate_evidence(self.db, ev.chunk_id)
                ev = replace(ev, validation=res.state, validation_detail=res.detail)
            hits.append(
                SearchHit(
                    evidence=ev,
                    text=row["text"],
                    snippet=_snippet(row["text"], rcfg.snippet_chars),
                    metadata={"language": row["language"], "ordinal": row["ordinal"]},
                )
            )

        return SearchResult(
            query=query,
            hits=tuple(hits),
            retrieval_method=retrieval_method,
            total_candidates=len(fused_ids),
            duration_ms=(time.perf_counter() - start) * 1000.0,
            channels_used=tuple(channels_used),
            degraded_reason=degraded,
        )

    # ------------------------------------------------------------------ internals
    def _embed_query(self, query: str) -> list[float]:
        if self._embedder is None:
            from ..embedding import build_embedder

            self._embedder = build_embedder(self.config.embedding)
        return self._embedder.embed_query(query)


def rcg_min_vector(rcfg) -> float:  # type: ignore[no-untyped-def]
    """Effective minimum vector score, as configured.

    See :attr:`RetrievalConfig.min_vector_score` for why the default is above
    zero. A negative value disables the floor (every candidate is kept).
    """
    return rcfg.min_vector_score


def _has_terms(query: str) -> bool:
    """True when the query contains at least one searchable term.

    Mirrors the token rule used by the lexical channel so both channels agree on
    what counts as a query.
    """
    from .lexical import build_match_query

    return bool(build_match_query(query or "", mode="or"))


def _dedupe(fused: list[fusion_mod.FusedHit], db) -> list[fusion_mod.FusedHit]:
    """Drop near-duplicate chunks.

    Duplicates arise when overlapping chunking produced identical text, or when
    the same file exists at two paths with identical content. We keep the best
    scoring instance of each distinct ``content_hash``.
    """
    seen: set[str] = set()
    out: list[fusion_mod.FusedHit] = []
    for f in fused:
        row = db.query_one(
            "SELECT content_hash FROM chunks WHERE chunk_id=?", (f.chunk_id,)
        )
        h = row["content_hash"] if row else f.chunk_id
        if h in seen:
            continue
        seen.add(h)
        out.append(f)
    return out


def _snippet(text: str, width: int) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= width:
        return text
    return text[:width].rsplit(" ", 1)[0] + " …"