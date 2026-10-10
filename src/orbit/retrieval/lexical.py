"""Lexical retrieval over SQLite FTS5.

BM25 weighting comes from SQLite's built-in ``bm25()`` rather than a local
reimplementation. FTS5 returns "more negative is better", so scores are negated
on the way out to match the direction cosine uses.

Raw user text is never handed to MATCH -- it has its own syntax (quotes, NEAR,
``*``, AND/OR/NOT) and will either throw or silently match nothing.
:func:`build_match_query` re-tokenises input into quoted terms first.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import RetrievalError

__all__ = ["LexicalHit", "build_match_query", "search_lexical", "search_lexical_raw"]

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    score: float  # higher is better (negated bm25)
    bm25_raw: float
    rank: int


def build_match_query(query: str, *, mode: str = "and") -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    ``mode='and'`` requires all terms (precision); ``mode='or'`` is more
    forgiving (recall). Terms containing only punctuation are dropped. The result
    is a sequence of quoted terms joined by the operator, which is always valid
    FTS5 syntax.
    """
    mode = mode.lower()
    if mode not in ("and", "or"):
        raise RetrievalError("lexical match mode must be 'and' or 'or'")
    terms = _TOKEN.findall(query)
    # ``str.isalnum`` is False for identifiers containing underscores, which are
    # the single most important query terms when searching code. Accept any token
    # that contains at least one word character.
    cleaned = [t for t in terms if t and any(ch.isalnum() or ch == "_" for ch in t)]
    if not cleaned:
        return ""
    # Quote each term, escaping embedded double quotes by doubling them.
    quoted = ['"' + t.replace('"', '""') + '"' for t in cleaned]
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(quoted)


def search_lexical_raw(db, match_expr: str, limit: int) -> list[tuple[str, float, float]]:
    """Run a raw MATCH expression; returns ``(chunk_id, negated_bm25, bm25_raw)``."""
    if not match_expr:
        return []
    try:
        rows = db.query(
            "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY rank ASC LIMIT ?",
            (match_expr, limit),
        )
    except Exception as exc:  # sqlite3.OperationalError and friends
        raise RetrievalError(f"lexical search failed: {exc}") from exc
    return [(r["chunk_id"], -float(r["rank"]), float(r["rank"])) for r in rows]


def search_lexical(
    db,
    query: str,
    *,
    limit: int = 50,
    mode: str = "and",
    doc_id: str | None = None,
) -> list[LexicalHit]:
    """Lexical search returning ranked hits. Higher score = better.

    ``doc_id`` optionally restricts results to a single document (used by the
    "search within a file" flow and metadata filtering).
    """
    expr = build_match_query(query, mode=mode)
    if not expr:
        return []
    hits = search_lexical_raw(db, expr, limit * 4 if doc_id is None else limit)
    out: list[LexicalHit] = []
    for chunk_id, score, raw in hits:
        out.append(LexicalHit(chunk_id=chunk_id, score=score, bm25_raw=raw, rank=0))
        if len(out) >= limit:
            break
    for i, h in enumerate(out):
        object.__setattr__(h, "rank", i + 1)
    return out


def count_lexical(db, query: str, *, mode: str = "and") -> int:
    """Number of lexical matches (used for diagnostics)."""
    expr = build_match_query(query, mode=mode)
    if not expr:
        return 0
    try:
        return int(
            db.scalar(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (expr,)
            )
            or 0
        )
    except Exception as exc:
        raise RetrievalError(f"lexical count failed: {exc}") from exc


def search_lexical_terms(db, terms: Sequence[str], limit: int = 50) -> list[LexicalHit]:
    """Search for an explicit list of terms (OR-joined)."""
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms if t]
    if not quoted:
        return []
    raw = search_lexical_raw(db, " OR ".join(quoted), limit)
    return [
        LexicalHit(chunk_id=c, score=s, bm25_raw=r, rank=i + 1)
        for i, (c, s, r) in enumerate(raw)
    ]