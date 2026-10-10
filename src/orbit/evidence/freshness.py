"""Knowledge freshness and heuristic conflict candidates.

Flags evidence whose source moved under it (delegating to
:mod:`orbit.evidence.validation`), and surfaces pairs of passages that *might*
disagree.

These are leads for a human to check, not findings. Similar text is not
contradiction. Two rules ship today: ``numeric_conflict`` (same attribute name,
materially different number, across two different files) and
``negation_conflict`` (near-duplicates where exactly one side negates). Both set
``kind=inferred`` and carry the rule in ``detail``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import ConflictCandidate, DocumentStatus, RelationKind

__all__ = [
    "ConflictRule",
    "find_conflicts",
    "detect_numeric_conflicts",
    "detect_negation_conflicts",
    "stale_evidence",
]

# Numeric assignments. Only the explicit `key = value` / `key: value` /
# `key DEFAULT value` shapes are recognised.
#
# Prose quantities ("a lifetime of 3600 seconds") are deliberately NOT used.
# There is no reliable way to know what a bare number in prose measures, and
# guessing produced false positives across unrelated sentences — that is worse
# than missing a real conflict, because a conflict detector nobody trusts is
# worse than no detector at all.
_NUM_ASSIGN = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_\-]{2,})\s*(?:=|:|\bis\b)\s*(?P<value>\d+(?:\.\d+)?)"
)
# SQL / config column defaults: `<column> <type...> DEFAULT <value>`. The key is
# the identifier immediately before the type keywords, which is why this needs its
# own pattern rather than a cleverer single regex.
_NUM_DEFAULT = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_\-]{2,})\s+"
    r"(?:INTEGER|INT|BIGINT|SMALLINT|NUMERIC|REAL|DOUBLE|VARCHAR|CHAR|TEXT|TIMESTAMP|BOOLEAN)?\s*"
    r"(?:NOT\s+NULL\s*)?DEFAULT\s+(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NEG = re.compile(r"\b(?:not|never|no|non|disabled|off|false|without)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")

# Words that look like an attribute name in `key = value` position but are not.
_ASSIGN_STOPWORDS = frozenset(
    {
        "the", "and", "for", "not", "with", "value", "default", "values",
        "int", "integer", "text", "varchar", "null", "primary", "foreign", "unique",
        "create", "table", "index", "insert", "select", "where", "order", "group",
    }
)


@dataclass
class ConflictRule:
    name: str
    description: str


RULES = (
    ConflictRule("numeric_conflict", "shared attribute name with materially different numeric values"),
    ConflictRule("negation_conflict", "near-duplicate text where one side negates a shared proposition"),
)


def _content_hash_for(text: str, start_line: int) -> str:
    from ..models import sha256_text

    return sha256_text(f"{start_line}:{text}")


def _load_chunks(db, *, limit: int, language: str | None = None) -> list[dict]:
    sql = (
        "SELECT c.chunk_id, c.text, c.start_line, c.end_line, c.content_hash, "
        "c.doc_id, d.path, d.language, d.status FROM chunks c "
        "JOIN documents d ON d.doc_id=c.doc_id WHERE d.status='current'"
    )
    params: list[object] = []
    if language:
        sql += " AND d.language=?"
        params.append(language)
    sql += " LIMIT ?"
    params.append(limit)
    return [dict(r) for r in db.query(sql, params)]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_numeric_conflicts(chunks: Sequence[dict], *, tol: float = 0.25) -> list[tuple[str, str, str, float, str]]:
    """Yield ``(a, b, rule, score, detail)`` for numeric-attribute disagreements.

    Only explicit assignments are compared, and only across *different* files: a
    document disagreeing with itself is a chunking artifact, not a conflict. The
    rule is intentionally conservative. It is a research lead, and a lead that
    cries wolf is worse than no lead.
    """
    attrs: dict[str, list[tuple[str, float, str, str, str]]] = {}

    for c in chunks:
        text = c["text"]
        lines = f"{c['start_line']}-{c['end_line']}"
        for m in _NUM_ASSIGN.finditer(text):
            key = m.group("key").lower()
            if key in _ASSIGN_STOPWORDS:
                continue
            try:
                val = float(m.group("value"))
            except ValueError:
                continue
            attrs.setdefault(key, []).append((c["chunk_id"], val, c["path"], lines, "unitless"))
        for m in _NUM_DEFAULT.finditer(text):
            key = m.group("key").lower()
            if key in _ASSIGN_STOPWORDS:
                continue
            try:
                val = float(m.group("value"))
            except ValueError:
                continue
            attrs.setdefault(key, []).append((c["chunk_id"], val, c["path"], lines, "default"))

    out: list[tuple[str, str, str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, entries in attrs.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a_id, a_val, a_path, a_lines, _au = entries[i]
                b_id, b_val, b_path, b_lines, _bu = entries[j]
                if a_id == b_id or a_path == b_path:
                    continue  # a document disagreeing with itself is a chunking artifact
                pair = tuple(sorted((a_id, b_id)))
                if pair in seen:
                    continue
                denom = max(abs(a_val), abs(b_val), 1e-9)
                rel = abs(a_val - b_val) / denom
                if rel > tol:
                    seen.add(pair)
                    score = min(1.0, rel)
                    detail = f"attribute '{key}' = {a_val:g} vs {b_val:g}"
                    out.append((a_id, b_id, "numeric_conflict", score, detail))
    return out



def detect_negation_conflicts(
    chunks: Sequence[dict], *, overlap_thresh: float = 0.55
) -> list[tuple[str, str, str, float, str]]:
    """Yield ``(a, b, rule, score, detail)`` for near-duplicate negated pairs."""
    prepared = []
    for c in chunks:
        words = set(_WORD.findall(c["text"].lower()))
        has_neg = bool(_NEG.search(c["text"]))
        prepared.append((c, words, has_neg))

    out: list[tuple[str, str, str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for i in range(len(prepared)):
        c_a, wa, na = prepared[i]
        for j in range(i + 1, len(prepared)):
            c_b, wb, nb = prepared[j]
            if na == nb:  # both or neither negated -> not a conflict candidate
                continue
            # Two overlapping chunks of the SAME document overlap because of
            # chunking, not because two sources disagree. Those are artifacts,
            # so a genuine conflict candidate must span two different files.
            if c_a["path"] == c_b["path"]:
                continue
            pair = tuple(sorted((c_a["chunk_id"], c_b["chunk_id"])))
            if pair in seen:
                continue
            overlap = _jaccard(wa, wb)
            if overlap >= overlap_thresh:
                seen.add(pair)
                neg_side = "negated" if na else "affirmative"
                detail = f"high overlap ({overlap:.2f}); one side {neg_side}"
                out.append(
                    (
                        c_a["chunk_id"],
                        c_b["chunk_id"],
                        "negation_conflict",
                        overlap,
                        detail,
                    )
                )
    return out


def find_conflicts(
    db, *, limit: int = 5000, language: str | None = None, persist: bool = True
) -> list[ConflictCandidate]:
    """Run all conflict rules over the current index and (optionally) persist them."""
    import time

    chunks = _load_chunks(db, limit=limit, language=language)
    raw: list[tuple[str, str, str, float, str]] = []
    raw.extend(detect_numeric_conflicts(chunks))
    raw.extend(detect_negation_conflicts(chunks))

    path_map: dict[str, tuple[str, str]] = {}
    for c in chunks:
        path_map[c["chunk_id"]] = (c["path"], f"{c['start_line']}-{c['end_line']}")

    candidates: list[ConflictCandidate] = []
    for a, b, rule, score, detail in raw:
        pa, la = path_map.get(a, ("?", "?"))
        pb, lb = path_map.get(b, ("?", "?"))
        candidates.append(
            ConflictCandidate(
                chunk_id_a=a,
                chunk_id_b=b,
                path_a=pa,
                path_b=pb,
                line_range_a=la,
                line_range_b=lb,
                rule=rule,
                score=score,
                detail=detail,
                kind=RelationKind.INFERRED,
            )
        )

    if persist:
        with db.transaction():
            db.execute("DELETE FROM conflicts")
            now = time.time()
            for c in candidates:
                cid = f"{c.chunk_id_a}|{c.chunk_id_b}|{c.rule}"
                db.execute(
                    "INSERT OR REPLACE INTO conflicts(conflict_id, chunk_id_a, chunk_id_b, "
                    "rule, score, kind, detail, detected_at) VALUES(?,?,?,?,?,?,?,?)",
                    (cid, c.chunk_id_a, c.chunk_id_b, c.rule, c.score, c.kind.value, c.detail, now),
                )

    candidates.sort(key=lambda c: -c.score)
    return candidates


def stale_evidence(db, *, limit: int = 500) -> list[dict]:
    """List evidence whose source no longer matches what was indexed.

    Every chunk is re-validated against the live file. An earlier version
    filtered on ``documents.status != 'current'``, which could never match:
    deleting a document purges its chunks, and editing one leaves the status as
    ``current``. Staleness is therefore a property of the *file on disk* versus
    the indexed content hash, not of the document row.
    """
    from .validation import validate_evidence

    rows = db.query("SELECT chunk_id FROM chunks ORDER BY chunk_id LIMIT ?", (limit,))
    out: list[dict] = []
    for r in rows:
        res = validate_evidence(db, r["chunk_id"])
        if not res.is_valid:
            out.append(
                {"chunk_id": r["chunk_id"], "state": res.state.value, "detail": res.detail}
            )
    return out


def freshness_summary(db) -> dict[str, int]:
    """Counts by freshness state across the current index."""
    from .validation import validate_evidence

    counts: dict[str, int] = {}
    rows = db.query(
        "SELECT c.chunk_id FROM chunks c JOIN documents d ON d.doc_id=c.doc_id"
    )
    for r in rows:
        state = validate_evidence(db, r["chunk_id"]).state.value
        counts[state] = counts.get(state, 0) + 1
    return counts