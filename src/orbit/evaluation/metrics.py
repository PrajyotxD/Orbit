"""Retrieval metrics.

Pure functions -- no engine, no database, no I/O, so each one is testable with
just a list of strings.

Chunks have two identities and metrics can be computed against either:
``chunk_id`` (only meaningful next to the chunking config that produced it) or
``(source_path, start_line, end_line)`` (survives a re-chunk). Reports carry both
because neither alone is honest.

Empty input returns a defined zero rather than raising.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "normalise_path",
    "location_key",
    "ranges_overlap",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "hit_rate_at_k",
    "citation_location_accuracy",
    "recall_at_k_relevant",
    "precision_at_k_relevant",
    "hit_rate_at_k_relevant",
    "mrr_relevant",
    "ndcg_at_k_relevant",
    "aggregate",
    "percentile",
    "latency_summary",
]

Number = float
Path_ = str


# ---
# location helpers
# ---
def normalise_path(path: str) -> str:
    """Normalise a path for comparison.

    Case-folded, backslashes turned into forward slashes, ``./`` stripped. This
    is what makes a label written by hand on Windows match the index on any
    platform.
    """
    if path is None:
        return ""
    p = str(path).replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p.lower()


def location_key(path: str, start_line: int, end_line: int) -> str:
    """Canonical string key for a source location, e.g. ``fusion.md:18-22``."""
    lo, hi = (start_line, end_line) if start_line <= end_line else (end_line, start_line)
    return f"{normalise_path(path)}:{lo}-{hi}"


def ranges_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> bool:
    """True when two inclusive 1-based line ranges share at least one line."""
    a_lo, a_hi = min(a_start, a_end), max(a_start, a_end)
    b_lo, b_hi = min(b_start, b_end), max(b_start, b_end)
    return a_lo <= b_hi and b_lo <= a_hi


def _clean(ids: Iterable[Any] | None) -> list[str]:
    """Deduplicate ids while preserving order, dropping empty values."""
    out: list[str] = []
    seen: set[str] = set()
    for i in ids or ():
        if i is None:
            continue
        s = i if isinstance(i, str) else str(i)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ---
# ranking metrics
# ---
def recall_at_k(
    retrieved_ids: Sequence[Any], relevant_ids: Sequence[Any], k: int
) -> float:
    """Fraction of relevant items that appear in the top ``k`` results.

    ``k <= 0`` or no relevant items yields ``0.0`` rather than an exception.
    Duplicates in ``retrieved_ids`` are collapsed, so a system cannot inflate
    recall by returning the same chunk twice.
    """
    rel = set(_clean(relevant_ids))
    if not rel or k is None or k <= 0:
        return 0.0
    top = set(_clean(retrieved_ids)[:k])
    return len(top & rel) / len(rel)


def precision_at_k(
    retrieved_ids: Sequence[Any], relevant_ids: Sequence[Any], k: int
) -> float:
    """Fraction of the top ``k`` results that are relevant.

    The denominator is ``k``, not ``min(k, len(retrieved))``: returning fewer
    results than asked for is not rewarded.
    """
    if k is None or k <= 0:
        return 0.0
    rel = set(_clean(relevant_ids))
    top = _clean(retrieved_ids)[:k]
    if not top:
        return 0.0
    return sum(1 for i in top if i in rel) / float(k)


def mrr(retrieved_ids: Sequence[Any], relevant_ids: Sequence[Any]) -> float:
    """Reciprocal rank of the first relevant hit, for ONE query.

    ``1.0`` if the relevant item is first, ``1/2`` if second, ``0.0`` if absent.
    Use :func:`mean_reciprocal_rank` to average this over many queries.
    """
    rel = set(_clean(relevant_ids))
    if not rel:
        return 0.0
    for position, item in enumerate(_clean(retrieved_ids), start=1):
        if item in rel:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(per_query_rr: Sequence[float]) -> float:
    """Mean of per-query :func:`mrr` values. Empty input gives ``0.0``."""
    values = [float(v) for v in per_query_rr if v is not None]
    return sum(values) / len(values) if values else 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    relevant_dict: Mapping[Any, Number] | None,
    k: int,
) -> float:
    """Normalised discounted cumulative gain at ``k``.

    ``relevant_dict`` maps an id to a *graded* relevance (any non-negative
    number; 0 means "not relevant" and may be omitted). The ideal ranking is
    computed by sorting the supplied gains in descending order, so a case with a
    single relevant item gives ``1/log2(rank+1)`` regardless of the gain value's
    scale.

    An empty ``relevant_dict`` or ``k <= 0`` gives ``0.0``.
    """
    grades: dict[str, float] = {}
    for key, value in (relevant_dict or {}).items():
        if key is None:
            continue
        try:
            g = float(value)
        except (TypeError, ValueError):
            continue
        if g <= 0:
            continue
        grades[key if isinstance(key, str) else str(key)] = g

    if not grades or k is None or k <= 0:
        return 0.0

    top = _clean(retrieved_ids)[:k]
    gains = [grades.get(item, 0.0) for item in top]
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return _dcg(gains) / idcg


def hit_rate_at_k(
    retrieved_ids: Sequence[Any], relevant_ids: Sequence[Any], k: int
) -> float:
    """``1.0`` if at least one relevant item appears in the top ``k``, else ``0.0``."""
    rel = set(_clean(relevant_ids))
    if not rel or k is None or k <= 0:
        return 0.0
    return 1.0 if set(_clean(retrieved_ids)[:k]) & rel else 0.0


# ---
# Orbit-specific: citation location accuracy
# ---
def _hit_location(hit: Any) -> tuple[str, int, int] | None:
    """Pull ``(path, start_line, end_line)`` out of a hit/evidence/dict."""
    ev = getattr(hit, "evidence", None)
    if ev is not None:
        hit = ev
    if isinstance(hit, Mapping):
        path = hit.get("source_path") or hit.get("path") or ""
        start = hit.get("start_line")
        end = hit.get("end_line", start)
    else:
        path = getattr(hit, "source_path", None) or getattr(hit, "path", "") or ""
        start = getattr(hit, "start_line", None)
        end = getattr(hit, "end_line", start)
    if path is None or start is None:
        return None
    try:
        return normalise_path(path), int(start), int(end)
    except (TypeError, ValueError):
        return None


def _gold_locations(gold: Any) -> list[tuple[str, int, int]]:
    """Normalise gold into ``[(path, start, end), ...]``.

    Accepts a sequence of tuples/lists, a sequence of dicts, a single tuple, a
    mapping ``{"path": ..., "start": ..., "end": ...}``, or any object exposing
    ``relevant_paths`` + ``relevant_line_ranges`` (i.e. a ``BenchmarkCase``).
    """
    if gold is None:
        return []
    paths = getattr(gold, "relevant_paths", None)
    ranges = getattr(gold, "relevant_line_ranges", None)
    if paths is not None and ranges is not None:
        out: list[tuple[str, int, int]] = []
        paths = list(paths)
        ranges = list(ranges)
        for i, rng in enumerate(ranges):
            p = paths[i] if len(paths) > 1 else paths[0]
            try:
                out.append((normalise_path(p), int(rng[0]), int(rng[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return out
    if isinstance(gold, Mapping):
        if "path" in gold and ("start" in gold or "start_line" in gold):
            gold = [gold]
        else:
            gold = [
                {"path": p, "start": r[0], "end": r[1]}
                for p, r in zip(gold.get("relevant_paths", []), gold.get("relevant_line_ranges", []))
            ]
    if isinstance(gold, (str, bytes)):
        return []
    out = []
    for item in gold or ():
        if isinstance(item, Mapping):
            path = item.get("source_path") or item.get("path") or ""
            start = item.get("start", item.get("start_line"))
            end = item.get("end", item.get("end_line", start))
        else:
            try:
                path, start, end = item[0], item[1], item[2]
            except (TypeError, IndexError):
                try:
                    path, start, end = item[0], item[1], item[1]
                except (TypeError, IndexError):
                    continue
        try:
            out.append((normalise_path(path), int(start), int(end)))
        except (TypeError, ValueError):
            continue
    return out


def citation_location_accuracy(
    results: Sequence[Any] | None, gold: Any
) -> dict[str, Any]:
    """Orbit's citation-location metric.

    Question answered: *does the retrieved passage point at the right place in
    the right file?* A hit counts only when its ``(path, start_line, end_line)``
    **overlaps** a gold location in the **same file**. Returning the right topic
    from the wrong file scores zero.

    ``results`` may contain :class:`~orbit.models.SearchHit`, :class:`~orbit.models.Evidence`
    or plain dicts; anything unparseable is skipped rather than guessed at.

    Returns a dict so the caller gets the detail, not just the headline:

    ``accuracy``
        fraction of gold locations covered by at least one retrieved hit.
        With no gold locations this is ``0.0`` and ``n_gold`` is ``0`` — an
        unlabelled question is never counted as a hit.
    ``file_accuracy``
        weaker: fraction of gold *files* that appear anywhere in the results.
    ``top1_location_hit`` / ``top1_file_hit``
        whether the first returned hit already satisfies location / file level.
    ``per_location``
        one entry per gold location with the covering hit's citation, for
        debugging a miss.
    """
    gold_locs = _gold_locations(gold)
    # Order is preserved: hit 0 is rank 1 and the top-1 checks depend on it.
    hits = [loc for loc in (_hit_location(h) for h in (results or ())) if loc]

    if not gold_locs:
        return {
            "accuracy": 0.0,
            "file_accuracy": 0.0,
            "n_gold": 0,
            "n_covered": 0,
            "n_hits": len(hits),
            "top1_location_hit": False,
            "top1_file_hit": False,
            "per_location": [],
        }

    per_location: list[dict[str, Any]] = []
    covered_files: set[str] = set()
    n_covered = 0
    for path, start, end in gold_locs:
        cover: tuple[str, int, int] | None = None
        for hp, hs, he in hits:
            if hp == path and ranges_overlap(hs, he, start, end):
                cover = (hp, hs, he)
                break
        if cover is not None:
            n_covered += 1
            covered_files.add(path)
        per_location.append(
            {
                "expected": location_key(path, start, end),
                "covered": cover is not None,
                "hit": location_key(*cover) if cover else None,
            }
        )

    gold_files = {p for p, _, _ in gold_locs}
    hit_files = {p for p, _, _ in hits}
    top = hits[0] if hits else None
    top_loc = top is not None and any(
        top[0] == p and ranges_overlap(top[1], top[2], s, e) for p, s, e in gold_locs
    )
    top_file = top is not None and top[0] in gold_files

    return {
        "accuracy": n_covered / float(len(gold_locs)),
        "file_accuracy": len(gold_files & hit_files) / float(len(gold_files)),
        "n_gold": len(gold_locs),
        "n_covered": n_covered,
        "n_hits": len(hits),
        "top1_location_hit": bool(top_loc),
        "top1_file_hit": bool(top_file),
        "per_location": per_location,
    }


# ---
# aggregation
# ---
def recall_at_k_relevant(relevant: Sequence[bool], total_gold: int, k: int) -> float:
    """Recall@k over a relevance vector.

    ``relevant[i]`` is True when retrieved item *i* satisfies at least one gold
    criterion. ``total_gold`` is how many distinct gold items exist (for the
    location level, how many gold line ranges there are).

    This exists because location-level relevance is an *overlap* property, not an
    identity one: a chunk covering lines 1-40 is relevant to a gold range of
    18-22 without being equal to it. Comparing ``path:start-end`` strings for
    equality yields a meaningless zero.
    """
    if total_gold <= 0 or k <= 0:
        return 0.0
    found = sum(1 for r in list(relevant)[:k] if r)
    # One retrieved chunk can cover several gold ranges, so cap at total_gold.
    return min(1.0, found / float(total_gold))


def precision_at_k_relevant(relevant: Sequence[bool], k: int) -> float:
    """Precision@k over a relevance vector."""
    if k <= 0:
        return 0.0
    window = list(relevant)[:k]
    if not window:
        return 0.0
    return sum(1 for r in window if r) / float(len(window))


def hit_rate_at_k_relevant(relevant: Sequence[bool], k: int) -> float:
    """1.0 when at least one of the first *k* items is relevant."""
    if k <= 0:
        return 0.0
    return 1.0 if any(relevant[:k]) else 0.0


def mrr_relevant(relevant: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant item."""
    for i, r in enumerate(relevant, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k_relevant(grades: Sequence[float], k: int) -> float:
    """nDCG@k over an explicit per-item relevance grade vector."""
    if k <= 0:
        return 0.0
    window = [float(g) for g in grades[:k]]
    if not window:
        return 0.0
    dcg = sum((2**g - 1) / _log2(i + 2) for i, g in enumerate(window))
    ideal = sorted((float(g) for g in grades), reverse=True)[:k]
    idcg = sum((2**g - 1) / _log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _log2(x: float) -> float:
    import math

    return math.log2(x)


def aggregate(per_query: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
    """Mean of every numeric field across per-query dicts.

    Non-numeric fields (``query``, ``mode``, ``citation``, ...) are skipped.
    ``None`` means "this number does not exist for this query" (for example a
    mode that was unavailable) and is excluded from the mean rather than
    counted as zero — averaging a missing measurement as zero is how benchmarks
    start lying.
    """
    if not per_query:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in per_query:
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            if value is None or isinstance(value, (str, bytes, Mapping, list, tuple, set)):
                continue
            if isinstance(value, bool):
                number = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                number = float(value)
            else:
                continue
            if isinstance(number, float) and (math.isnan(number) or math.isinf(number)):
                continue
            sums[key] = sums.get(key, 0.0) + number
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key]}


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. ``pct`` is a fraction in ``[0, 1]``."""
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    if pct <= 0:
        return data[0]
    if pct >= 1:
        return data[-1]
    rank = math.ceil(pct * len(data))
    return data[min(len(data) - 1, max(0, rank - 1))]


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """``{n, mean, p50, p95, min, max}`` in milliseconds. Empty input -> zeros."""
    data = [float(v) for v in samples_ms if v is not None]
    if not data:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "p50": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "min": min(data),
        "max": max(data),
    }