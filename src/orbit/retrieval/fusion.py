"""Rank fusion: combine lexical and vector result lists into one ranking.

Two established strategies are implemented, and the choice is explicit in config.

Reciprocal Rank Fusion (RRF)
    score(d) = Σ_channels  weight_c / (k + rank_c(d))

    Introduced by Cormack, Clarke and Büttcher (SIGIR 2009) and used by several
    production search systems. Its advantage here is that it is *rank-based*: it
    needs no cross-channel score calibration, which matters because BM25 scores
    and cosine similarities live on completely different scales. The constant
    ``k`` (default 60, following the paper) dampens the influence of the very top
    ranks so a single channel's #1 cannot dominate.

Min-max weighted fusion
    Each channel's scores are min-max normalised to [0, 1] across that channel's
    own result list, then combined as a weighted sum. More sensitive to outliers
    than RRF but lets an operator express a real preference for one channel.

Both are *our own orchestration* of established ideas; neither invents a new
ranking function, and Orbit does not claim novelty here. The original work is
in deciding when each channel is trusted, how degraded modes are signalled, and
how the fused ranking is audited (every hit keeps both raw scores).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["FusedHit", "fuse", "min_max_normalize"]


@dataclass
class FusedHit:
    chunk_id: str
    fused_score: float
    lexical_score: float | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    vector_rank: int | None = None
    normalized_score: float = 0.0
    contributions: dict[str, float] = field(default_factory=dict)


def min_max_normalize(scores: Sequence[float]) -> list[float]:
    """Scale ``scores`` into [0, 1]. A single distinct value maps to 1.0."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


def fuse(
    lexical: Sequence[tuple[str, float]],
    vector: Sequence[tuple[str, float]],
    *,
    method: str = "rrf",
    lexical_weight: float = 1.0,
    vector_weight: float = 1.0,
    rrf_k: int = 60,
    min_normalized_score: float = 0.0,
) -> list[FusedHit]:
    """Fuse two ranked ``(chunk_id, score)`` lists into a single ranking.

    ``lexical``/``vector`` are expected in descending score order. Ties are broken
    by first-seen order in the input lists, then by channel priority (lexical
    first) for full determinism.
    """
    if method not in ("rrf", "weighted"):
        raise ValueError(f"unknown fusion method {method!r}; use 'rrf' or 'weighted'")

    lex_order = {cid: i for i, (cid, _) in enumerate(lexical, start=1)}
    vec_order = {cid: i for i, (cid, _) in enumerate(vector, start=1)}
    lex_score = dict(lexical)
    vec_score = dict(vector)

    all_ids = set(lex_order) | set(vec_order)

    if method == "rrf":
        results = []
        for cid in all_ids:
            contrib: dict[str, float] = {}
            total = 0.0
            if cid in lex_order and lexical_weight > 0:
                c = lexical_weight / (rrf_k + lex_order[cid])
                total += c
                contrib["lexical"] = c
            if cid in vec_order and vector_weight > 0:
                c = vector_weight / (rrf_k + vec_order[cid])
                total += c
                contrib["vector"] = c
            results.append(
                FusedHit(
                    chunk_id=cid,
                    fused_score=total,
                    lexical_score=lex_score.get(cid),
                    vector_score=vec_score.get(cid),
                    lexical_rank=lex_order.get(cid),
                    vector_rank=vec_order.get(cid),
                    contributions=contrib,
                )
            )
    else:  # weighted
        lex_norm = min_max_normalize([s for _, s in lexical])
        vec_norm = min_max_normalize([s for _, s in vector])
        lex_n = dict(zip([c for c, _ in lexical], lex_norm))
        vec_n = dict(zip([c for c, _ in vector], vec_norm))
        results = []
        for cid in all_ids:
            lw = lexical_weight * lex_n.get(cid, 0.0)
            vw = vector_weight * vec_n.get(cid, 0.0)
            total = lw + vw
            results.append(
                FusedHit(
                    chunk_id=cid,
                    fused_score=total,
                    lexical_score=lex_score.get(cid),
                    vector_score=vec_score.get(cid),
                    lexical_rank=lex_order.get(cid),
                    vector_rank=vec_order.get(cid),
                    contributions={"lexical": lw, "vector": vw},
                )
            )

    results.sort(key=lambda h: (-h.fused_score, _tiebreak_key(h, lex_order, vec_order)))

    if min_normalized_score > 0:
        # Compute a normalised 0..1 view of the fused scores for filtering.
        norm = min_max_normalize([h.fused_score for h in results])
        for h, n in zip(results, norm):
            h.normalized_score = n
        results = [h for h in results if h.normalized_score >= min_normalized_score]

    return results


def _tiebreak_key(h: FusedHit, lex_order: Mapping[str, int], vec_order: Mapping[str, int]):
    """Deterministic tie-break: more channels first, then lexical, then id."""
    channels = (1 if h.lexical_rank else 0) + (1 if h.vector_rank else 0)
    return (-channels, h.lexical_rank or 10**9, h.vector_rank or 10**9, h.chunk_id)