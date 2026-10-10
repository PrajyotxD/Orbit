"""Context assembly.

Turns a ranked result into a bounded set of blocks for a generator (or a human).

No padding -- blocks below ``min_score_ratio`` of the top score are dropped
rather than filling the budget. No duplicates -- by content hash, then by Jaccard
against ``similarity_threshold``. Ordering is deterministic so dict iteration
order never leaks into output.

Token counts are ``ceil(chars / tokens_per_char)`` unless a real tokenizer is
passed as ``token_counter``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .chunking import estimate_tokens
from .config import ContextBudgetConfig
from .models import AssembledContext, ContextBlock, SearchResult

__all__ = ["assemble_context"]

_WORD = __import__("re").compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assemble_context(
    result: SearchResult,
    budget: ContextBudgetConfig | None = None,
    *,
    token_counter: Callable[[str], int] | None = None,
    include_validation: bool = False,
) -> AssembledContext:
    """Build a bounded, de-duplicated context from a search result."""
    cfg = budget or ContextBudgetConfig()
    counter = token_counter or (lambda t: estimate_tokens(t, cfg_tokens_per_char(cfg)))

    hits = list(result.hits)
    if not hits:
        return AssembledContext(blocks=(), total_token_estimate=0, token_budget=cfg.token_budget, truncated=False)

    top_score = max((h.evidence.score for h in hits), default=0.0)
    floor = top_score * cfg.min_score_ratio if top_score > 0 else 0.0

    dropped_irrelevant = 0
    candidates = [h for h in hits if h.evidence.score >= floor]
    dropped_irrelevant += len(hits) - len(candidates)

    if cfg.ordering == "score":
        candidates.sort(key=lambda h: (-h.evidence.score, h.evidence.chunk_id))
    else:
        candidates.sort(key=lambda h: (h.evidence.source_path, h.evidence.start_line, h.evidence.chunk_id))

    # Dedupe.
    kept: list[ContextBlock] = []
    seen_hash: set[str] = set()
    kept_tokens: list[set[str]] = []
    dropped_duplicates = 0
    total_tokens = 0
    truncated = False

    for hit in candidates:
        ev = hit.evidence
        text = hit.text
        if cfg.dedupe_hash and ev.content_hash in seen_hash:
            dropped_duplicates += 1
            continue
        toks = counter(text)
        if total_tokens + toks > cfg.token_budget:
            truncated = True
            continue
        if len(kept) >= cfg.max_blocks:
            truncated = True
            continue

        word_set = _tokens(text)
        if cfg.dedupe_similarity:
            if any(_jaccard(word_set, prev) >= cfg.similarity_threshold for prev in kept_tokens):
                dropped_duplicates += 1
                continue

        block = ContextBlock(
            text=text,
            source_path=ev.source_path,
            start_line=ev.start_line,
            end_line=ev.end_line,
            content_hash=ev.content_hash,
            chunk_id=ev.chunk_id,
            token_estimate=toks,
            score=ev.score,
            rank=ev.rank,
        )
        kept.append(block)
        kept_tokens.append(word_set)
        seen_hash.add(ev.content_hash)
        total_tokens += toks

    return AssembledContext(
        blocks=tuple(kept),
        total_token_estimate=total_tokens,
        token_budget=cfg.token_budget,
        truncated=truncated,
        dropped_duplicates=dropped_duplicates,
        dropped_irrelevant=dropped_irrelevant,
    )


def cfg_tokens_per_char(cfg: ContextBudgetConfig) -> float:
    return getattr(cfg, "tokens_per_char", 4.0)


def render_context(ctx: AssembledContext, *, include_hashes: bool = True) -> str:
    """Render assembled context with numbered, source-linked blocks."""
    parts: list[str] = []
    for i, b in enumerate(ctx.blocks, start=1):
        header = f"[{i}] {b.source_path}:{b.start_line}-{b.end_line}"
        if include_hashes:
            header += f" (hash {b.content_hash[:12]})"
        parts.append(f"{header}\n{b.text}")
    return "\n\n".join(parts)


def blocks_as_dicts(ctx: AssembledContext) -> list[dict]:
    return [
        {
            "index": i,
            "source_path": b.source_path,
            "start_line": b.start_line,
            "end_line": b.end_line,
            "content_hash": b.content_hash,
            "chunk_id": b.chunk_id,
            "token_estimate": b.token_estimate,
            "score": b.score,
            "text": b.text,
        }
        for i, b in enumerate(ctx.blocks, start=1)
    ]