"""Deterministic chunking.

Lines are never split, so ``start_line``/``end_line`` always point at real lines
you can go re-read. Boundaries depend only on the text and the config -- never on
time, randomness or dict ordering -- so re-indexing an unchanged file reproduces
the same chunks byte for byte.

``content_hash`` covers the start line as well as the text, so identical prose
at two locations gets two identities instead of colliding.

Token counts are ``ceil(chars / tokens_per_char)`` estimates, which keeps the
chunker independent of any model tokenizer. Use a real tokenizer for context
budgets, not for changing where chunks split.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

from .config import ChunkingConfig
from .errors import ChunkingError
from .models import SourceChunk, sha256_text

__all__ = ["estimate_tokens", "chunk_document", "iter_chunks"]

# Lines that look like the start of a top-level definition in common languages.
_SYMBOL_START = re.compile(
    r"^\s*(?:"
    r"def\s+\w+"              # python
    r"|class\s+\w+"           # python, java, c++, c#
    r"|(?:public|private|protected|static|final|abstract|\s)*\w[\w<>\[\], \*]*\s+\w+\s*\("
    r"|function\s+\w+"        # js/php
    r"|(?:func|fn)\s+\w+"     # go, rust
    r"|struct\s+\w+"          # c, rust
    r"|impl\s+\w+"            # rust
    r"|interface\s+\w+"       # go, java, ts
    r"|module\s+\w+"          # rust
    r"|package\s+\w+"         # java
    r"|namespace\s+\w+"       # c++
    r"|#\s*region"            # c#
    r"|//\s*\w+"              # comment-anchored sections (c/c++/java)
    r"|\w+\s*=\s*(?:function|lambda)"  # js assignment
    r")"
)

_HEADING = re.compile(r"^(#{1,6})\s+\S")


def estimate_tokens(text: str, tokens_per_char: float = 4.0) -> int:
    """Estimate the token count of ``text``.

    This is a character-ratio heuristic, not a real tokenizer. It is used only
    to bound chunk size and to approximate context budgets; anything that needs
    exact model token counts must supply a real tokenizer.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / tokens_per_char))


def _lines_for_chars(lines: list[str], start: int, end: int, budget_chars: int) -> int:
    """How many lines back from ``end`` (toward ``start``) fit in ``budget_chars``."""
    total = 0
    count = 0
    for idx in range(end - 1, start, -1):
        total += len(lines[idx]) + 1
        if total > budget_chars:
            break
        count += 1
    return count


class _Chunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def _estimate(self, text: str) -> int:
        return estimate_tokens(text, self.config.tokens_per_char)

    def _is_boundary_candidate(self, line: str, index: int) -> bool:
        """True when ``line`` looks like a natural start for a new chunk.

        Only meaningful when a chunk is already at least ``min_chunk_tokens``.
        We avoid breaking at the very top of a file.
        """
        if self.config.respect_blank_lines and not line.strip():
            return True
        if self.config.respect_symbol_boundaries and _SYMBOL_START.match(line):
            return True
        if self.config.respect_symbol_boundaries and _HEADING.match(line):
            return True
        return False

    def run(self, text: str) -> Iterator[SourceChunk]:
        cfg = self.config
        max_tokens = cfg.max_tokens
        min_tokens = cfg.min_chunk_tokens
        overlap_chars = cfg.overlap_chars
        max_chars = cfg.max_chars_effective

        # Normalise line endings so offsets/line numbers are platform-stable.
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        line_tokens = [self._estimate(line) for line in lines]
        n_lines = len(lines)

        chunks: list[SourceChunk] = []
        ordinal = 0
        start = 0

        while start < n_lines:
            # Grow a window [start, end) until adding the next line would exceed
            # the budget. `boundary` tracks the last natural split point strictly
            # inside the window (and at least `min_chunk_tokens` in).
            end = start
            tokens = 0
            chars = 0
            boundary = -1
            while end < n_lines:
                lt = line_tokens[end]
                if tokens > 0 and (tokens + lt > max_tokens or chars + len(lines[end]) > max_chars):
                    break
                if (
                    end > start
                    and tokens >= min_tokens
                    and self._is_boundary_candidate(lines[end], end)
                ):
                    boundary = end
                tokens += lt
                chars += len(lines[end]) + 1
                end += 1

            # If the window is a single oversized line, force at least one line.
            if end == start:
                end = start + 1
                tokens = line_tokens[start]
                chars = len(lines[start])

            # Choose the cut: a natural boundary if we found one, else hard cut.
            cut = boundary if boundary > start else end

            body = "\n".join(lines[start:cut]).rstrip("\n")
            if body.strip():
                chunks.append(
                    SourceChunk(
                        ordinal=ordinal,
                        text=body,
                        start_line=start + 1,
                        end_line=cut,
                        start_offset=start,
                        end_offset=cut,
                        token_estimate=self._estimate(body),
                        metadata={"window_tokens": tokens, "hard_cut": boundary <= start},
                    )
                )
                ordinal += 1

            if cut >= n_lines:
                break

            # Next window begins `overlap_chars` worth of lines *before* the cut,
            # so consecutive chunks share context, but never before `cut`'s
            # predecessor boundary and always with forward progress.
            next_start = cut
            if overlap_chars > 0 and cut - start > 1:
                back_chars = 0
                k = cut - 1
                # Do not overlap the whole previous chunk; keep at least the
                # content from `cut - overlap` onward.
                min_start = max(start + 1, cut - _lines_for_chars(lines, start, cut, overlap_chars))
                while k >= min_start and back_chars < overlap_chars:
                    back_chars += len(lines[k]) + 1
                    k -= 1
                next_start = k + 1
                if next_start <= start:
                    next_start = start + 1
            start = next_start

        yield from chunks


def iter_chunks(text: str, config: ChunkingConfig | None = None) -> Iterator[SourceChunk]:
    """Return an iterator of :class:`SourceChunk` for ``text``."""
    return _Chunker(config or ChunkingConfig()).run(text)


def chunk_document(text: str, config: ChunkingConfig | None = None) -> list[SourceChunk]:
    """Split ``text`` into deterministic chunks.

    Raises :class:`ChunkingError` only for programmer error (e.g. an empty
    config); empty input yields an empty list rather than an exception so that
    empty files do not abort an index run.
    """
    cfg = config or ChunkingConfig()
    if cfg.overlap_tokens >= cfg.max_tokens:
        raise ChunkingError("overlap_tokens must be smaller than max_tokens")
    if not text:
        return []
    return list(iter_chunks(text, cfg))