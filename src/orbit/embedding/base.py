"""Embedding provider interface.

Three providers implement it: the built-in :class:`~orbit.embedding.hashing.HashingEmbedder`
(default, offline), :class:`~orbit.embedding.hf.HFEmbedder`, and
:class:`~orbit.embedding.openai_compat.OpenAICompatibleEmbedder`, which only gets
constructed when a base URL is set.

Providers declare ``dimension`` and ``model``; the dimension is stored in the
index and vectors of differing width are rejected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..errors import EmbeddingError

__all__ = ["EmbeddingProvider", "EmbedderResult", "tokenize_basic"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The contract every embedder implements."""

    name: str
    model: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    def health(self) -> tuple[bool, str]:
        """Return ``(ok, detail)`` describing whether the provider can be used."""
        ...


class EmbedderResult:
    """Convenience wrapper carrying vectors plus any provider metadata."""

    def __init__(self, vectors: list[list[float]], provider: str, model: str, dim: int) -> None:
        self.vectors = vectors
        self.provider = provider
        self.model = model
        self.dim = dim


def tokenize_basic(text: str) -> list[str]:
    """A minimal, dependency-free tokenizer: lowercase word tokens.

    Used by the hashing embedder and by lexical pre-processing. It splits on any
    non-alphanumeric character and keeps Unicode letters/digits.
    """
    import re

    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def as_float_list(vec: Any, *, expect_dim: int | None = None, provider: str = "embedding") -> list[float]:
    """Coerce a provider result to a validated ``list[float]``."""
    try:
        floats = [float(x) for x in vec]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"provider {provider!r} returned a non-numeric vector", detail=str(exc)
        ) from exc
    if expect_dim is not None and len(floats) != expect_dim:
        raise EmbeddingError(
            f"provider {provider!r} returned dimension {len(floats)}, expected {expect_dim}"
        )
    return floats