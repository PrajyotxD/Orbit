"""Embedding providers and the factory that selects one.

Selection is driven entirely by :class:`~orbit.config.EmbeddingConfig`. The
factory never reaches the network on its own: remote providers require an
explicit ``base_url``.
"""

from __future__ import annotations

from ..config import EmbeddingConfig
from ..errors import EmbeddingUnavailableError
from .base import EmbedderResult, EmbeddingProvider, as_float_list, tokenize_basic
from .hashing import HashingEmbedder

__all__ = [
    "EmbeddingProvider",
    "EmbedderResult",
    "HashingEmbedder",
    "tokenize_basic",
    "as_float_list",
    "build_embedder",
    "NullEmbedder",
]


class NullEmbedder:
    """A provider that does nothing. Used for explicit lexical-only operation."""

    name = "none"
    model = "none"
    dimension = 0

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return []

    def embed_query(self, text: str) -> list[float]:
        return []

    def health(self) -> tuple[bool, str]:
        return True, "lexical-only mode (no embedding provider)"


def build_embedder(config: EmbeddingConfig) -> EmbeddingProvider:
    """Instantiate the provider named by ``config``.

    Raises :class:`EmbeddingUnavailableError` only for genuinely unusable
    configuration (e.g. a remote provider with no base URL). Transient provider
    failures surface later via :meth:`EmbeddingProvider.health`, and the engine
    then degrades to lexical-only.
    """
    provider = config.provider
    if provider == "none":
        return NullEmbedder()
    if provider == "hashing":
        return HashingEmbedder(dimension=config.dimension)
    if provider == "hf":
        from .hf import HFEmbedder

        return HFEmbedder(
            model=config.model,
            model_path=config.model_path,
            device=config.device,
            max_length=config.max_length,
            normalize=config.normalize,
            batch_size=config.batch_size,
        )
    if provider == "openai_compatible":
        from .openai_compat import OpenAICompatibleEmbedder

        if not config.base_url:
            raise EmbeddingUnavailableError(
                "embedding.provider='openai_compatible' requires embedding.base_url; "
                "Orbit never contacts a remote host implicitly"
            )
        return OpenAICompatibleEmbedder(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key(),
            dimension=config.dimension if config.dimension > 0 else 0,
            batch_size=config.batch_size,
        )
    raise EmbeddingUnavailableError(f"unknown embedding provider {provider!r}")