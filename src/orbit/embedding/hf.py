"""HuggingFace Transformers embedding provider (optional).

Uses ``transformers.AutoModel`` + mean pooling + L2 normalisation — the standard
recipe for sentence embeddings — rather than depending on
``sentence-transformers``, so Orbit keeps a smaller dependency surface.

Requires the ``orbit[embeddings]`` extra and a local model directory or a Hub
model id. If either is unavailable this provider raises
:class:`~orbit.errors.EmbeddingUnavailableError` and the engine degrades to
lexical-only rather than crashing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from ..errors import EmbeddingError, EmbeddingUnavailableError

__all__ = ["HFEmbedder"]


class HFEmbedder:
    """Mean-pooled Transformer embedder."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        model_path: str = "",
        device: str = "auto",
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.name = "hf"
        self.model = model_path or model
        self.device = device
        self.max_length = max_length
        self.normalize = normalize
        self.batch_size = batch_size
        self._impl = None
        self._dimension = 0

    def _ensure_loaded(self) -> None:
        if self._impl is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "the 'hf' embedding provider needs the 'embeddings' extra: "
                "pip install 'orbit[embeddings]'",
                detail=str(exc),
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModel.from_pretrained(self.model)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"could not load embedding model {self.model!r}; provide a local "
                f"path via embedding.model_path or pre-download the weights",
                detail=str(exc)[:200],
            ) from exc

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = model.to(device)
        model.eval()
        self._impl = (torch, tokenizer, model)
        # Infer dimension from a probe.
        self._dimension = int(model.config.hidden_size)

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._ensure_loaded()
        return self._dimension

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        self._ensure_loaded()
        _, tokenizer, model = self._impl  # type: ignore[misc]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.extend(pooled.detach().cpu().float().tolist())
        return out

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def health(self) -> tuple[bool, str]:
        try:
            self._ensure_loaded()
            return True, f"loaded {self.model} on {self.device}"
        except EmbeddingError as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover
            return False, str(exc)[:200]


def default_offline_env() -> bool:  # pragma: no cover - helper
    """True when HF should be forced offline (no Hub round-trips)."""
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1"