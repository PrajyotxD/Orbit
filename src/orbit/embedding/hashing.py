"""Deterministic hashing embedder — the offline default.

Feature hashing, not a language model. Tokens are hashed into a fixed-width
vector with signed hashing, accumulated with sublinear TF, then normalised.

The point is determinism: same text, same vector, every machine, forever. No
download, no inference cost, reproducible tests.

The limitation is synonymy — two passages saying the same thing in different
words stay far apart. That gap is why the lexical channel exists, and why
``--embedding-provider hf`` is the answer when paraphrase matters.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ..errors import EmbeddingError

__all__ = ["HashingEmbedder"]

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _hash_token(token: str, salt: str) -> int:
    """Stable across processes and platforms (unlike Python's ``hash``)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=salt.encode("utf-8")).digest()
    return int.from_bytes(digest, "big", signed=False)


class HashingEmbedder:
    """Feature-hashing embedder. Deterministic, offline, numpy-accelerated."""

    def __init__(
        self,
        dimension: int = 512,
        *,
        model: str = "orbit-hashing-v1",
        use_bigrams: bool = True,
        use_char_ngrams: bool = True,
        sublinear_tf: bool = True,
    ) -> None:
        if dimension <= 0:
            raise EmbeddingError("hashing embedder dimension must be > 0")
        self.dimension = dimension
        self.name = "hashing"
        self.model = model
        self.use_bigrams = use_bigrams
        self.use_char_ngrams = use_char_ngrams
        self.sublinear_tf = sublinear_tf
        self._salt = "orbit-hashing-v1"

    # ------------------------------------------------------------------ helpers
    def _features(self, text: str) -> list[str]:
        words = _tokens(text)
        feats: list[str] = list(words)
        if self.use_bigrams and len(words) > 1:
            feats.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
        if self.use_char_ngrams:
            joined = "".join(words)
            if len(joined) >= 4:
                # 4-gram fingerprint, cheap and surprisingly useful for code.
                feats.extend(joined[i : i + 4] for i in range(len(joined) - 3))
        return feats

    def _vector(self, text: str) -> list[float]:
        import numpy as np

        vec = np.zeros(self.dimension, dtype=np.float32)
        if not text or not text.strip():
            return vec.tolist()
        counts: dict[str, int] = {}
        for feat in self._features(text):
            counts[feat] = counts.get(feat, 0) + 1
        for feat, count in counts.items():
            h = _hash_token(feat, self._salt)
            idx = h % self.dimension
            sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
            weight = (1.0 + math.log(count)) if self.sublinear_tf else float(count)
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32).tolist()

    # ------------------------------------------------------------------- public
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def health(self) -> tuple[bool, str]:
        try:
            import numpy  # noqa: F401

            return True, "numpy available"
        except ImportError:  # pragma: no cover
            return False, "numpy is required for the hashing embedder"