"""OpenAI-compatible HTTP embedding provider (optional, explicit opt-in).

Only reachable when ``embedding.base_url`` is set. Nothing is sent to any remote
host otherwise. The key is read from the process environment at call time via
the name given in ``embedding.api_key_env``; it is never written to the config
file, the database, or logs.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import EmbeddingError, EmbeddingUnavailableError

__all__ = ["OpenAICompatibleEmbedder"]


class OpenAICompatibleEmbedder:
    """Calls ``POST {base_url}/embeddings`` with an OpenAI-shaped payload."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimension: int = 0,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> None:
        if not base_url:
            raise EmbeddingUnavailableError(
                "the 'openai_compatible' embedding provider requires embedding.base_url"
            )
        self.name = "openai_compatible"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        if not self._dimension:
            raise EmbeddingError(
                "embedding dimension unknown until the first call; ensure at least one "
                "document is indexed before running vector-only retrieval"
            )
        return self._dimension

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingUnavailableError(
                "the 'openai_compatible' provider needs httpx: pip install 'orbit[llm]'"
            ) from exc

        url = self.base_url if self.base_url.endswith("/embeddings") else f"{self.base_url}/embeddings"
        try:
            resp = httpx.post(
                url,
                headers=self._headers(),
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise EmbeddingError(f"embedding request failed: {exc}", detail=url) from exc

        data = payload.get("data") or []
        if len(data) != len(texts):
            raise EmbeddingError(
                f"embedding endpoint returned {len(data)} vectors for {len(texts)} inputs"
            )
        vectors = [item["embedding"] for item in data]
        if not self._dimension and vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(texts[i : i + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]

    def health(self) -> tuple[bool, str]:
        try:
            vec = self._post(["ping"])
            if not self._dimension:
                self._dimension = len(vec[0])
            return True, f"reachable, dimension={self._dimension}"
        except Exception as exc:
            return False, str(exc)[:200]