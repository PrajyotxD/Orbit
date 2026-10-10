"""Concrete generators.

``OpenAICompatibleGenerator`` talks to anything speaking the OpenAI
``/chat/completions`` schema (vLLM, llama.cpp, Ollama's shim, LM Studio, and so
on). The key comes from the environment at call time and is never written to
config or echoed in errors; construction does no I/O.

``NullGenerator`` / ``EchoGenerator`` generate nothing and return the retrieved
evidence verbatim with ``insufficient_evidence=True``. They exist so the whole
pipeline is demonstrable with no LLM configured -- an answer built this way can't
invent anything, since nothing was generated.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..config import GenerationConfig
from ..errors import GenerationError
from ..models import ContextBlock, Evidence, GeneratedAnswer
from .base import NO_LLM_NOTICE, Completion, EvidenceGenerator, Generator

__all__ = [
    "OpenAICompatibleGenerator",
    "NullGenerator",
    "EvidenceOnlyGenerator",
    "EchoGenerator",
    "build_generator",
]


class OpenAICompatibleGenerator(EvidenceGenerator):
    """Chat-completions client for OpenAI-compatible HTTP endpoints."""

    name = "openai_compatible"

    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        if not config.base_url:
            raise GenerationError(
                "generation.base_url must be set for an OpenAI-compatible generator",
                detail="set generation.base_url in config, or leave generation disabled",
            )

    @property
    def endpoint(self) -> str:
        return self.config.base_url.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        """Build request headers, reading the API key from the environment now.

        The key is resolved on every call rather than cached at construction so
        that rotating or late-exported credentials are picked up, and so that
        merely constructing the generator never reads credentials.
        """
        headers = {"Content-Type": "application/json"}
        key = self.config.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def health(self) -> tuple[bool, str]:
        """Report local readiness only. Never contacts the endpoint."""
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on install extras
            return False, f"httpx is not installed ({exc})"
        notes = [f"endpoint={self.endpoint}", f"model={self.config.model}"]
        if self.config.api_key_env and not self.config.api_key():
            notes.append(f"warning: ${self.config.api_key_env} is not set in the environment")
        if not self.config.api_key_env:
            notes.append("no api_key_env configured; endpoint must accept unauthenticated requests")
        return True, "; ".join(notes)

    def _complete(self, prompt: str) -> Completion:
        try:
            import httpx
        except ImportError as exc:
            raise GenerationError(
                "httpx is required for HTTP generation",
                detail="install the 'llm' extra: pip install httpx",
            ) from exc

        try:
            response = httpx.post(
                self.endpoint,
                json=self._payload(prompt),
                headers=self._headers(),
                timeout=self.config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise GenerationError(
                f"generation request timed out after {self.config.timeout_seconds}s",
                detail=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(
                "generation request failed", detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise GenerationError(
                f"generation endpoint returned HTTP {response.status_code}",
                detail=response.text[:500],
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GenerationError(
                "generation endpoint returned a non-JSON body", detail=response.text[:500]
            ) from exc

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(
                "generation response is missing choices[0].message.content",
                detail=json.dumps(payload, default=str)[:500],
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise GenerationError(
                "generation response contained empty content",
                detail=json.dumps(payload, default=str)[:500],
            )
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return Completion(text=text, usage=usage, raw=json.dumps(payload, default=str))


class NullGenerator(EvidenceGenerator):
    """No-model generator: returns the supplied evidence verbatim.

    Used whenever generation is disabled or misconfigured, which makes the
    "Orbit works with zero LLM configuration" claim testable rather than
    aspirational. It performs no network I/O, needs no configuration, and
    cannot fabricate anything because it never generates anything.
    """

    name = "null"

    def __init__(
        self,
        config: GenerationConfig | None = None,
        reason: str = "generation is disabled",
    ) -> None:
        super().__init__(config)
        self.reason = reason
        self.model = "none"

    def health(self) -> tuple[bool, str]:
        return True, f"null generator active ({self.reason}); no model is contacted"

    def _complete(self, prompt: str) -> Completion:
        # Unreachable: generate() is overridden below and never calls _complete.
        # Kept as an explicit guard so a future refactor cannot silently make
        # the null generator issue a model call.
        raise GenerationError(
            "NullGenerator never contacts a model", detail=self.reason
        )

    def render_evidence(self, blocks: Sequence[ContextBlock]) -> str:
        return "\n\n".join(block.text for block in blocks)

    def generate(
        self,
        question: str,
        context_blocks: Sequence[ContextBlock],
        *,
        evidence: Sequence[Evidence] | None = None,
    ) -> GeneratedAnswer:
        from .base import evidence_from_blocks

        blocks = tuple(context_blocks)
        if blocks:
            body = (
                f"Question: {question}\n\n"
                f"Retrieved evidence ({len(blocks)} passage(s)):\n\n"
                f"{self.render_evidence(blocks)}"
            )
        else:
            body = (
                f"Question: {question}\n\n"
                "Retrieved evidence: none. No passage in the index matched."
            )
        return GeneratedAnswer(
            text=f"{NO_LLM_NOTICE}\n\n{body}",
            citations=evidence_from_blocks(blocks, evidence),
            model="none",
            insufficient_evidence=True,
            usage={"mode": "no_llm", "passages": len(blocks)},
        )


#: Readability alias: this generator is the "evidence only, no model" mode.
EvidenceOnlyGenerator = NullGenerator


class EchoGenerator(NullGenerator):
    """Null generator variant that labels each passage with its source line.

    Useful in demos and smoke tests where the citation format should be visible
    in the rendered text and not only in the returned records.
    """

    name = "echo"

    def render_evidence(self, blocks: Sequence[ContextBlock]) -> str:
        return "\n\n".join(
            f"[{i}] {b.source_path}:{b.start_line}-{b.end_line}\n{b.text}"
            for i, b in enumerate(blocks, start=1)
        )


def build_generator(config: GenerationConfig | None = None) -> Generator:
    """Pick a generator from configuration without touching the network.

    Construction is offline by contract: no HTTP client is created, no endpoint
    is probed, no credential value is read. A caller can therefore build a
    generator in a sandbox with no egress and still get a working object.
    Disabled generation or a missing base URL always yields a
    :class:`NullGenerator` rather than an error, so a misconfigured LLM setting
    degrades to "no LLM" instead of breaking search.
    """
    cfg = config or GenerationConfig()
    if not cfg.enabled:
        return NullGenerator(cfg, reason="generation.enabled is false")
    provider = (cfg.provider or "openai_compatible").lower()
    if provider == "null":
        return NullGenerator(cfg, reason=f"generation.provider is {provider!r}")
    if not cfg.base_url:
        return NullGenerator(cfg, reason="generation.base_url is empty")
    if provider != "openai_compatible":
        return NullGenerator(cfg, reason=f"unknown generation provider {provider!r}")
    return OpenAICompatibleGenerator(cfg)