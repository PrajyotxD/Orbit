"""Generator abstractions for the optional answer layer.

Retrieval is complete without a language model; this layer sits on top and can be
swapped without touching the index or the retriever.

The system prompt asks the model to stick to the supplied evidence and to say so
when the evidence is thin. That reduces hallucination risk. It does not remove
it, and it is not a guarantee.

What Orbit does guarantee is provenance: every citation on a
:class:`~orbit.models.GeneratedAnswer` is a real record that was retrieved and
shown to the generator, and it stays re-verifiable with
:func:`orbit.evidence.validate_evidence`. A citation says where the material came
from, not that the claim built on it is true.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import GenerationConfig
from ..models import (
    ContextBlock,
    Evidence,
    GeneratedAnswer,
    RetrievalMethod,
    SearchHit,
    ValidationState,
)

__all__ = [
    "Generator",
    "EvidenceGenerator",
    "Completion",
    "INSUFFICIENCY_MARKERS",
    "detect_insufficiency",
    "evidence_from_blocks",
    "blocks_from_hits",
    "NO_LLM_NOTICE",
]


INSUFFICIENCY_MARKERS: tuple[str, ...] = (
    "insufficient evidence",
    "evidence is insufficient",
    "not enough evidence",
    "no evidence",
    "does not contain enough",
    "does not support",
    "cannot be answered",
    "cannot answer",
    "not stated in the evidence",
    "no relevant evidence",
)

NO_LLM_NOTICE = (
    "Orbit has no generator configured, so no language model was consulted. "
    "The passages below are the retrieved evidence, verbatim. Any answer must be "
    "derived from them by a human reader; Orbit itself is not asserting that they "
    "answer the question."
)


def detect_insufficiency(text: str) -> bool:
    """Heuristically decide whether a generated answer declares itself unsupported.

    This is a substring check, not a semantic judgement. It exists so callers can
    branch on ``insufficient_evidence`` without paying for a second model call;
    it can both miss a refusal and flag an honest negative statement about the
    evidence. Treat it as a hint, never as proof.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in INSUFFICIENCY_MARKERS)


def evidence_from_blocks(
    blocks: Sequence[ContextBlock],
    evidence: Sequence[Evidence] | None = None,
) -> tuple[Evidence, ...]:
    """Pair each supplied block with a real :class:`Evidence` record.

    ``evidence`` is the authority when supplied: it is indexed by ``chunk_id``
    and filtered down to exactly the block ids that were handed to the
    generator, so a caller cannot inflate a citation list with records that were
    never shown to the model. That filtering is what makes "citations contain
    only what was supplied" a structural property rather than a promise.

    When ``evidence`` is ``None`` the records are reconstructed from the blocks
    themselves. Those reconstructions carry ``doc_id=""`` because
    :class:`ContextBlock` does not retain the document id, and they are
    therefore weaker than a retrieved record — callers holding real hits should
    pass them via :func:`blocks_from_hits`.
    """
    if evidence is None:
        return tuple(
            Evidence(
                chunk_id=b.chunk_id,
                doc_id="",
                source_path=b.source_path,
                start_line=b.start_line,
                end_line=b.end_line,
                content_hash=b.content_hash,
                retrieval_method=RetrievalMethod.HYBRID,
                score=b.score,
                rank=b.rank,
                validation=ValidationState.NOT_VALIDATED,
            )
            for b in blocks
        )
    by_id = {e.chunk_id: e for e in evidence}
    return tuple(
        by_id[b.chunk_id] for b in blocks if b.chunk_id in by_id
    )


def blocks_from_hits(hits: Sequence[SearchHit]) -> tuple[ContextBlock, ...]:
    """Project ranked hits onto context blocks, preserving their order."""
    return tuple(
        ContextBlock(
            text=h.text,
            source_path=h.evidence.source_path,
            start_line=h.evidence.start_line,
            end_line=h.evidence.end_line,
            content_hash=h.evidence.content_hash,
            chunk_id=h.evidence.chunk_id,
            token_estimate=max(1, len(h.text) // 4),
            score=h.evidence.score,
            rank=h.evidence.rank,
        )
        for h in hits
    )


@dataclass(frozen=True)
class Completion:
    """Raw provider response, before it is combined with evidence records."""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    insufficient: bool | None = None


class Generator(abc.ABC):
    """The abstract base every answer source must satisfy.

    A generator is anything that can turn a question plus a bounded set of
    retrieved passages into a :class:`~orbit.models.GeneratedAnswer` whose
    ``citations`` name only the passages it was actually given.
    """

    #: Stable machine name of this generator, e.g. ``"null"``.
    name: str = "generator"
    #: Model identifier reported on produced answers. Empty when no model is used.
    model: str = ""

    @abc.abstractmethod
    def generate(
        self,
        question: str,
        context_blocks: Sequence[ContextBlock],
        *,
        evidence: Sequence[Evidence] | None = None,
    ) -> GeneratedAnswer:
        """Answer ``question`` using only ``context_blocks``.

        ``evidence``, when supplied, is the authority for the citation list and
        is filtered to the block ids present in ``context_blocks``. Implementations
        must never add a citation that is not backed by a supplied block.
        """

    @abc.abstractmethod
    def health(self) -> tuple[bool, str]:
        """Return ``(usable, detail)``. Must not perform network I/O."""

    def generate_from_hits(
        self,
        question: str,
        hits: Sequence[SearchHit],
    ) -> GeneratedAnswer:
        """Answer from ranked hits, citing the real retrieved records.

        Preferred over calling :meth:`generate` with hand-built blocks: it keeps
        the full :class:`Evidence` objects retrieval produced, including their
        document ids and validation state, instead of reconstructing weaker
        records from context blocks.
        """
        ordered = tuple(hits)
        return self.generate(
            question,
            blocks_from_hits(ordered),
            evidence=[h.evidence for h in ordered],
        )


class EvidenceGenerator(Generator):
    """Shared prompt rendering, evidence bookkeeping and health reporting.

    Subclasses implement only :meth:`_complete`, which turns a rendered prompt
    into text. Everything that makes an answer *attributable* — the prompt, the
    citation list, the insufficiency flag — lives here so no provider can
    accidentally produce an uncited answer.
    """

    name: str = "evidence"

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.model = self.config.model

    # ------------------------------------------------------------------ prompt
    def render_prompt(self, blocks: Sequence[ContextBlock], question: str) -> str:
        """Render the evidence payload and question into one user message.

        Each passage is numbered so the system prompt's "[n]" citation
        convention maps onto a concrete record the caller can re-verify. Paths
        and truncated hashes are included because a model asked to cite sources
        needs those sources in front of it, not just the prose.

        The prompt is an instruction, not a guarantee: it lowers the chance of
        hallucination but cannot prevent it.
        """
        if not blocks:
            return (
                f"{self.config.system_prompt}\n\n"
                "EVIDENCE:\n(none — no passages were retrieved)\n\n"
                f"QUESTION: {question}\n"
                "The evidence set is empty. State that the question cannot be "
                "answered from the retrieved material and stop."
            )
        parts: list[str] = []
        for i, block in enumerate(blocks, start=1):
            parts.append(
                f"[{i}] {block.source_path}:{block.start_line}-{block.end_line}\n"
                f"    content_hash={block.content_hash[:16]}\n"
                f"{block.text}"
            )
        return (
            f"{self.config.system_prompt}\n\n"
            "EVIDENCE:\n" + "\n\n".join(parts) + f"\n\nQUESTION: {question}\n"
        )

    # -------------------------------------------------------------- generation
    @abc.abstractmethod
    def _complete(self, prompt: str) -> Completion:
        """Produce raw text for a rendered prompt, or raise ``GenerationError``."""

    def generate(
        self,
        question: str,
        context_blocks: Sequence[ContextBlock],
        *,
        evidence: Sequence[Evidence] | None = None,
    ) -> GeneratedAnswer:
        blocks = tuple(context_blocks)
        citations = evidence_from_blocks(blocks, evidence)
        prompt = self.render_prompt(blocks, question)
        completion = self._complete(prompt)
        insufficient = (
            completion.insufficient
            if completion.insufficient is not None
            else detect_insufficiency(completion.text)
        )
        return GeneratedAnswer(
            text=completion.text,
            citations=citations,
            model=self.model,
            insufficient_evidence=bool(insufficient),
            usage=dict(completion.usage),
            raw=completion.raw,
        )

    def generate_from_hits(
        self,
        question: str,
        hits: Sequence[SearchHit],
    ) -> GeneratedAnswer:
        """Generate from ranked hits so citations are the retrieved records.

        Inherited behaviour from :class:`Generator`; kept as an explicit
        override point for generators that want to specialise it.
        """
        return super().generate_from_hits(question, hits)

    # ------------------------------------------------------------------ health
    def health(self) -> tuple[bool, str]:
        """Report whether this generator can be used, without side effects."""
        return True, f"{self.name} ready (model={self.model or 'unset'})"