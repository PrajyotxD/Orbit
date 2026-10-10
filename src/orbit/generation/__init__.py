"""Optional answer generation for Orbit.

The public surface is intentionally small::

    from orbit.generation import answer, build_generator, NullGenerator

Generation is disabled by default. When it is disabled, :func:`build_generator`
returns a :class:`NullGenerator`, which returns the retrieved evidence verbatim
and reports ``insufficient_evidence=True``. Orbit is fully functional without
any language model, no API key and no network access.

Prompt instructions reduce hallucination risk but do not eliminate it and do
not guarantee factual accuracy. What Orbit guarantees is provenance: the
citations attached to a generated answer are real retrieved records, filtered
to exactly the passages the generator was shown, and each can be re-validated
against its source file with :func:`orbit.evidence.validate_evidence`. A
citation says where the supporting material came from, not that the generated
claim is true.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .base import (
    INSUFFICIENCY_MARKERS,
    NO_LLM_NOTICE,
    Completion,
    EvidenceGenerator,
    Generator,
    blocks_from_hits,
    detect_insufficiency,
    evidence_from_blocks,
)
from .providers import (
    EchoGenerator,
    EvidenceOnlyGenerator,
    NullGenerator,
    OpenAICompatibleGenerator,
    build_generator,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import graph acyclic
    from ..engine import OrbitEngine
    from ..models import GeneratedAnswer

__all__ = [
    "Generator",
    "EvidenceGenerator",
    "Completion",
    "INSUFFICIENT_MARKERS",
    "NO_LLM_NOTICE",
    "detect_insufficiency",
    "evidence_from_blocks",
    "blocks_from_hits",
    "OpenAICompatibleGenerator",
    "NullGenerator",
    "EvidenceOnlyGenerator",
    "EchoGenerator",
    "build_generator",
    "answer",
]


def answer(
    engine: "OrbitEngine",
    question: str,
    *,
    limit: int | None = None,
    generator: Generator | None = None,
    config: Any | None = None,
) -> "GeneratedAnswer":
    """Retrieve, assemble a bounded context, then answer from it.

    This is the convenience path shared by the CLI's ``answer`` command and
    library callers. It performs exactly three steps:

    1. ``engine.search(question, limit=limit)`` — ranked passages with full
       provenance;
    2. :func:`orbit.context.assemble_context` — bounded, de-duplicated blocks,
       which is the exact set the generator is permitted to reason over;
    3. ``generator.generate_from_hits(question, kept_hits)`` — citations come
       from the retained retrieval records, not from anything the model wrote.

    ``generator`` overrides configuration; otherwise ``build_generator`` is
    called with ``config`` or ``engine.config.generation``. Pass
    ``NullGenerator`` (or ``--no-llm`` on the CLI) to answer with no model at
    all.
    """
    from ..context import assemble_context

    gen = generator if generator is not None else build_generator(
        config if config is not None else engine.config.generation
    )
    result = engine.search(question, limit=limit)
    assembled = assemble_context(result, engine.config.context)
    kept_ids = {b.chunk_id for b in assembled.blocks}
    kept_hits: Sequence[Any] = [h for h in result.hits if h.evidence.chunk_id in kept_ids]
    # Order by the assembler's ranking so the text the generator saw and the
    # citation list it returns are the same sequence.
    order = {b.chunk_id: i for i, b in enumerate(assembled.blocks)}
    ordered = sorted(kept_hits, key=lambda h: order.get(h.evidence.chunk_id, 0))
    return gen.generate_from_hits(question, ordered)