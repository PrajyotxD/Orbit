"""Context assembly: budget, dedupe, ordering, provenance preservation."""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit import OrbitConfig, OrbitEngine
from orbit.config import ContextBudgetConfig
from orbit.context import assemble_context, blocks_as_dicts, render_context
from orbit.models import AssembledContext, SearchResult


def _result(engine: OrbitEngine, query: str, limit: int = 10) -> SearchResult:
    return engine.search(query, limit=limit)


# ------------------------------------------------------------------- basics
def test_empty_result_yields_empty_context() -> None:
    from orbit.models import RetrievalMethod

    empty = SearchResult(query="q", hits=(), retrieval_method=RetrievalMethod.LEXICAL)
    ctx = assemble_context(empty, ContextBudgetConfig())
    assert isinstance(ctx, AssembledContext)
    assert ctx.blocks == ()
    assert ctx.total_token_estimate == 0
    assert ctx.truncated is False
    assert ctx.render() == ""


def test_context_blocks_preserve_source_references(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    result = _result(eng, "quaternion")
    assert result.hits
    ctx = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0))
    assert ctx.blocks
    by_chunk = {h.evidence.chunk_id: h for h in result.hits}
    for block in ctx.blocks:
        hit = by_chunk[block.chunk_id]
        assert block.source_path == hit.evidence.source_path
        assert block.start_line == hit.evidence.start_line
        assert block.end_line == hit.evidence.end_line
        assert block.content_hash == hit.evidence.content_hash
        assert block.text == hit.text
        assert block.token_estimate > 0
        assert block.rank == hit.evidence.rank


def test_rendered_context_contains_path_and_lines(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(_result(eng, "quaternion"), ContextBudgetConfig(min_score_ratio=0.0))
    rendered = ctx.render()
    for block in ctx.blocks:
        assert f"{block.source_path}:{block.start_line}-{block.end_line}" in rendered
        assert block.text[:40] in rendered
        assert block.content_hash[:12] in rendered


def test_render_context_helper_and_blocks_as_dicts(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(_result(eng, "quaternion"), ContextBudgetConfig(min_score_ratio=0.0))
    rendered = render_context(ctx)
    assert rendered
    assert "[1]" in rendered
    dicts = blocks_as_dicts(ctx)
    assert len(dicts) == len(ctx.blocks)
    assert dicts[0]["index"] == 1
    assert dicts[0]["source_path"] == ctx.blocks[0].source_path
    assert dicts[0]["text"] == ctx.blocks[0].text


# -------------------------------------------------------------------- budget
def test_token_budget_is_enforced(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(
        _result(eng, "scheduler threads compaction telemetry allocate", limit=20),
        ContextBudgetConfig(token_budget=40, min_score_ratio=0.0),
    )
    assert ctx.total_token_estimate <= ctx.token_budget
    assert ctx.truncated is True
    assert len(ctx.blocks) < 20


def test_max_blocks_is_enforced(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    result = _result(eng, "scheduler threads compaction telemetry allocate", limit=20)
    ctx = assemble_context(result, ContextBudgetConfig(max_blocks=2, min_score_ratio=0.0))
    assert len(ctx.blocks) <= 2
    assert ctx.truncated is True


def test_zero_blocks_budget_produces_nothing(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(
        _result(eng, "quaternion"), ContextBudgetConfig(token_budget=1, min_score_ratio=0.0)
    )
    assert ctx.blocks == ()


def test_custom_token_counter_is_used(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    ctx = assemble_context(
        _result(eng, "quaternion"), ContextBudgetConfig(min_score_ratio=0.0), token_counter=counter
    )
    assert calls
    assert ctx.total_token_estimate == sum(b.token_estimate for b in ctx.blocks)


# -------------------------------------------------------------------- dedupe
def test_identical_blocks_are_deduped() -> None:
    from orbit.models import Evidence, RetrievalMethod, SearchHit

    ev = Evidence(
        chunk_id="a:0",
        doc_id="a",
        source_path="a.md",
        start_line=1,
        end_line=2,
        content_hash="deadbeef",
        retrieval_method=RetrievalMethod.LEXICAL,
        score=1.0,
        rank=1,
    )
    result = SearchResult(
        query="q",
        hits=(
            SearchHit(evidence=ev, text="same text"),
            SearchHit(evidence=ev, text="same text"),
        ),
        retrieval_method=RetrievalMethod.LEXICAL,
    )
    ctx = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0))
    assert len(ctx.blocks) == 1
    assert ctx.dropped_duplicates == 1


def test_near_duplicate_blocks_are_deduped() -> None:
    from orbit.models import Evidence, RetrievalMethod, SearchHit

    text = "the quick brown fox jumps over the lazy dog every single morning"
    hits = []
    for i, (cid, h) in enumerate(
        [(f"a:{i}", f"h{i}") for i in range(2)], start=0
    ):
        ev = Evidence(
            chunk_id=cid,
            doc_id="a",
            source_path="a.md",
            start_line=1,
            end_line=2,
            content_hash=h,
            retrieval_method=RetrievalMethod.LEXICAL,
            score=1.0 - i * 0.01,
            rank=i + 1,
        )
        hits.append(SearchHit(evidence=ev, text=text if i == 0 else text))
    result = SearchResult(query="q", hits=tuple(hits), retrieval_method=RetrievalMethod.LEXICAL)
    ctx = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0))
    assert len(ctx.blocks) == 1
    assert ctx.dropped_duplicates == 1

    ctx2 = assemble_context(
        result, ContextBudgetConfig(min_score_ratio=0.0, dedupe_similarity=False)
    )
    assert len(ctx2.blocks) == 2


def test_distinct_blocks_are_not_deduped(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(
        _result(eng, "the OR a OR of", limit=20),
        ContextBudgetConfig(min_score_ratio=0.0, dedupe_similarity=False),
    )
    hashes = [b.content_hash for b in ctx.blocks]
    assert len(hashes) == len(set(hashes))


# ----------------------------------------------------------------- ordering
def test_score_ordering_is_descending(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(_result(eng, "the OR a OR of", limit=20), ContextBudgetConfig(min_score_ratio=0.0))
    scores = [b.score for b in ctx.blocks]
    assert scores == sorted(scores, reverse=True)


def test_document_ordering_groups_by_path(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = assemble_context(
        _result(eng, "the OR a OR of", limit=20),
        ContextBudgetConfig(ordering="document", min_score_ratio=0.0),
    )
    keys = [(b.source_path, b.start_line) for b in ctx.blocks]
    assert keys == sorted(keys)


def test_ordering_is_deterministic(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    result = _result(eng, "the OR a OR of", limit=20)
    a = [b.chunk_id for b in assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0)).blocks]
    b = [b.chunk_id for b in assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0)).blocks]
    assert a == b


# ----------------------------------------------------------------- relevance
def test_low_scoring_blocks_are_dropped_as_irrelevant(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    result = _result(eng, "the OR a OR of", limit=20)
    lenient = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0))
    strict = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.99))
    assert len(strict.blocks) <= len(lenient.blocks)
    assert strict.dropped_irrelevant >= lenient.dropped_irrelevant


def test_no_padding_with_irrelevant_material(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "relevant.md").write_text("# R\n\nQuokka scheduling ring buffer allocation policy.\n")
    (root / "noise.txt").write_text("Lorem ipsum dolor sit amet consectetur adipiscing elit sed.\n")
    eng = make_engine()
    eng.index([root])
    ctx = eng.assemble_context("quokka scheduling ring buffer", limit=10)
    assert ctx.blocks
    for block in ctx.blocks:
        assert "quokka" in block.text.lower() or "scheduling" in block.text.lower() or (
            "ring" in block.text.lower()
        )


# --------------------------------------------------------------- engine API
def test_engine_assemble_context_end_to_end(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    ctx = eng.assemble_context("quaternion", limit=5)
    assert isinstance(ctx, AssembledContext)
    assert ctx.blocks
    assert ctx.token_budget == eng.config.context.token_budget


def test_assemble_context_after_the_corpus_is_gone(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nQuokka scheduling ring buffer policy.\n")
    eng = make_engine()
    eng.index([root])
    for p in root.iterdir():
        p.unlink()
    eng.index([root])
    ctx = eng.assemble_context("quokka")
    assert ctx.blocks == ()


def test_context_config_validation() -> None:
    from orbit.errors import ConfigurationError

    for kwargs in (
        {"token_budget": 0},
        {"max_blocks": 0},
        {"similarity_threshold": 2.0},
        {"min_score_ratio": -1.0},
        {"ordering": "nope"},
    ):
        with pytest.raises(ConfigurationError):
            ContextBudgetConfig(**kwargs)


def test_context_budget_config_reaches_the_engine() -> None:
    from orbit.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        OrbitConfig(db_path="x.db", context=ContextBudgetConfig(token_budget=-1))