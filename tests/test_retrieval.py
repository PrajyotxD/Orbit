"""Retrieval: ranking, fusion, filters, robustness, determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit import OrbitConfig, OrbitEngine
from orbit.config import ChunkingConfig, RetrievalConfig
from orbit.errors import RetrievalError
from orbit.retrieval import Retriever
from orbit.retrieval.fusion import fuse, min_max_normalize
from orbit.retrieval.lexical import build_match_query, search_lexical
from orbit.models import RetrievalMethod

from conftest import write


def _citations(result) -> list[str]:
    return [h.citation for h in result.hits]


# ------------------------------------------------------------ match query safety
@pytest.mark.parametrize(
    "raw",
    [
        '" OR 1=1 --',
        "NEAR(",
        'unbalanced "quote',
        "*",
        "AND OR NOT ()",
        "\\",
        "'",
        '""',
        "((((",
        'a"*b',
        "%",
        "_",
        "drop table documents;--",
        "\x00\x01",
        "Ω≈ç√∫˜µ≤≥÷",
        "こんにちは",
        "a" * 5000,
        "'; DELETE FROM chunks; --",
    ],
)
def test_match_query_builder_never_raises(raw: str) -> None:
    expr = build_match_query(raw, mode="or")
    assert isinstance(expr, str)
    if expr:
        assert expr.count('"') % 2 == 0


def test_build_match_query_rejects_bad_mode() -> None:
    with pytest.raises(RetrievalError):
        build_match_query("hello", mode="xor")


def test_build_match_query_drops_punctuation_only_input() -> None:
    assert build_match_query("!!! ??? ...", mode="and") == ""
    assert build_match_query("", mode="and") == ""


# ------------------------------------------------------------------- fixtures
@pytest.fixture
def ranked_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "quantum.md", "# Quantum\n\nQuokka scheduling uses a deterministic ring buffer.\n")
    write(root, "biology.md", "# Biology\n\nAxolotls regenerate limbs using blastema tissue.\n")
    write(root, "finance.md", "# Finance\n\nBond yields invert when the curve steepens sharply.\n")
    write(root, "misc.txt", "Miscellaneous notes about gardening and tomatoes.\n")
    return root


# ------------------------------------------------------------------- ranking
def test_exact_keyword_match_ranks_the_right_document_first(
    make_engine, ranked_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    result = eng.search("blastema", limit=4)
    assert result.hits
    assert result.hits[0].evidence.source_path == "biology.md"
    assert "blastema" in result.hits[0].text
    assert result.hits[0].citation.startswith("biology.md:")


def test_every_hit_carries_evidence_provenance(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    for hit in eng.search("turbines OR ring OR yields OR gardening", limit=10):
        ev = hit.evidence
        assert ev.chunk_id and ev.doc_id
        assert ev.source_path
        assert ev.start_line >= 1 and ev.end_line >= ev.start_line
        assert ev.rank >= 1
        assert isinstance(ev.score, float)
        assert ev.retrieval_method in tuple(RetrievalMethod)
        assert hit.citation == f"{ev.source_path}:{ev.line_range}"


# ------------------------------------------------------------------- channels
def test_lexical_only_mode_works_with_no_embeddings(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine(embedding_provider="none")
    report = eng.index([ranked_corpus])
    assert report.added == 4
    assert report.embeddings_written == 0
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == 0

    result = eng.search("blastema", limit=5)
    assert result.hits
    assert result.hits[0].evidence.source_path == "biology.md"
    assert result.hits[0].evidence.retrieval_method is RetrievalMethod.LEXICAL
    assert all(h.evidence.vector_score is None for h in result.hits)


def test_fusion_lexical_only_config_disables_the_vector_channel(
    make_engine, ranked_corpus: Path
) -> None:
    eng = OrbitEngine(
        OrbitConfig(
            db_path=eng_db_path(make_engine),
            retrieval=RetrievalConfig(fusion="lexical_only"),
        )
    )
    eng.index([ranked_corpus])
    result = eng.search("blastema")
    assert result.retrieval_method is RetrievalMethod.LEXICAL
    assert all(h.evidence.vector_score is None for h in result.hits)
    eng.close()


def eng_db_path(make_engine) -> str:
    return make_engine().config.db_path


def test_vector_only_mode(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    assert eng.db.scalar("SELECT count(*) FROM embeddings") > 0

    result = eng.search("blastema", modes=("vector",))
    assert result.retrieval_method is RetrievalMethod.VECTOR
    assert result.channels_used == ("vector",)
    assert result.hits
    assert result.hits[0].evidence.source_path == "biology.md"
    assert result.hits[0].evidence.vector_score is not None
    assert result.hits[0].evidence.lexical_score is None


def test_hybrid_mode_populates_both_channel_scores(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    result = eng.search("blastema", limit=10)
    assert result.retrieval_method is RetrievalMethod.HYBRID
    assert set(result.channels_used) == {"lexical", "vector"}
    both = [
        h
        for h in result.hits
        if h.evidence.lexical_score is not None and h.evidence.vector_score is not None
    ]
    assert both, "at least one hit must come from both channels with both scores populated"
    assert both[0].evidence.source_path == "biology.md"


def test_vector_channel_does_not_return_zero_similarity_chunks(
    make_engine, ranked_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    result = eng.search("blastema", limit=10)
    unrelated = {h.evidence.chunk_id for h in result.hits if h.evidence.source_path == "finance.md"}
    assert not unrelated, "an unrelated document was returned for an unrelated query"


# --------------------------------------------------------------------- fusion
def test_rrf_fusion_formula() -> None:
    lex = [("a", 5.0), ("b", 4.0), ("c", 3.0)]
    vec = [("c", 0.9), ("a", 0.5)]
    out = fuse(lex, vec, method="rrf", rrf_k=60)
    by_id = {h.chunk_id: h for h in out}
    expected_a = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    expected_c = 1.0 / (60 + 3) + 1.0 / (60 + 1)
    assert by_id["a"].fused_score == pytest.approx(expected_a)
    assert by_id["c"].fused_score == pytest.approx(expected_c)
    assert by_id["b"].fused_score == pytest.approx(1.0 / (60 + 2))
    assert out[0].chunk_id == "a"


def test_rrf_k_is_respected() -> None:
    lex = [("a", 5.0), ("b", 4.0)]
    vec = [("a", 0.5), ("b", 0.9)]
    k1 = {h.chunk_id: h.fused_score for h in fuse(lex, vec, method="rrf", rrf_k=1)}
    k60 = {h.chunk_id: h.fused_score for h in fuse(lex, vec, method="rrf", rrf_k=60)}
    k1000 = {h.chunk_id: h.fused_score for h in fuse(lex, vec, method="rrf", rrf_k=1000)}
    assert k1 != k60 != k1000
    assert k1["a"] == pytest.approx(1 / 2 + 1 / 2)
    assert k1000["a"] == pytest.approx(1 / 1001 + 1 / 1001)
    # damping: larger k compresses the gap between rank 1 and rank 2
    assert (k1["a"] - k1["b"]) > (k1000["a"] - k1000["b"])
    assert k60["a"] == pytest.approx(2 / 61)


def test_rrf_k_changes_scores_end_to_end(make_engine, ranked_corpus: Path) -> None:
    """The configured ``rrf_k`` must actually reach the fusion step."""
    db = make_engine().config.db_path
    scores = {}
    for k in (1, 1000):
        eng = OrbitEngine(
            OrbitConfig(db_path=db, retrieval=RetrievalConfig(fusion="rrf", rrf_k=k))
        )
        eng.index([ranked_corpus])
        result = eng.search("blastema", limit=10)
        assert result.retrieval_method is RetrievalMethod.HYBRID
        scores[k] = [h.evidence.score for h in result.hits]
        eng.close()
    assert scores[1] != scores[1000], "rrf_k had no effect on the fused scores"
    # a large k must compress the whole score range toward zero
    assert max(scores[1000]) < max(scores[1])


def test_weighted_fusion_normalises_channels() -> None:
    lex = [("a", 100.0), ("b", 1.0)]
    vec = [("b", 0.2), ("a", 0.1)]
    out = fuse(lex, vec, method="weighted", lexical_weight=1.0, vector_weight=1.0)
    by_id = {h.chunk_id: h for h in out}
    assert by_id["a"].fused_score == pytest.approx(1.0 + 0.0)
    assert by_id["b"].fused_score == pytest.approx(0.0 + 1.0)


def test_weighted_fusion_respects_weights() -> None:
    lex = [("a", 10.0)]
    vec = [("b", 10.0)]
    lex_heavy = {h.chunk_id: h.fused_score for h in fuse(lex, vec, method="weighted", lexical_weight=5.0)}
    vec_heavy = {h.chunk_id: h.fused_score for h in fuse(lex, vec, method="weighted", vector_weight=5.0)}
    assert lex_heavy["a"] > lex_heavy["b"]
    assert vec_heavy["b"] > vec_heavy["a"]


def test_fusion_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        fuse([("a", 1.0)], [], method="magic")


def test_min_max_normalize_edges() -> None:
    assert min_max_normalize([]) == []
    assert min_max_normalize([1, 2, 3]) == [0.0, 0.5, 1.0]
    assert min_max_normalize([7, 7, 7]) == [1.0, 1.0, 1.0]
    assert min_max_normalize([-5, 5]) == [0.0, 1.0]


@pytest.mark.parametrize("method", ["rrf", "weighted"])
def test_both_fusion_modes_work_end_to_end(
    make_engine, ranked_corpus: Path, method: str
) -> None:
    eng = OrbitEngine(
        OrbitConfig(db_path=make_engine().config.db_path, retrieval=RetrievalConfig(fusion=method))
    )
    eng.index([ranked_corpus])
    result = eng.search("blastema", limit=5)
    assert result.hits
    assert result.retrieval_method is RetrievalMethod.HYBRID
    eng.close()


# ----------------------------------------------------------------- duplicates
def test_results_never_repeat_a_chunk_id(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    body = "Duplicate payload about the flux capacitor.\n"
    for rel in ("a.txt", "b.txt", "c.txt", "d/e.txt"):
        write(root, rel, body)
    eng = make_engine()
    eng.index([root])
    for limit in (1, 3, 10, 50):
        ids = [h.evidence.chunk_id for h in eng.search("flux capacitor", limit=limit)]
        assert len(ids) == len(set(ids)), f"duplicate chunk_id at limit={limit}"
    cids = [h.evidence.chunk_id for h in eng.search("flux capacitor", limit=20)]
    assert len(cids) == len(set(cids))


# --------------------------------------------------------------------- filters
def test_language_filter(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    res = eng.search("blastema OR yields OR gardening", limit=20, language="markdown")
    assert res.hits
    assert {h.evidence.source_path for h in res.hits} <= {
        "quantum.md",
        "biology.md",
        "finance.md",
    }
    py = eng.search("anything", limit=20, language="python")
    assert py.hits == ()


def test_path_glob_filter(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    md = {h.evidence.source_path for h in eng.search("blastema", path_glob="*.md")}
    assert "biology.md" in md
    assert "misc.txt" not in md, "a .md glob must exclude the .txt document"
    assert eng.search("blastema", path_glob="*.rs").hits == ()
    assert eng.search("blastema", path_glob="*nomatch*").hits == ()
    txt = {h.evidence.source_path for h in eng.search("gardening", path_glob="*.txt")}
    assert txt == {"misc.txt"}


def test_tag_filter_that_nothing_matches(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    assert eng.search("blastema", tags=["does-not-exist"]).hits == ()


def test_filters_do_not_break_ranking_of_the_remainder(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    all_hits = eng.search("regenerate limbs", limit=10).hits
    md_hits = eng.search("regenerate limbs", limit=10, language="markdown").hits
    assert all_hits and md_hits
    assert md_hits[0].evidence.source_path == "biology.md"


# --------------------------------------------------------------- degenerate input
@pytest.mark.parametrize("query", ["", "   ", "\t\n", "\r\n", None])
def test_empty_or_whitespace_queries_return_nothing(
    make_engine, ranked_corpus: Path, query
) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    result = eng.search(query)
    assert result.hits == ()
    assert result.degraded_reason == "empty query"


def test_punctuation_only_query_returns_nothing(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    for query in ("...", "!!!", "***", "()"):
        assert eng.search(query).hits == (), f"{query!r} should match nothing"


@pytest.mark.parametrize(
    "query",
    [
        '" OR 1=1 --',
        "NEAR(",
        'unbalanced "quote',
        "*",
        "AND OR NOT ()",
        "\\",
        "'",
        '""',
        "((((",
        'a"*b',
        "%",
        "_",
        "'; DELETE FROM chunks; --",
        "Ω≈ç√∫˜µ≤≥÷",
        "こんにちは世界 😀",
        "a" * 4000,
    ],
)
def test_hostile_queries_never_raise_sqlite_errors(
    make_engine, ranked_corpus: Path, query: str
) -> None:
    import sqlite3

    eng = make_engine()
    eng.index([ranked_corpus])
    try:
        result = eng.search(query, limit=10)
    except sqlite3.Error as exc:  # pragma: no cover - this is the bug we hunt
        pytest.fail(f"sqlite error for query {query!r}: {exc}")
    assert isinstance(result.hits, tuple)
    # the index must be untouched by hostile input
    assert eng.db.scalar("SELECT count(*) FROM chunks") > 0
    assert eng.search("blastema").hits


# ---------------------------------------------------------------- determinism
def test_repeated_queries_are_identical(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    query = "regenerate limbs blastema"
    a = eng.search(query, limit=10)
    b = eng.search(query, limit=10)
    assert [h.evidence.chunk_id for h in a.hits] == [h.evidence.chunk_id for h in b.hits]
    assert _citations(a) == _citations(b)
    assert [h.evidence.rank for h in a.hits] == [h.evidence.rank for h in b.hits]
    assert [h.evidence.score for h in a.hits] == [h.evidence.score for h in b.hits]


def test_ordering_is_stable_across_a_fresh_engine(make_engine, db_path: Path, ranked_corpus: Path) -> None:
    first = make_engine()
    first.index([ranked_corpus])
    citations = _citations(first.search("regenerate limbs", limit=10))
    first.close()

    second = OrbitEngine(OrbitConfig(db_path=str(db_path)))
    try:
        assert _citations(second.search("regenerate limbs", limit=10)) == citations
    finally:
        second.close()


# ---------------------------------------------------------------------- misc
def test_limit_is_respected(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    for limit in (1, 2, 3):
        assert len(eng.search("the OR a OR of", limit=limit)) <= limit


def test_search_on_an_empty_index_returns_nothing(make_engine, empty_corpus: Path) -> None:
    eng = make_engine()
    eng.index([empty_corpus])
    assert eng.search("anything at all").hits == ()


def test_ranks_are_dense_and_ascending(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    ranks = [h.evidence.rank for h in eng.search("the OR a OR of", limit=10)]
    assert ranks == list(range(1, len(ranks) + 1))


def test_scores_are_non_increasing(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    scores = [h.evidence.score for h in eng.search("the OR a OR of", limit=10)]
    assert scores == sorted(scores, reverse=True)


def test_search_lexical_directly(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    hits = search_lexical(eng.db, "blastema", limit=5)
    assert hits
    assert hits[0].score > 0  # bm25 negated: higher is better
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_chunks_have_a_single_embedding_each(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == eng.stats().chunks
    rows = eng.db.query("SELECT dim FROM embeddings")
    assert {r["dim"] for r in rows} == {512}


def test_custom_chunking_config_changes_chunk_counts(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "long.txt", "\n".join(f"paragraph {i} about turbines" for i in range(1, 400)) + "\n")

    def build(db_name: str, **chunk_kwargs) -> int:
        eng = OrbitEngine(
            OrbitConfig(db_path=str(tmp_path / db_name), chunking=ChunkingConfig(**chunk_kwargs))
        )
        eng.index([root])
        n = eng.stats().chunks
        eng.close()
        return n

    n_small = build("small.db", max_tokens=30, overlap_tokens=4)
    n_large = build("large.db", max_tokens=400, overlap_tokens=20)
    assert n_small > n_large, (n_small, n_large)


def test_changing_chunking_config_forces_a_rechunk(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "long.txt", "\n".join(f"paragraph {i} about turbines" for i in range(1, 400)) + "\n")
    db = str(tmp_path / "d.db")

    small = OrbitEngine(
        OrbitConfig(db_path=db, chunking=ChunkingConfig(max_tokens=30, overlap_tokens=4))
    )
    small.index([root])
    n_small = small.stats().chunks
    small.close()

    large = OrbitEngine(
        OrbitConfig(db_path=db, chunking=ChunkingConfig(max_tokens=400, overlap_tokens=20))
    )
    large.index([root])
    n_large = large.stats().chunks
    large.close()
    assert n_large < n_small


def test_retriever_is_constructible_directly(make_engine, ranked_corpus: Path) -> None:
    eng = make_engine()
    eng.index([ranked_corpus])
    r = Retriever(eng.db, eng.config)
    result = r.search("blastema", limit=3)
    assert result.hits[0].evidence.source_path == "biology.md"


def test_channels_used_only_reports_channels_that_actually_produced_results(
    make_engine, ranked_corpus: Path
) -> None:
    eng = make_engine(embedding_provider="none")
    eng.index([ranked_corpus])
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == 0
    result = eng.search("blastema")
    assert "vector" not in result.channels_used
    assert result.degraded_reason, "an inert vector channel must be reported as degraded"


def test_reindex_does_not_clobber_the_recorded_embedding_provider(
    tmp_path: Path, ranked_corpus: Path
) -> None:
    db = str(tmp_path / "provider.db")
    first = OrbitEngine(OrbitConfig(db_path=db))
    first.index([ranked_corpus])
    assert first.db.get_meta("embedding_provider") == "hashing"
    first.close()

    # a second engine/process re-runs index() on an unchanged corpus
    second = OrbitEngine(OrbitConfig(db_path=db))
    try:
        report = second.index([ranked_corpus])
        assert report.unchanged == 4
        assert second.db.get_meta("embedding_provider") == "hashing"
    finally:
        second.close()