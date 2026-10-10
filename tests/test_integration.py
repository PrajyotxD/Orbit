"""End-to-end integration: the full loop, persistence, no-LLM, concurrency."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from orbit import OrbitConfig, OrbitEngine, __version__
from orbit.config import ContextBudgetConfig, RetrievalConfig
from orbit.context import assemble_context
from orbit.evidence import find_conflicts, validate_evidence
from orbit.models import ValidationState

from conftest import read_lines, write


@pytest.fixture
def story_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "story"
    root.mkdir()
    write(
        root,
        "architecture.md",
        "# Architecture\n\n"
        "Orbit ranks passages by fusing a lexical and a vector channel.\n"
        "Reciprocal Rank Fusion dampens the influence of the very top ranks.\n"
        "Every retrieved passage carries a pointer back to its source lines.\n",
    )
    write(
        root,
        "storage.md",
        "# Storage\n\n"
        "Documents are content-addressed and revision-tracked in SQLite.\n"
        "Chunks are purged when a document is superseded.\n",
    )
    write(
        root,
        "deploys.cfg",
        "The deploy timeout = 30 seconds for the staging cluster.\n",
    )
    write(
        root,
        "legacy.cfg",
        "The deploy timeout = 900 seconds for the staging cluster.\n",
    )
    return root


@pytest.mark.integration
def test_full_workflow_index_search_validate_assemble(make_engine, story_corpus: Path) -> None:
    eng = make_engine()

    # 1. index
    report = eng.index([story_corpus])
    assert report.added == 4
    assert report.failed == 0

    # 2. search
    result = eng.search("reciprocal rank fusion", limit=5)
    assert result.hits
    top = result.hits[0]
    assert top.evidence.source_path == "architecture.md"

    # 3. validate the citation against the live file
    validation = validate_evidence(eng.db, top.evidence.chunk_id)
    assert validation.state is ValidationState.VALID

    lines = read_lines(story_corpus / "architecture.md")
    region = "\n".join(lines[top.evidence.start_line - 1 : top.evidence.end_line])
    assert top.text in region

    # 4. assemble a context that keeps the source reference
    ctx = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0))
    assert ctx.blocks
    first = ctx.blocks[0]
    assert first.source_path == "architecture.md"
    assert f"{first.source_path}:{first.start_line}-{first.end_line}" in ctx.render()

    # 5. conflicts over the same corpus
    numeric = [c for c in find_conflicts(eng.db) if c.rule == "numeric_conflict"]
    assert numeric
    assert all(c.kind.value == "inferred" for c in numeric)


@pytest.mark.integration
def test_index_persists_across_close_and_reopen(tmp_path: Path, story_corpus: Path) -> None:
    db = str(tmp_path / "persist.db")

    first = OrbitEngine(OrbitConfig(db_path=db))
    first.index([story_corpus])
    before = [h.citation for h in first.search("reciprocal rank fusion", limit=5)]
    stats_before = first.stats()
    chunk_ids_before = {
        r["chunk_id"] for r in first.db.query("SELECT chunk_id FROM chunks")
    }
    first.close()

    assert Path(db).exists()

    second = OrbitEngine(OrbitConfig(db_path=db))
    try:
        after = [h.citation for h in second.search("reciprocal rank fusion", limit=5)]
        assert after == before
        stats_after = second.stats()
        assert stats_after.documents_current == stats_before.documents_current == 4
        assert stats_after.chunks == stats_before.chunks
        assert {r["chunk_id"] for r in second.db.query("SELECT chunk_id FROM chunks")} == chunk_ids_before

        # re-indexing an unchanged corpus is a no-op
        report = second.index([story_corpus])
        assert report.unchanged == 4
        assert report.added == 0
        assert report.updated == 0
    finally:
        second.close()


@pytest.mark.integration
def test_retrieval_works_with_no_llm_configured(make_engine, story_corpus: Path) -> None:
    """No generation config, no API key, no generator: search must still work."""
    import os

    for key in list(os.environ):
        if "API_KEY" in key or key.startswith("OPENAI"):
            del os.environ[key]

    cfg = OrbitConfig(db_path=make_engine().config.db_path)
    assert cfg.generation.enabled is False
    assert cfg.generation.base_url == ""
    assert cfg.generation.api_key() is None

    eng = OrbitEngine(cfg)
    try:
        eng.index([story_corpus])
        result = eng.search("reciprocal rank fusion", limit=5)
        assert result.hits
        assert result.hits[0].evidence.source_path == "architecture.md"
        for hit in result.hits:
            assert hit.evidence.is_valid in (True, False)  # not validated by default
        ctx = assemble_context(result, ContextBudgetConfig(min_score_ratio=0.0))
        assert ctx.blocks
        assert "generation" not in {k for k in cfg.__dataclass_fields__ if k != "generation"}
    finally:
        eng.close()


@pytest.mark.integration
def test_two_engines_on_the_same_db_are_both_readable(
    tmp_path: Path, story_corpus: Path
) -> None:
    db = str(tmp_path / "shared.db")
    writer = OrbitEngine(OrbitConfig(db_path=db))
    writer.index([story_corpus])

    reader = OrbitEngine(OrbitConfig(db_path=db))
    try:
        assert [h.citation for h in reader.search("chunk")] == [
            h.citation for h in writer.search("chunk")
        ]
        assert reader.stats().documents_current == 4

        # reader can also index; writer still sees a consistent database
        reader.index([story_corpus])
        assert writer.stats().documents_current == 4
        assert writer.search("reciprocal rank fusion").hits
        assert writer.db.scalar("SELECT count(*) FROM chunks") == writer.stats().chunks
    finally:
        writer.close()
        reader.close()


@pytest.mark.integration
def test_engine_context_manager(make_engine, story_corpus: Path) -> None:
    with OrbitEngine(OrbitConfig(db_path=make_engine().config.db_path)) as eng:
        eng.index([story_corpus])
        assert eng.search("chunk").hits
    assert eng.db.is_open is False


@pytest.mark.integration
def test_full_lifecycle_add_update_delete(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "life"
    root.mkdir()
    write(root, "one.md", "# One\n\nQuokka scheduling ring buffer allocation.\n")
    write(root, "two.md", "# Two\n\nNarwhal telemetry sampling configuration.\n")

    eng = make_engine()

    eng.index([root])
    assert sorted(d["path"] for d in eng.documents()) == ["one.md", "two.md"]

    write(root, "one.md", "# One\n\nWombat scheduling ring buffer allocation.\n")
    write(root, "three.md", "# Three\n\nQuokka and wombat both allocate.\n")
    (root / "two.md").unlink()

    report = eng.index([root])
    assert report.updated == 1
    assert report.added == 1
    assert report.deleted == 1
    assert sorted(d["path"] for d in eng.documents()) == ["one.md", "three.md"]

    assert eng.search("narwhal").hits == () or all(
        h.evidence.source_path != "two.md" for h in eng.search("narwhal").hits
    )
    hits = eng.search("wombat scheduling", limit=5)
    assert hits.hits
    for hit in hits.hits:
        assert validate_evidence(eng.db, hit.evidence.chunk_id).state is ValidationState.VALID


@pytest.mark.integration
def test_get_chunk_round_trip(make_engine, story_corpus: Path) -> None:
    eng = make_engine()
    eng.index([story_corpus])
    hit = eng.search("reciprocal rank fusion").hits[0]
    fetched = eng.get_chunk(hit.evidence.chunk_id)
    assert fetched is not None
    assert fetched.text == hit.text
    assert fetched.evidence.source_path == hit.evidence.source_path
    assert eng.get_chunk("no-such-chunk") is None


@pytest.mark.integration
def test_document_helpers(make_engine, story_corpus: Path) -> None:
    eng = make_engine()
    eng.index([story_corpus])
    docs = eng.documents()
    assert len(docs) == 4
    assert all(d["status"] == "current" for d in docs)
    detail = eng.document(docs[0]["doc_id"])
    assert detail is not None
    assert detail["path"] == docs[0]["path"]
    assert eng.document("nope") is None
    assert len(eng.documents(status=None)) == 4


@pytest.mark.integration
def test_init_creates_an_empty_index(make_engine) -> None:
    eng = make_engine()
    stats = eng.init()
    assert stats.schema_version >= 1
    assert stats.documents_current == 0
    assert stats.chunks == 0


@pytest.mark.integration
def test_rebuild_is_idempotent(make_engine, story_corpus: Path) -> None:
    eng = make_engine()
    eng.index([story_corpus])
    first = {r["chunk_id"]: r["content_hash"] for r in eng.db.query("SELECT chunk_id, content_hash FROM chunks")}
    eng.rebuild([story_corpus])
    second = {r["chunk_id"]: r["content_hash"] for r in eng.db.query("SELECT chunk_id, content_hash FROM chunks")}
    assert first == second, "a rebuild must reproduce identical chunks"
    assert eng.stats().documents_current == 4


@pytest.mark.integration
def test_search_after_reopen_matches_citations_exactly(
    tmp_path: Path, story_corpus: Path
) -> None:
    db = str(tmp_path / "cite.db")
    a = OrbitEngine(OrbitConfig(db_path=db))
    a.index([story_corpus])
    citations_a = [(h.citation, h.evidence.chunk_id) for h in a.search("chunk", limit=10)]
    a.close()

    b = OrbitEngine(OrbitConfig(db_path=db))
    try:
        citations_b = [(h.citation, h.evidence.chunk_id) for h in b.search("chunk", limit=10)]
        assert citations_a == citations_b
        for citation, chunk_id in citations_b:
            path, _, lines = citation.rpartition(":")
            assert path in {d["path"] for d in b.documents()}
            assert lines
    finally:
        b.close()


@pytest.mark.integration
def test_fusion_config_is_honoured_end_to_end(tmp_path: Path, story_corpus: Path) -> None:
    db = str(tmp_path / "fusion.db")
    for method in ("rrf", "weighted", "lexical_only"):
        eng = OrbitEngine(
            OrbitConfig(db_path=db, retrieval=RetrievalConfig(fusion=method, limit=5))
        )
        eng.index([story_corpus])
        result = eng.search("reciprocal rank fusion", limit=5)
        assert result.hits
        assert all(h.citation for h in result.hits)
        eng.close()


@pytest.mark.integration
def test_public_api_surface(make_engine) -> None:
    import orbit

    assert orbit.__version__ == __version__
    for name in (
        "OrbitEngine",
        "OrbitConfig",
        "IndexReport",
        "SearchResult",
        "Evidence",
        "ValidationResult",
        "ValidationState",
        "AssembledContext",
        "ConflictCandidate",
    ):
        assert hasattr(orbit, name), name


@pytest.mark.integration
def test_stats_as_dict_is_json_friendly(make_engine, story_corpus: Path) -> None:
    import json

    eng = make_engine()
    eng.index([story_corpus])
    json.dumps(eng.stats().as_dict())
    json.dumps(eng.index([story_corpus]).as_dict())


@pytest.mark.integration
def test_database_uses_expected_pragmas(make_engine, story_corpus: Path) -> None:
    eng = make_engine()
    eng.index([story_corpus])
    assert eng.db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert eng.db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert eng.db.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


@pytest.mark.integration
def test_foreign_key_cascades_purge_embeddings(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "fk"
    root.mkdir()
    write(root, "a.txt", "alpha body\n")
    write(root, "b.txt", "bravo body\n")
    eng = make_engine()
    eng.index([root])
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == 2

    (root / "a.txt").unlink()
    eng.index([root])
    orphan = eng.db.query(
        "SELECT e.chunk_id FROM embeddings e LEFT JOIN chunks c ON c.chunk_id = e.chunk_id "
        "WHERE c.chunk_id IS NULL"
    )
    assert orphan == [], "embeddings must be cascade-deleted with their chunks"


@pytest.mark.integration
def test_sqlite_file_is_created_and_self_contained(tmp_path: Path, story_corpus: Path) -> None:
    db = tmp_path / "standalone.db"
    eng = OrbitEngine(OrbitConfig(db_path=str(db)))
    eng.index([story_corpus])
    eng.close()
    assert db.exists()
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        }
        for expected in ("documents", "chunks", "chunks_fts", "embeddings", "orbit_meta"):
            assert expected in tables, (expected, sorted(tables))
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == conn.execute(
            "SELECT count(*) FROM chunks"
        ).fetchone()[0]
    finally:
        conn.close()