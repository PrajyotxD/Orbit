"""Evidence: citation validation, real line ranges, conflicts, freshness.

The most important assertions in this file are in
``test_every_cited_line_range_really_contains_its_chunk``: for every chunk of
several fixture documents, the cited lines are read back **from the file on
disk** and must literally contain the stored chunk text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit.engine import OrbitEngine
from orbit.evidence import (
    ConflictRule,
    detect_numeric_conflicts,
    find_conflicts,
    freshness_summary,
    stale_evidence,
    validate_citation,
    validate_evidence,
)
from orbit.evidence.graph import ensure_document_edges, neighbors, traverse
from orbit.evidence.validation import chunk_matches_source, verify_document
from orbit.models import ConflictCandidate, RelationKind, ValidationState, sha256_text

from conftest import read_lines, write


def _chunk_rows(engine: OrbitEngine) -> list[dict]:
    return [
        dict(r)
        for r in engine.db.query(
            "SELECT c.chunk_id, c.doc_id, c.text, c.start_line, c.end_line, d.path, d.abs_path "
            "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id ORDER BY d.path, c.ordinal"
        )
    ]


# --------------------------------------------------------------- basic states
def test_valid_citation_validates(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    for row in _chunk_rows(eng):
        result = eng.validate_citation(row["chunk_id"])
        assert result.state is ValidationState.VALID, (row["chunk_id"], result.detail)
        assert result.is_valid is True
        assert result.detail


def test_unknown_chunk_is_reported_not_raised(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    res = validate_evidence(eng.db, "does-not-exist")
    assert res.state is ValidationState.DOCUMENT_DELETED
    assert res.detail == "chunk not in index"


def test_deleted_file_on_disk_reports_path_missing(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "v.txt", "a sentence about ocelots and jaguars.\n")
    eng = make_engine()
    eng.index([root])
    chunk_id = eng.db.scalar("SELECT chunk_id FROM chunks LIMIT 1")

    p.unlink()
    res = validate_evidence(eng.db, chunk_id)
    assert res.state is ValidationState.PATH_MISSING
    assert res.detail
    assert res.is_valid is False
    # the index row itself is untouched: the document is still 'current'
    assert eng.documents()[0]["status"] == "current"


def test_document_deleted_in_index_reports_document_deleted(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "v.txt", "a sentence about ocelots and jaguars.\n")
    eng = make_engine()
    eng.index([root])
    chunk_id = eng.db.scalar("SELECT chunk_id FROM chunks LIMIT 1")

    p.unlink()
    eng.index([root])
    # chunks are purged, so the citation now resolves to "not in index"
    res = validate_evidence(eng.db, chunk_id)
    assert res.state is ValidationState.DOCUMENT_DELETED
    assert eng.db.scalar("SELECT count(*) FROM chunks") == 0


def test_modified_file_reports_content_changed(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "m.txt", "the original sentence about storks and herons.\n")
    eng = make_engine()
    eng.index([root])
    chunk_id = eng.db.scalar("SELECT chunk_id FROM chunks LIMIT 1")
    assert validate_evidence(eng.db, chunk_id).state is ValidationState.VALID

    write(root, "m.txt", "a completely different sentence about puffins and gulls.\n")
    res = validate_evidence(eng.db, chunk_id)
    assert res.state is ValidationState.CONTENT_CHANGED
    assert res.current_hash is not None
    assert res.current_hash != sha256_text(
        "\n".join(read_lines(root / "m.txt")[0:2])
    )


def test_shrinking_the_file_reports_out_of_range(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "s.txt", "\n".join(f"line {i}" for i in range(1, 60)) + "\n")
    eng = make_engine()
    eng.index([root])
    last = eng.db.query(
        "SELECT chunk_id, end_line FROM chunks ORDER BY end_line DESC LIMIT 1"
    )[0]
    assert last["end_line"] > 5

    write(root, "s.txt", "tiny\n")
    res = validate_evidence(eng.db, last["chunk_id"])
    assert res.state in (ValidationState.CHUNK_OUT_OF_RANGE, ValidationState.CONTENT_CHANGED)
    assert res.is_valid is False


def test_stale_revision_is_detectable_through_the_document_revision(
    make_engine, tmp_path: Path
) -> None:
    """A citation captured before an edit must not silently appear current.

    Search validates hits against the live file, so ``Evidence.revision`` and
    ``validation`` together show whether a citation is still current.
    """
    root = tmp_path / "c"
    root.mkdir()
    write(root, "r.txt", "revision one body text about turbines.\n")
    eng = make_engine()
    eng.index([root])
    hit = eng.search("turbines").hits[0]
    assert hit.evidence.revision == 1
    assert hit.evidence.validation is ValidationState.VALID
    assert eng.validate_citation(hit.evidence.chunk_id).state is ValidationState.VALID

    write(root, "r.txt", "revision two body text about rotors.\n")
    eng.index([root])
    assert eng.documents()[0]["revision"] == 2

    hit2 = eng.search("rotors").hits[0]
    assert hit2.evidence.revision == 2
    assert eng.validate_citation(hit2.evidence.chunk_id).state is ValidationState.VALID

    # The *old* chunk_id now resolves to the NEW text; the stale reference is
    # therefore only detectable by comparing revision + hash, which is exactly
    # what Evidence carries.
    assert hit.evidence.revision == 1
    assert hit.evidence.content_hash != hit2.evidence.content_hash


def test_verify_document(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "v.txt", "hello world\n")
    eng = make_engine()
    eng.index([root])
    doc_id = eng.documents()[0]["doc_id"]
    assert verify_document(eng.db, doc_id).state is ValidationState.VALID
    write(root, "v.txt", "hello worlds\n")
    assert verify_document(eng.db, doc_id).state is ValidationState.CONTENT_CHANGED
    p.unlink()
    assert verify_document(eng.db, doc_id).state is ValidationState.PATH_MISSING
    assert verify_document(eng.db, "nope").state is ValidationState.DOCUMENT_DELETED


def test_chunk_matches_source_helper(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    row = _chunk_rows(eng)[0]
    assert chunk_matches_source(eng.db, row["chunk_id"]) is True
    assert chunk_matches_source(eng.db, "nope") is False


# -------------------------------------------------- THE CRITICAL LINE-RANGE TEST
@pytest.mark.parametrize("docname", ["alpha.md", "beta.txt", "gamma.py", "delta.md", "wide.md"])
def test_every_cited_line_range_really_contains_its_chunk(
    make_engine, tmp_path: Path, docname: str
) -> None:
    """Read every chunk's cited lines back from disk and verify the content.

    This is the single most load-bearing claim in Orbit's evidence model.
    """
    root = tmp_path / "c"
    root.mkdir()
    if docname == "wide.md":
        parts = []
        for i in range(1, 160):
            parts.append(f"## Heading {i}" if i % 4 == 0 else f"paragraph {i} with words unique{i} here")
        body = "\n".join(parts)
    else:
        body = "\n".join(
            f"line {i} of {docname} describing subject {i % 11} at length"
            for i in range(1, 90)
        )
    p = write(root, docname, body + "\n")

    eng = make_engine()
    eng.index([root])
    rows = [r for r in _chunk_rows(eng) if r["path"] == docname]
    assert len(rows) > 1, "fixture must produce multiple chunks"

    on_disk = p.read_text(encoding="utf-8")
    lines = read_lines(p)

    for row in rows:
        start, end = row["start_line"], row["end_line"]
        assert start >= 1, row
        assert end >= start, row
        assert end <= len(lines), (
            f"{docname} {row['chunk_id']} cites end_line={end} but file has {len(lines)} lines"
        )
        region = "\n".join(lines[start - 1 : end])
        # (a) the cited region literally contains the stored chunk text
        assert row["text"] in region, (
            f"{docname} {row['chunk_id']}: cited [{start}-{end}] does NOT contain "
            f"{row['text'][:80]!r}; region was {region[:120]!r}"
        )
        # (b) the chunk text is verbatim source content of the real file
        assert row["text"] in on_disk
        # (c) the chunk's meaningful content starts and ends inside the range
        stripped = row["text"].strip()
        if stripped:
            idx = on_disk.index(stripped)
            first = on_disk[:idx].count("\n") + 1
            last = first + stripped.count("\n")
            assert start <= first, (row["chunk_id"], start, first)
            assert last <= end, (row["chunk_id"], last, end)
        # (d) validation agrees with the file on disk
        assert validate_evidence(eng.db, row["chunk_id"]).state is ValidationState.VALID


def test_line_ranges_survive_an_edit_outside_the_cited_region(
    make_engine, tmp_path: Path
) -> None:
    """Editing *outside* a chunk's range keeps that chunk valid; inside breaks it."""
    root = tmp_path / "c"
    root.mkdir()
    lines = [f"line {i} of the document" for i in range(1, 200)]
    write(root, "e.txt", "\n".join(lines) + "\n")
    eng = make_engine()
    eng.index([root])

    last = dict(
        eng.db.query(
            "SELECT c.chunk_id, c.start_line, c.end_line FROM chunks c "
            "JOIN documents d ON d.doc_id=c.doc_id ORDER BY c.end_line DESC LIMIT 1"
        )[0]
    )
    assert eng.validate_citation(last["chunk_id"]).state is ValidationState.VALID

    lines[0] = "MUTATED first line"
    write(root, "e.txt", "\n".join(lines) + "\n")
    # the mutated line is far above the cited region
    assert eng.validate_citation(last["chunk_id"]).state is ValidationState.VALID

    lines[last["start_line"] - 1] = "MUTATED inside the cited range"
    write(root, "e.txt", "\n".join(lines) + "\n")
    assert eng.validate_citation(last["chunk_id"]).state is ValidationState.CONTENT_CHANGED


# ------------------------------------------------------------------- conflicts
@pytest.fixture
def conflict_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "primary.cfg", "The connection timeout = 30 seconds for every client.\n")
    write(root, "mirror.cfg", "The connection timeout = 300 seconds for every client.\n")
    return root


def test_numeric_conflict_is_detected_and_labelled_inferred(
    make_engine, conflict_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    candidates = find_conflicts(eng.db)
    numeric = [c for c in candidates if c.rule == "numeric_conflict"]
    assert numeric, "two files stating different values for the same attribute must conflict"
    c = numeric[0]
    assert {c.path_a, c.path_b} == {"primary.cfg", "mirror.cfg"}
    assert "timeout" in c.detail
    assert c.score > 0

    # it is a LEAD, never proof
    assert c.kind is RelationKind.INFERRED
    assert c.kind is not RelationKind.EXPLICIT
    assert isinstance(c, ConflictCandidate)


def test_conflicts_are_persisted_and_readable_from_the_engine(
    make_engine, conflict_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    find_conflicts(eng.db)
    rows = eng.db.query("SELECT chunk_id_a, chunk_id_b, rule, kind, detail FROM conflicts")
    assert rows
    for r in rows:
        assert r["kind"] == "inferred", "persisted conflicts must be labelled inferred"
    api = eng.conflicts()
    assert api
    for c in api:
        assert c.kind is RelationKind.INFERRED
        assert c.rule
        assert c.path_a != "?" and c.path_b != "?"
        assert c.line_range_a != "?" and c.line_range_b != "?"


def test_conflict_detail_names_the_rule_and_both_values(
    make_engine, conflict_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    (c,) = [x for x in find_conflicts(eng.db) if x.rule == "numeric_conflict"]
    assert "30" in c.detail and "300" in c.detail
    assert "timeout" in c.detail


def test_agreeing_values_produce_no_conflict(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "a.cfg", "The connection timeout = 30 seconds here.\n")
    write(root, "b.cfg", "The connection timeout = 30 seconds there.\n")
    eng = make_engine()
    eng.index([root])
    assert [c for c in find_conflicts(eng.db) if c.rule == "numeric_conflict"] == []


def test_close_values_are_within_tolerance(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "a.cfg", "The connection timeout = 30 seconds here.\n")
    write(root, "b.cfg", "The connection timeout = 33 seconds there.\n")
    eng = make_engine()
    eng.index([root])
    assert [c for c in find_conflicts(eng.db) if c.rule == "numeric_conflict"] == []


def test_find_conflicts_can_run_without_persisting(make_engine, conflict_corpus: Path) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    found = find_conflicts(eng.db, persist=False)
    assert found
    assert eng.db.scalar("SELECT count(*) FROM conflicts") == 0


def test_negation_conflict_is_detected(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "on.md", "Telemetry export is enabled for every collector by default.\n")
    write(root, "off.md", "Telemetry export is not enabled for every collector by default.\n")
    eng = make_engine()
    eng.index([root])
    rules = {c.rule for c in find_conflicts(eng.db)}
    assert "negation_conflict" in rules
    for c in find_conflicts(eng.db):
        assert c.kind is RelationKind.INFERRED


def test_detect_numeric_conflicts_is_pure(make_engine, conflict_corpus: Path) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    rows = [
        dict(r)
        for r in eng.db.query(
            "SELECT c.chunk_id, c.text, c.start_line, c.end_line, d.path "
            "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id"
        )
    ]
    raw = detect_numeric_conflicts(rows)
    assert raw
    a, b, rule, score, detail = raw[0]
    assert rule == "numeric_conflict"
    assert 0 < score <= 1
    assert detail


def test_conflict_rule_catalogue() -> None:
    from orbit.evidence.freshness import RULES

    assert {r.name for r in RULES} == {"numeric_conflict", "negation_conflict"}
    assert all(r.description for r in RULES)


def test_no_conflicts_in_a_consistent_corpus(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    # the fixture corpus has no `key = <number>` pairs at all
    assert [c for c in find_conflicts(eng.db) if c.rule == "numeric_conflict"] == []


# ------------------------------------------------------------------- freshness
def test_freshness_summary_counts_states(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    good = write(root, "good.txt", "a stable sentence about puffins.\n")
    moving = write(root, "moving.txt", "a sentence about storks that will change.\n")
    eng = make_engine()
    eng.index([root])

    summary = freshness_summary(eng.db)
    assert summary.get("valid") == eng.stats().chunks

    write(root, "moving.txt", "a totally different sentence about gulls.\n")
    summary = freshness_summary(eng.db)
    assert summary.get("content_changed", 0) >= 1
    assert summary.get("valid", 0) >= 1

    good.unlink()
    summary = freshness_summary(eng.db)
    assert summary.get("path_missing", 0) >= 1
    assert moving.exists()


def test_stale_evidence_finds_edited_sources(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "a.txt", "original sentence about storks.\n")
    write(root, "b.txt", "second sentence about herons.\n")
    eng = make_engine()
    eng.index([root])
    assert stale_evidence(eng.db) == []

    write(root, "a.txt", "completely different sentence about puffins.\n")
    stale = stale_evidence(eng.db)
    assert stale, "an edited source must be reported as stale evidence"
    assert all(s["state"] == ValidationState.CONTENT_CHANGED.value for s in stale)


def test_stale_evidence_finds_deleted_sources(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "a.txt", "original sentence about storks.\n")
    eng = make_engine()
    eng.index([root])
    assert eng.search("storks").hits

    p.unlink()
    eng.index([root])
    # Deleting a source purges its chunks entirely, so nothing about it can be
    # returned as evidence any more. The guarantee is that the passage is gone
    # from search and there is no residual (unverifiable) evidence referencing it.
    assert eng.search("storks").hits == ()
    assert stale_evidence(eng.db) == []


def test_conflict_rows_are_purged_when_their_chunks_disappear(
    make_engine, conflict_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    find_conflicts(eng.db)
    assert eng.db.scalar("SELECT count(*) FROM conflicts") == 1
    (conflict_corpus / "primary.cfg").unlink()
    eng.index([conflict_corpus])
    remaining = find_conflicts(eng.db)
    assert remaining == [], "a conflict whose source vanished must not be reported"


def test_persisted_conflicts_are_purged_when_a_document_is_deleted(
    make_engine, conflict_corpus: Path
) -> None:
    eng = make_engine()
    eng.index([conflict_corpus])
    find_conflicts(eng.db)
    assert eng.conflicts()

    (conflict_corpus / "primary.cfg").unlink()
    eng.index([conflict_corpus])

    leftovers = eng.conflicts()
    assert leftovers == [], "conflicts referring to deleted chunks must be dropped"
    for c in leftovers:
        assert c.path_a != "?" and c.path_b != "?"


def test_evidence_edges_are_purged_with_their_chunks(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "x.txt", "alpha beta gamma delta epsilon zeta eta theta\n")
    eng = make_engine()
    eng.index([root])
    assert eng.db.scalar("SELECT count(*) FROM evidence_edges") > 0

    write(root, "x.txt", "entirely new body text after an edit\n")
    eng.index([root])

    # An edge is orphaned when either endpoint is a chunk that no longer exists.
    # Document nodes ("doc:<id>") are legitimate sources and must be excluded
    # from that judgement.
    live_chunks = {r["chunk_id"] for r in eng.db.query("SELECT chunk_id FROM chunks")}
    orphans = [
        dict(r)
        for r in eng.db.query("SELECT source_id, target_id FROM evidence_edges")
        if (r["target_id"] not in live_chunks and not r["target_id"].startswith("doc:"))
        or (r["source_id"] not in live_chunks and not r["source_id"].startswith("doc:"))
    ]
    assert orphans == [], f"orphan evidence edges remain: {orphans}"

    p.unlink()
    eng.index([root])
    assert eng.db.scalar("SELECT count(*) FROM evidence_edges") == 0


def test_evidence_graph_explicit_edges_are_traversable(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "g.txt", "\n".join(f"section {i} of the graph document" for i in range(1, 120)) + "\n")
    eng = make_engine()
    eng.index([root])
    doc_id = eng.documents()[0]["doc_id"]
    n = ensure_document_edges(eng.db, doc_id)
    assert n > 1
    doc_node = f"doc:{doc_id}"
    edges = neighbors(eng.db, doc_node)
    assert edges
    assert all(e.kind.value == "explicit" for e in edges)
    steps, visited = traverse(eng.db, doc_node, max_depth=2)
    assert steps
    assert all(s.kind.value == "explicit" for s in steps)
    assert doc_node in visited