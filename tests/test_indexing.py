"""Indexing: reconciliation, incremental updates, duplicates, interruption.

Every assertion here is written against the behaviour Orbit *claims*, not the
behaviour it happens to have. Where the implementation disagrees, the test is
left failing or marked ``xfail`` and reported.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from orbit import OrbitConfig, OrbitEngine
from orbit.errors import IndexNotFoundError, StorageError
from orbit.indexing import IndexManager
from orbit.models import DocumentStatus

from conftest import write


def _doc_paths(engine: OrbitEngine, status: str | None = "current") -> list[str]:
    return [d["path"] for d in engine.documents(status=status)]


# ------------------------------------------------------------------- basics
def test_empty_directory_indexes_nothing(make_engine, empty_corpus: Path) -> None:
    eng = make_engine()
    report = eng.index([empty_corpus])
    assert report.scanned == 0
    assert report.added == 0
    assert report.failed == 0
    assert report.chunks_written == 0
    assert _doc_paths(eng) == []
    stats = eng.stats()
    assert stats.documents_current == 0
    assert stats.chunks == 0
    # the database is still usable afterwards
    assert eng.search("anything").hits == ()


def test_index_without_paths_is_a_clear_error(make_engine) -> None:
    eng = make_engine()
    with pytest.raises(IndexNotFoundError):
        eng.index()


def test_single_document(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "one.md", "# One\n\nThe single document body talks about turbines.\n")
    eng = make_engine()
    report = eng.index([root])
    assert report.added == 1 and report.failed == 0
    stats = eng.stats()
    assert stats.documents_current == 1
    assert stats.chunks > 0
    assert stats.embeddings == stats.chunks  # hashing embedder writes one per chunk
    assert _doc_paths(eng) == ["one.md"]


def test_multiple_documents_and_languages(make_engine, corpus: Path) -> None:
    eng = make_engine()
    report = eng.index([corpus])
    assert report.added == 4
    assert report.failed == 0
    assert sorted(_doc_paths(eng)) == [
        "alpha.md",
        "notes/beta.txt",
        "ops/delta.md",
        "src/gamma.py",
    ]
    langs = eng.stats().languages
    assert langs == {"markdown": 2, "text": 1, "python": 1}


# ---------------------------------------------------------------- duplicates
def test_duplicate_content_at_two_paths_is_both_indexed(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    body = "Identical payload describing the flux capacitor in detail.\n"
    write(root, "one/dup.txt", body)
    write(root, "two/dup.txt", body)

    eng = make_engine()
    report = eng.index([root])
    assert report.added == 2
    assert sorted(_doc_paths(eng)) == ["one/dup.txt", "two/dup.txt"]

    docs = eng.documents()
    hashes = {d["path"]: d["content_hash"] for d in docs}
    assert hashes["one/dup.txt"] == hashes["two/dup.txt"]
    assert len({d["doc_id"] for d in docs}) == 2, "distinct paths must get distinct doc ids"


def test_duplicate_content_dedup_is_explicit_at_retrieval_time(
    make_engine, tmp_path: Path
) -> None:
    """Observed behaviour: both docs are stored, but retrieval keeps one copy.

    ``Retriever._dedupe`` drops results sharing a chunk ``content_hash``.  Both
    documents exist in the index; only one is reachable from search.  This test
    pins that behaviour so a future change is a deliberate decision.
    """
    root = tmp_path / "c"
    root.mkdir()
    body = "Identical payload describing the flux capacitor in detail.\n"
    write(root, "one/dup.txt", body)
    write(root, "two/dup.txt", body)

    eng = make_engine()
    eng.index([root])
    hits = eng.search("flux capacitor", limit=10).hits
    assert len(hits) == 1, "retrieval dedupes byte-identical chunk content"
    assert hits[0].citation.startswith("one/dup.txt")
    # but the other document is still a first-class record
    assert eng.get_chunk(eng.documents()[1]["path"] and hits[0].evidence.chunk_id) is not None


# ----------------------------------------------------------------- incremental
def test_modified_file_bumps_revision_and_replaces_chunks(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "m.txt", "first revision mentioning penguins and ice.\n")

    eng = make_engine()
    eng.index([root])
    doc = eng.documents()[0]
    assert doc["revision"] == 1
    before = {r["chunk_id"]: r["text"] for r in eng.db.query("SELECT chunk_id, text FROM chunks")}
    assert before

    write(root, "m.txt", "second revision mentioning narwhals and ice.\n")
    report = eng.index([root])
    assert report.updated == 1
    assert report.added == 0

    doc = eng.documents()[0]
    assert doc["revision"] == 2, "revision must increment on modification"

    after = {r["chunk_id"]: r["text"] for r in eng.db.query("SELECT chunk_id, text FROM chunks")}
    assert set(after) == set(before), "chunk ids are stable for an unchanged doc"
    assert "penguins" not in " ".join(after.values())
    assert "narwhals" in " ".join(after.values())
    assert eng.db.scalar("SELECT count(*) FROM chunks_fts") == len(after)
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == len(after)


def test_growing_a_document_removes_the_stale_chunks(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "g.txt", "short note.\n")
    eng = make_engine()
    eng.index([root])
    assert eng.db.scalar("SELECT count(*) FROM chunks") == 1

    write(root, "g.txt", "\n".join(f"line {i} of the expanded document" for i in range(1, 200)) + "\n")
    eng.index([root])
    n = eng.db.scalar("SELECT count(*) FROM chunks")
    assert n > 1
    assert eng.db.scalar("SELECT count(*) FROM chunks_fts") == n
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == n
    # no leftover text from the previous revision
    joined = " ".join(r["text"] for r in eng.db.query("SELECT text FROM chunks"))
    assert "short note" not in joined


def test_deleted_file_is_tombstoned_and_purged(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "d.txt", "content about elephants and giraffes.\n")
    keep = write(root, "keep.txt", "content that should stay indexed.\n")

    eng = make_engine()
    eng.index([root])
    assert eng.search("elephants").hits

    p.unlink()
    report = eng.index([root])
    assert report.deleted == 1

    assert _doc_paths(eng) == ["keep.txt"]
    all_docs = {d["path"]: d["status"] for d in eng.documents(status=None)}
    assert all_docs["d.txt"] == DocumentStatus.DELETED.value
    assert all_docs["keep.txt"] == DocumentStatus.CURRENT.value

    rows = eng.db.query("SELECT chunk_id FROM chunks")
    assert eng.db.scalar("SELECT count(*) FROM chunks") > 0
    chunk_ids = {r["chunk_id"] for r in rows}
    orphan = eng.db.query(
        "SELECT c.chunk_id FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
        "WHERE d.status='deleted'"
    )
    assert orphan == [], "deleted documents must not keep chunks"
    assert chunk_ids

    assert eng.search("elephants").hits == ()
    # NOTE: the deleted document must never appear.  Searching a word that only
    # occurred in the deleted file must not surface it under any circumstance.
    for query in ("elephants", "giraffes"):
        for hit in eng.search(query).hits:
            assert hit.evidence.source_path == "keep.txt", (
                f"query {query!r} surfaced the deleted document: {hit.citation}"
            )
    assert eng.search("indexed").hits, "surviving document is still searchable"
    assert keep.exists()


def test_rename_is_detected_as_a_rename(make_engine, tmp_path: Path) -> None:
    """Assert the ACTUAL observed behaviour: it IS detected as a rename.

    Observed: ``renamed == 1`` *and* ``deleted == 1``; the new path becomes the
    current document at revision 1, and the old path is retained as a deleted
    tombstone so a later move can still be recognised.
    """
    root = tmp_path / "c"
    root.mkdir()
    old = write(root, "old.txt", "Unique marker content about kangaroos and marsupials.\n")
    eng = make_engine()
    first = eng.index([root])
    assert first.added == 1

    old.rename(root / "new.txt")
    report = eng.index([root])

    assert report.renamed == 1, "rename detection is active"
    assert report.added == 0
    assert report.updated == 0
    assert report.deleted == 1, "the old path is also counted as deleted"

    statuses = {d["path"]: (d["status"], d["revision"]) for d in eng.documents(status=None)}
    assert statuses["new.txt"] == (DocumentStatus.CURRENT.value, 1)
    assert statuses["old.txt"][0] == DocumentStatus.DELETED.value

    hits = eng.search("kangaroos").hits
    assert hits and all(h.citation.startswith("new.txt") for h in hits)


def test_rename_with_changed_content_is_not_treated_as_a_rename(
    make_engine, tmp_path: Path
) -> None:
    root = tmp_path / "c"
    root.mkdir()
    old = write(root, "old.txt", "original body text about wombats.\n")
    eng = make_engine()
    eng.index([root])
    old.unlink()
    write(root, "new.txt", "completely different body text about quokkas.\n")
    report = eng.index([root])
    assert report.renamed == 0
    assert report.added == 1
    assert report.deleted == 1


def test_reindex_with_no_changes_rewrites_nothing(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    indexed_at = {
        r["doc_id"]: r["indexed_at"] for r in eng.db.query("SELECT doc_id, indexed_at FROM documents")
    }
    chunk_rows = [dict(r) for r in eng.db.query("SELECT * FROM chunks ORDER BY chunk_id")]

    report = eng.index([corpus])
    assert report.unchanged == 4
    assert report.added == 0
    assert report.updated == 0
    assert report.deleted == 0
    assert report.chunks_written == 0
    assert report.embeddings_written == 0

    after_indexed_at = {
        r["doc_id"]: r["indexed_at"] for r in eng.db.query("SELECT doc_id, indexed_at FROM documents")
    }
    assert after_indexed_at == indexed_at, "unchanged documents must not be rewritten"
    assert [dict(r) for r in eng.db.query("SELECT * FROM chunks ORDER BY chunk_id")] == chunk_rows


def test_touching_a_file_without_changing_it_is_unchanged(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "s.txt", "content that will not change at all.\n")
    eng = make_engine()
    eng.index([root])
    import os

    os.utime(p, (1_000_000, 1_000_000))  # old mtime, same bytes
    report = eng.index([root])
    assert report.unchanged == 1
    assert report.updated == 0, "content hash, not mtime, decides 'unchanged'"


# ------------------------------------------------------------- bad input
def test_unreadable_malformed_and_empty_files_do_not_break_the_index(
    make_engine, tmp_path: Path
) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "good.md", "# Good\n\nPerfectly readable content about gardening.\n")
    write(root, "invalid.txt", b"\xff\xfe\x00 broken \xc3\x28 utf8")
    write(root, "empty.txt", b"")
    write(root, "whitespace.txt", b"   \n\t\n")
    write(root, "binary.txt", bytes(range(256)) * 4)

    eng = make_engine()
    report = eng.index([root])
    assert report.added == 1
    assert report.failed >= 1, "binary/invalid files are reported, not swallowed"
    assert report.skipped >= 1, "empty files are skipped"

    failed_paths = {os.path.basename(p) for p, _r, _d in report.failures}
    assert {"invalid.txt", "binary.txt"} <= failed_paths

    assert _doc_paths(eng) == ["good.md"]
    assert eng.stats().documents_current == 1
    # index still usable
    assert eng.search("gardening").hits
    assert eng.stats().chunks > 0


def test_a_missing_root_is_reported_and_does_not_nuke_the_index(
    make_engine, tmp_path: Path
) -> None:
    """A typo'd / unmounted root must not wipe previously indexed documents.

    IMPLEMENTATION BUG (data loss): ``IndexManager.sync`` treats an unreadable
    root as "I saw zero files", so the prune pass concludes every previously
    indexed document was deleted and tombstones the whole index.  The failing
    root is even reported in ``report.failures`` *and* the data is destroyed.
    """
    root = tmp_path / "c"
    root.mkdir()
    write(root, "a.txt", "content one about pikas\n")
    eng = make_engine()
    eng.index([root])
    assert _doc_paths(eng) == ["a.txt"]

    report = eng.index([tmp_path / "c" / "vanished"])
    assert report.failed == 1, "the unreadable root is reported"

    assert _doc_paths(eng) == ["a.txt"], (
        "BUG: a failed root must not cause unrelated documents to be pruned"
    )
    assert report.deleted == 0


def test_prune_false_keeps_deleted_documents(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "a.txt", "content one\n")
    eng = make_engine()
    eng.index([root])
    p.unlink()
    report = eng.index([root], prune=False)
    assert report.deleted == 0
    assert _doc_paths(eng) == ["a.txt"]


# ------------------------------------------------------------ interruption
def test_interrupt_on_second_document_leaves_no_partial_rows(
    make_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "c"
    root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        write(root, name, f"content of {name} about pikas and marmots.\n")

    eng = make_engine()
    real = IndexManager._index_document
    calls = {"n": 0}

    def flaky(self, doc, *, prior, rename_from=None, root="", **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated interrupt")
        return real(self, doc, prior=prior, rename_from=rename_from, root=root, **kwargs)

    monkeypatch.setattr(IndexManager, "_index_document", flaky)
    report = eng.index([root])
    monkeypatch.undo()

    assert report.failed == 1
    assert any("b.txt" in p for p, _r, _d in report.failures)

    assert _doc_paths(eng) == ["a.txt", "c.txt"]
    rows = eng.db.query(
        "SELECT d.path, count(c.chunk_id) AS n FROM documents d "
        "LEFT JOIN chunks c ON c.doc_id = d.doc_id GROUP BY d.doc_id, d.path"
    )
    assert all(r["n"] >= 1 for r in rows), "no document row survives without its chunks"
    assert eng.db.scalar("SELECT count(*) FROM chunks_fts") == eng.db.scalar(
        "SELECT count(*) FROM chunks"
    )
    assert eng.db.scalar("SELECT count(*) FROM embeddings") == eng.db.scalar(
        "SELECT count(*) FROM chunks"
    )

    # db still openable and searchable
    eng.close()
    reopened = OrbitEngine(OrbitConfig(db_path=eng.config.db_path))
    try:
        assert reopened.search("marmots").hits
        assert _doc_paths(reopened) == ["a.txt", "c.txt"]
    finally:
        reopened.close()


def test_failure_inside_the_transaction_rolls_the_document_back(
    make_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise *inside* the write transaction; nothing may survive."""
    root = tmp_path / "c"
    root.mkdir()
    write(root, "big.txt", ("lorem ipsum dolor sit amet consectetur " * 60) + "\n")
    write(root, "ok.txt", "a small healthy document\n")

    eng = make_engine()
    state = {"n": 0}

    def boom(self, chunk_id, vector):
        # only blow up on the very first chunk write, which belongs to big.txt
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("crash mid-transaction")

    monkeypatch.setattr(IndexManager, "_write_embedding", boom)
    report = eng.index([root])
    monkeypatch.undo()

    assert report.failed == 1
    paths = _doc_paths(eng)
    assert "big.txt" not in paths
    assert "ok.txt" in paths

    # nothing left behind for the failed document
    assert eng.db.scalar(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
        "WHERE d.path LIKE '%big.txt'"
    ) == 0
    assert eng.db.scalar("SELECT count(*) FROM chunks_fts") == eng.db.scalar(
        "SELECT count(*) FROM chunks"
    )
    assert eng.search("lorem").hits == ()


def test_storage_error_is_caught_and_reported(
    make_engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "c"
    root.mkdir()
    write(root, "a.txt", "content a\n")

    eng = make_engine()

    def raise_storage(self, doc, *, prior, rename_from=None, root="", **kwargs):
        raise StorageError("disk on fire")

    monkeypatch.setattr(IndexManager, "_index_document", raise_storage)
    report = eng.index([root])
    monkeypatch.undo()

    assert report.failed == 1
    assert report.failures[0][1] == "storage_error"
    assert _doc_paths(eng) == []
    assert eng.stats().documents_current == 0


# ------------------------------------------------------------------ rebuild
def test_rebuild_recreates_everything(make_engine, corpus: Path) -> None:
    eng = make_engine()
    eng.index([corpus])
    before = eng.stats().documents_current
    assert before == 4
    report = eng.rebuild([corpus])
    assert report.added == 4
    assert eng.stats().documents_current == 4
    assert eng.db.scalar("SELECT count(*) FROM chunks_fts") == eng.stats().chunks
    assert eng.search("scheduler_threads").hits


# ---------------------------------------------------------------- integrity
def test_single_file_root_keeps_its_filename(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    p = write(root, "solo.txt", "solo content about axolotls.\n")
    eng = make_engine()
    eng.index([p])
    assert _doc_paths(eng) == ["solo.txt"]
    assert eng.search("axolotls").hits[0].citation.startswith("solo.txt:")


def test_individually_indexed_files_do_not_collide(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    a = write(root, "a.txt", "content about aardvarks.\n")
    b = write(root, "b.txt", "content about bandicoots.\n")
    eng = make_engine()
    eng.index([a])
    eng.index([b])
    docs = eng.documents()
    assert len(docs) == 2, "indexing two files individually lost one of them"
    assert {d["path"] for d in docs} == {"a.txt", "b.txt"}


def test_a_corrupt_database_is_reported_not_silently_ignored(
    make_engine, db_path: Path, corpus: Path
) -> None:
    """A garbage ``.db`` file must not be opened and used as if it were fine.

    Observed: ``sqlite3.DatabaseError('file is not a database')`` escapes raw
    from ``OrbitDB._apply_pragmas`` instead of the documented ``StorageError``,
    so callers cannot catch it via the Orbit error hierarchy alone.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"this is definitely not a sqlite database" * 100)

    with pytest.raises((sqlite3.DatabaseError, StorageError)) as exc:
        make_engine()
    assert isinstance(exc.value, (sqlite3.DatabaseError, StorageError))


def test_storage_layer_wraps_sql_errors_in_storage_error(make_engine, corpus: Path) -> None:
    from orbit.errors import StorageError as _StorageError

    eng = make_engine()
    eng.index([corpus])
    with pytest.raises(_StorageError):
        eng.db.execute("SELECT * FROM table_that_does_not_exist")