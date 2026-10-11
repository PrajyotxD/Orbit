"""Tests for the Orbit TUI service layer (orbit.tui.service).

These exercise the data layer that the terminal UI renders. They need no
terminal: every method opens a fresh, thread-local engine over the real engine
and returns plain snapshots. The whole module is skipped if textual is not
installed only where a test constructs the app; the service itself has no
textual dependency, so these run unconditionally.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from orbit.tui.service import (
    CATEGORIES,
    OrbitService,
    category_for_language,
)

from conftest import make_config, write


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def service(db_path: Path, corpus: Path) -> OrbitService:
    """A service over an index built from the default corpus."""
    from orbit.engine import OrbitEngine

    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([corpus])
    eng.close()
    return OrbitService(cfg)


@pytest.fixture
def empty_service(db_path: Path) -> OrbitService:
    """A service whose database was never initialised."""
    return OrbitService(make_config(db_path))


# --------------------------------------------------------------------------
# category mapping
# --------------------------------------------------------------------------
def test_category_for_language_buckets() -> None:
    assert category_for_language("python") == "code"
    assert category_for_language("java") == "code"
    assert category_for_language("markdown") == "docs"
    assert category_for_language("json") == "config"
    assert category_for_language("yaml") == "config"
    assert category_for_language("") == "other"
    assert category_for_language("brainfuck") == "other"


def test_categories_include_the_reference_filters() -> None:
    names = [c[0] for c in CATEGORIES]
    # "comments" is gone: Orbit indexes by language, not comment spans, so that
    # tab would always come back empty.
    assert names == ["all", "code", "docs", "config"]
    assert "comments" not in names


# --------------------------------------------------------------------------
# icon sets
# --------------------------------------------------------------------------
def test_icon_sets_all_define_every_name() -> None:
    from orbit.tui.icons import ICON_NAMES, ICON_SETS

    assert set(ICON_SETS) == {"unicode", "nerd", "ascii"}
    for set_name, icon_set in ICON_SETS.items():
        for name in ICON_NAMES:
            glyph = getattr(icon_set, name, "")
            assert glyph, f"{set_name} icon set is missing '{name}'"


def test_unicode_icon_set_is_single_width() -> None:
    from orbit.tui.icons import ICON_SETS, verify_single_width

    offenders = verify_single_width(ICON_SETS["unicode"])
    assert offenders == [], f"unicode glyphs wider than one cell: {offenders}"


def test_ascii_icon_set_is_pure_ascii() -> None:
    from orbit.tui.icons import ICON_NAMES, ICON_SETS

    ascii_set = ICON_SETS["ascii"]
    for name in ICON_NAMES:
        glyph = getattr(ascii_set, name)
        assert glyph.isascii(), f"ascii icon '{name}' = {glyph!r} is not ASCII"


def test_icon_set_precedence_flag_over_env_over_default() -> None:
    from orbit.tui.icons import DEFAULT_ICON_SET, resolve_icon_set_name

    assert DEFAULT_ICON_SET == "unicode"
    # flag wins over env
    assert resolve_icon_set_name("nerd", "ascii") == "nerd"
    # env used when no flag
    assert resolve_icon_set_name(None, "ascii") == "ascii"
    # default when neither is set
    assert resolve_icon_set_name(None, None) == "unicode"
    # an unknown value is ignored and falls through
    assert resolve_icon_set_name("bogus", "ascii") == "ascii"
    assert resolve_icon_set_name("bogus", "bogus") == "unicode"


def test_get_icon_set_falls_back_on_unknown() -> None:
    from orbit.tui.icons import get_icon_set

    assert get_icon_set("nerd").name == "nerd"
    assert get_icon_set(None).name == "unicode"
    assert get_icon_set("bogus").name == "unicode"


# --------------------------------------------------------------------------
# top bar / status on an uninitialised index
# --------------------------------------------------------------------------
def test_top_info_uninitialized(empty_service: OrbitService) -> None:
    info = empty_service.top_info()
    assert info.initialized is False
    assert info.file_count == 0
    assert info.last_indexed == "never"
    assert info.db_name.endswith(".db")


def test_status_uninitialized(empty_service: OrbitService) -> None:
    status = empty_service.status()
    assert status.initialized is False
    assert status.documents_current == 0
    assert status.chunks == 0


def test_search_uninitialized_reports_not_initialized(empty_service: OrbitService) -> None:
    snap = empty_service.search("anything", limit=5)
    assert snap.hits == ()
    assert "not initialized" in snap.degraded_reason
    assert snap.error == ""


# --------------------------------------------------------------------------
# top bar / status on a populated index
# --------------------------------------------------------------------------
def test_top_info_populated(service: OrbitService) -> None:
    info = service.top_info()
    assert info.initialized is True
    assert info.file_count == 4  # DEFAULT_CORPUS has four docs
    assert info.embedding_provider == "hashing"
    assert info.last_indexed != "never"


def test_status_populated_counts_match_engine(service: OrbitService) -> None:
    status = service.status(with_freshness=True)
    assert status.initialized is True
    assert status.documents_current == 4
    assert status.chunks > 0
    assert status.embeddings == status.chunks  # hashing embedder writes one per chunk
    # Every freshly indexed chunk validates against disk.
    assert status.freshness.get("valid", 0) == status.chunks


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def test_search_returns_ranked_hits_with_scores(service: OrbitService) -> None:
    snap = service.search("quaternion scheduling", limit=5)
    assert snap.error == ""
    assert snap.hits, "expected at least one hit for a corpus term"
    # No relevance percentage any more — we show the fused score instead.
    assert not hasattr(snap.hits[0], "relevance_pct")
    # Hits come back ranked, so the scores shouldn't increase down the list.
    scores = [h.score for h in snap.hits]
    assert scores == sorted(scores, reverse=True)
    for h in snap.hits:
        assert h.chunk_id
        assert h.path
        assert isinstance(h.score, float)
        # lexical and vector are shown separately; either can be None.
        assert h.lexical_score is None or isinstance(h.lexical_score, float)
        assert h.vector_score is None or isinstance(h.vector_score, float)
        assert h.line_range
        assert h.category in {"code", "docs", "config", "other"}


def test_search_hits_carry_real_citations(service: OrbitService) -> None:
    snap = service.search("quaternion scheduling", limit=5)
    top = snap.hits[0]
    # The path and lines it reports should check out against the live source.
    v = service.validate(top.chunk_id)
    assert v.found is True
    assert v.state == "valid"
    assert v.is_valid is True


def test_search_empty_query_is_handled(service: OrbitService) -> None:
    snap = service.search("   ", limit=5)
    # Nothing to search for: no hits, but not an error either.
    assert snap.hits == ()
    assert snap.error == ""


def test_search_no_match_reports_no_hits(service: OrbitService) -> None:
    snap = service.search("zzz_nonexistent_term_qwxyz", limit=5)
    assert snap.hits == ()
    assert snap.error == ""


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------
def test_evidence_resolves_related_passages(service: OrbitService) -> None:
    snap = service.search("quaternion scheduling", limit=5)
    chunk_id = snap.hits[0].chunk_id
    ev = service.evidence(chunk_id, max_depth=2)
    assert ev.found is True
    assert ev.error == ""
    # Related entries (if any) must resolve to a real path + line range.
    for rel in ev.related:
        assert rel.path and rel.path != "?"
        assert rel.line_range and rel.line_range != "?"
        assert rel.kind in {"explicit", "inferred"}


def test_evidence_blank_chunk_id(service: OrbitService) -> None:
    ev = service.evidence("", max_depth=1)
    assert ev.found is False
    assert ev.error


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def test_validate_detects_content_change(service: OrbitService, corpus: Path) -> None:
    snap = service.search("quaternion scheduling", limit=5)
    # Find a hit in alpha.md so we can edit its source out from under the index.
    target = next((h for h in snap.hits if h.filename == "alpha.md"), snap.hits[0])
    before = service.validate(target.chunk_id)
    assert before.state == "valid"

    # Prepend lines so the cited line range now covers *different* text. Editing
    # the region's content (not just appending past it) is what Orbit's
    # region-level hash check is designed to catch.
    src = corpus / target.path
    src.write_text(
        "PREPENDED HEADER LINE ONE\nPREPENDED HEADER LINE TWO\n"
        + src.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    after = service.validate(target.chunk_id)
    assert after.found is True
    assert after.state in {"content_changed", "chunk_out_of_range"}
    assert after.is_valid is False


def test_validate_unknown_chunk(service: OrbitService) -> None:
    v = service.validate("no-such-chunk:0")
    assert v.found is False
    assert v.is_valid is False


# --------------------------------------------------------------------------
# conflicts
# --------------------------------------------------------------------------
def test_conflicts_are_labelled_inferred(db_path: Path, tmp_path: Path) -> None:
    from orbit.engine import OrbitEngine

    root = tmp_path / "conf"
    root.mkdir()
    write(root, "a.cfg", "The deploy timeout = 30 seconds for staging.\n")
    write(root, "b.cfg", "The deploy timeout = 900 seconds for staging.\n")
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    svc = OrbitService(cfg)
    scan = svc.conflicts(limit=50)
    rows = scan.rows
    assert rows, "expected a numeric conflict candidate across the two cfg files"
    assert all(c.kind == "inferred" for c in rows)
    assert any(c.rule == "numeric_conflict" for c in rows)
    # The snapshot reports how much of the index it scanned.
    assert scan.total >= scan.scanned >= len(rows)


def test_conflicts_view_does_not_persist(db_path: Path, tmp_path: Path) -> None:
    """Opening the conflicts view must be read-only — it can't touch the table.

    ``service.conflicts`` calls ``find_conflicts(..., persist=False)``, so just
    looking at the Detect Conflicts view won't write rows to the DB.
    """
    from orbit.engine import OrbitEngine
    from orbit.storage.db import OrbitDB

    root = tmp_path / "conf"
    root.mkdir()
    write(root, "a.cfg", "The deploy timeout = 30 seconds for staging.\n")
    write(root, "b.cfg", "The deploy timeout = 900 seconds for staging.\n")
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    def conflict_row_count() -> int:
        db = OrbitDB(str(db_path))
        db.open()
        try:
            row = db.query_one("SELECT COUNT(*) AS n FROM conflicts")
            return int(row["n"]) if row else 0
        finally:
            db.close()

    # Seed the conflicts table with a real persisted scan so the baseline isn't
    # zero. That way the assertion proves persist=False neither inserts NOR
    # deletes — find_conflicts(persist=True) runs DELETE FROM conflicts first.
    seed_db = OrbitDB(str(db_path))
    seed_db.open()
    try:
        from orbit.evidence import find_conflicts

        find_conflicts(seed_db, persist=True)
    finally:
        seed_db.close()

    before = conflict_row_count()
    assert before > 0, "fixture should persist at least one candidate when seeded"
    svc = OrbitService(cfg)
    scan = svc.conflicts(limit=50)
    assert scan.rows, "fixture should produce at least one candidate"
    after = conflict_row_count()
    assert after == before, (
        f"viewing conflicts must not write to the DB: "
        f"conflicts rows went {before} -> {after}"
    )


def test_conflicts_empty_on_uninitialized(empty_service: OrbitService) -> None:
    scan = empty_service.conflicts()
    assert scan.rows == ()
    assert scan.total == 0
    assert scan.scanned == 0


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------
def test_index_through_service(db_path: Path, corpus: Path) -> None:
    svc = OrbitService(make_config(db_path))
    snap = svc.index([str(corpus)])
    assert snap.error == ""
    assert snap.ok is True
    assert snap.added == 4
    assert snap.chunks_written > 0
    # A second pass is a no-op.
    again = svc.index([str(corpus)])
    assert again.unchanged == 4
    assert again.added == 0


# --------------------------------------------------------------------------
# snippets: line breaks preserved, capped at 3 lines
# --------------------------------------------------------------------------
def test_snippet_preserves_line_breaks(db_path: Path, tmp_path: Path) -> None:
    from orbit.engine import OrbitEngine

    root = tmp_path / "snip"
    root.mkdir()
    # A python function whose body spans several distinct lines.
    write(
        root,
        "multi.py",
        "def quaternion_scheduler(threads):\n"
        "    total = 0\n"
        "    for t in range(threads):\n"
        "        total += t\n"
        "    return total\n",
    )
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    svc = OrbitService(cfg)
    snap = svc.search("quaternion_scheduler threads", limit=5)
    hit = next((h for h in snap.hits if h.filename == "multi.py"), None)
    assert hit is not None, "expected a hit in the multi-line python file"
    # The snippet keeps its newlines instead of collapsing to one line…
    assert "\n" in hit.snippet
    # …and shows at most 3 content lines (plus an optional ellipsis line).
    lines = hit.snippet.split("\n")
    content_lines = [ln for ln in lines if ln != "…"]
    assert len(content_lines) <= 3
    # No carriage returns slip through.
    assert "\r" not in hit.snippet


def test_snippet_lines_helper_caps_and_ellipsizes() -> None:
    from orbit.tui.service import _snippet_lines

    text = "line one\nline two\nline three\nline four\nline five"
    out = _snippet_lines(text, max_lines=3)
    assert out == "line one\nline two\nline three\n…"
    # Leading/trailing blank lines go; a short text comes back whole.
    assert _snippet_lines("\n\nonly line\n\n", max_lines=3) == "only line"
    # An over-long single line is hard-cut with a trailing ellipsis.
    long_line = "x" * 200
    cut = _snippet_lines(long_line, max_lines=3, width=50)
    assert cut.endswith("…")
    assert len(cut) <= 52  # 50 chars + space + ellipsis


# --------------------------------------------------------------------------
# mcp / settings
# --------------------------------------------------------------------------
def test_mcp_info_lists_the_real_tools(service: OrbitService) -> None:
    snap = service.mcp_info()
    names = {t[0] for t in snap.tools}
    # These are the tools build_tools actually registers.
    assert "orbit_search" in names
    assert "orbit_validate_citation" in names
    assert snap.launch_command.endswith("mcp")


def test_config_rows_are_readonly_strings(service: OrbitService) -> None:
    rows = service.config_rows()
    labels = {label for label, _ in rows}
    assert "Database" in labels
    assert "Embedding provider" in labels
    assert all(isinstance(v, str) for _, v in rows)


# --------------------------------------------------------------------------
# thread safety: the whole reason the service opens per-call engines
# --------------------------------------------------------------------------
def test_service_is_usable_from_a_worker_thread(service: OrbitService) -> None:
    """A search must succeed when driven from a non-main thread.

    This is the exact failure mode that a single shared engine would hit
    (SQLite objects are thread-affine). The service opens a fresh engine per
    call, so it must work.
    """
    out: dict[str, object] = {}

    def worker() -> None:
        try:
            snap = service.search("quaternion scheduling", limit=3)
            out["hits"] = len(snap.hits)
            out["error"] = snap.error
        except Exception as exc:  # noqa: BLE001
            out["exc"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert "exc" not in out, out.get("exc")
    assert out.get("error") == ""
    assert isinstance(out.get("hits"), int) and out["hits"] > 0
