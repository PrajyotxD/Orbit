"""Headless Textual tests for the Orbit TUI app (orbit.tui.app).

Driven with Textual's ``App.run_test`` pilot — no real terminal is attached.
Every test asserts against the real engine's output: search hits, navigation
routing, filters, empty/degraded states and the detail panel. The whole module
is skipped if textual is not installed.

The suite has no ``pytest-asyncio`` dependency (and the project runs
``--strict-markers``, so the ``asyncio`` marker is unavailable): each test is a
plain synchronous function wrapping an async body run via :func:`asyncio.run`,
through the :func:`_pilot_test` decorator.
"""

from __future__ import annotations

import asyncio
import inspect
import socket
from pathlib import Path

import pytest

pytest.importorskip("textual")

from orbit.engine import OrbitEngine  # noqa: E402
from orbit.tui.app import OrbitTUI  # noqa: E402
from orbit.tui.widgets import ResultItem  # noqa: E402

from conftest import make_config, write  # noqa: E402

from textual.widgets import Input, ListView, Static  # noqa: E402
from textual.widgets import ContentSwitcher  # noqa: E402


# --------------------------------------------------------------------------
# The conftest autouse `_offline_env` fixture replaces socket.socket with one
# that raises on ANY connect. That is correct for the engine tests, but it also
# trips the asyncio event loop's own loopback self-pipe on Windows (the
# ProactorEventLoop builds it with socket.socketpair()), which has nothing to do
# with network egress. Override it here with a loopback-aware guard: real
# external connects still fail loudly, loopback is allowed so run_test works.
# --------------------------------------------------------------------------
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "", "0.0.0.0", "::"}


def _is_loopback(address: object) -> bool:
    if isinstance(address, (bytes, str)):
        return True  # AF_UNIX / named pipe path — not network egress
    try:
        host = address[0]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return False
    return str(host) in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch):  # noqa: PT004 - overrides conftest fixture
    real_socket = socket.socket

    class _LoopbackOnlySocket(real_socket):  # type: ignore[misc, valid-type]
        def connect(self, address, *a, **k):  # type: ignore[override]
            if not _is_loopback(address):
                raise AssertionError(f"network access attempted: {address!r}")
            return super().connect(address, *a, **k)

        def connect_ex(self, address, *a, **k):  # type: ignore[override]
            if not _is_loopback(address):
                raise AssertionError(f"network access attempted: {address!r}")
            return super().connect_ex(address, *a, **k)

    monkeypatch.setattr(socket, "socket", _LoopbackOnlySocket)
    yield


def _pilot_test(fn):
    """Run an ``async def`` test body under a fresh event loop.

    Pytest has no native async support here and the project runs
    ``--strict-markers`` (no ``asyncio`` marker), so each async body is wrapped
    in a plain sync function. The real signature is published via
    ``__signature__`` so pytest still injects fixtures by name; ``__wrapped__``
    is deliberately NOT set (that would make pytest unwrap to the coroutine and
    treat the test as an unsupported ``async def``).
    """

    def wrapper(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__signature__ = inspect.signature(fn)
    return wrapper


async def _wait_idle(app: OrbitTUI, pilot) -> None:
    """Let background workers finish and the UI settle.

    Awaits every worker individually (so thread workers actually run to
    completion), but swallows the ``WorkerCancelled`` / ``WorkerFailed`` that a
    replaced *exclusive* worker raises — those are expected when, e.g., a new
    search supersedes an in-flight related-files lookup. ``wait_for_complete``
    can't be used directly because it re-raises ``WorkerCancelled``.
    """
    from textual.worker import WorkerError

    # Snapshot: new workers may be spawned by call_from_thread callbacks, so
    # loop until no worker is left pending/running.
    for _ in range(200):  # ~ generous upper bound; each pass yields to the loop
        pending = [w for w in list(app.workers) if not w.is_finished]
        if not pending:
            break
        for w in pending:
            try:
                await w.wait()
            except WorkerError:
                pass  # cancelled/failed exclusive worker — expected
        await pilot.pause()
    await pilot.pause()


def _plain(widget_static: Static) -> str:
    rendered = widget_static.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def indexed_config(db_path: Path, corpus: Path):
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([corpus])
    eng.close()
    return cfg


@pytest.fixture
def empty_config(db_path: Path):
    return make_config(db_path)


# --------------------------------------------------------------------------
# boot
# --------------------------------------------------------------------------
@_pilot_test
async def test_app_boots_with_three_panels(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        assert app.query_one("#sidebar")
        assert app.query_one("#details")
        switcher = app.query_one("#center", ContentSwitcher)
        assert switcher.current == "view-search"


@_pilot_test
async def test_topbar_shows_real_file_count(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        # The sidebar info box renders the indexed-file count from the engine.
        assert "4" in _plain(app.query_one("#info-box", Static))


# --------------------------------------------------------------------------
# navigation
# --------------------------------------------------------------------------
@_pilot_test
async def test_number_keys_route_between_views(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        switcher = app.query_one("#center", ContentSwitcher)

        # Must not be focused on an Input or digits get typed instead of routing.
        app.set_focus(None)
        await pilot.press("5")
        await _wait_idle(app, pilot)
        assert switcher.current == "view-conflicts"

        app.set_focus(None)
        await pilot.press("7")
        await _wait_idle(app, pilot)
        assert switcher.current == "view-settings"

        app.set_focus(None)
        await pilot.press("2")
        await pilot.pause()
        assert switcher.current == "view-search"


@_pilot_test
async def test_slash_focuses_search(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        app.set_focus(None)
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.focused, Input)
        assert app.focused.id == "search-input"


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
@_pilot_test
async def test_search_populates_results_and_details(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        app.action_focus_search()
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        results = app.query_one("#results", ListView)
        items = [c for c in results.children if isinstance(c, ResultItem)]
        assert items, "expected real hits in the results list"
        # No relevance percentage; the fused score is a real float.
        assert not hasattr(items[0].hit, "relevance_pct")
        assert isinstance(items[0].hit.score, float)
        # The detail panel filled in for the auto-selected first hit.
        assert items[0].hit.path in _plain(app.query_one("#detail-fileinfo", Static))


@_pilot_test
async def test_search_no_match_is_reported_not_fabricated(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "zzz_absent_term_qwxyz"
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        results = app.query_one("#results", ListView)
        items = [c for c in results.children if isinstance(c, ResultItem)]
        assert items == []
        assert "No results" in _plain(app.query_one("#result-meta", Static))


@_pilot_test
async def test_search_on_empty_index_degrades_cleanly(empty_config) -> None:
    app = OrbitTUI(empty_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "anything"
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        results = app.query_one("#results", ListView)
        items = [c for c in results.children if isinstance(c, ResultItem)]
        assert items == []
        rendered = _plain(app.query_one("#result-meta", Static))
        assert "not initialized" in rendered or "No results" in rendered


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
@_pilot_test
async def test_filter_cycle_filters_results(db_path: Path, tmp_path: Path) -> None:
    # Build a corpus with both code and docs so a filter actually changes the set.
    root = tmp_path / "mixed"
    root.mkdir()
    write(root, "mod.py", "def scheduling_quaternion():\n    return 'quaternion scheduling'\n")
    write(root, "guide.md", "# Guide\n\nThe quaternion scheduling subsystem is documented here.\n")
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    app = OrbitTUI(cfg)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        def result_paths() -> list[str]:
            results = app.query_one("#results", ListView)
            return [c.hit.path for c in results.children if isinstance(c, ResultItem)]

        all_paths = result_paths()
        assert any(p.endswith(".py") for p in all_paths)
        assert any(p.endswith(".md") for p in all_paths)

        # Cycle one step: all -> code.
        app.set_focus(None)
        await pilot.press("f")
        await pilot.pause()
        assert app._filter == "code"
        code_paths = result_paths()
        assert code_paths and all(p.endswith(".py") for p in code_paths)


# --------------------------------------------------------------------------
# index view
# --------------------------------------------------------------------------
@_pilot_test
async def test_index_view_runs_the_engine(empty_config, corpus: Path) -> None:
    app = OrbitTUI(empty_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        app.set_focus(None)
        await pilot.press("1")  # Index Repository
        await pilot.pause()
        index_input = app.query_one("#index-input", Input)
        app.set_focus(index_input)
        index_input.value = str(corpus)
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        rendered = _plain(app.query_one("#index-log", Static))
        assert "Index complete" in rendered
        assert "Added" in rendered
        # Having really indexed, a follow-up search now returns hits.
        app.set_focus(None)
        await pilot.press("2")
        await pilot.pause()
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        results = app.query_one("#results", ListView)
        assert [c for c in results.children if isinstance(c, ResultItem)]


# --------------------------------------------------------------------------
# validate view
# --------------------------------------------------------------------------
@_pilot_test
async def test_validate_view_reports_state(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        # Grab a real chunk id via a search first.
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        chunk_id = app._selected_chunk_id
        assert chunk_id

        app.set_focus(None)
        await pilot.press("4")  # Validate Citations
        await pilot.pause()
        vin = app.query_one("#validate-input", Input)
        # The validate field is pre-filled with the selected chunk.
        assert vin.value == chunk_id
        app.set_focus(vin)
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        rendered = _plain(app.query_one("#validate-block", Static))
        assert "valid" in rendered
        assert chunk_id in rendered
        # No raw python booleans slip into the rendered output.
        assert "True" not in rendered
        assert "False" not in rendered
        # The state WORD is shown, not just an icon or colour.
        assert "state" in rendered


@_pilot_test
async def test_e_and_v_keys_prefill_from_selected_hit(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        chunk_id = app._selected_chunk_id
        assert chunk_id

        # `e` opens Explore Evidence prefilled with the selected hit.
        app.set_focus(None)
        await pilot.press("e")
        await _wait_idle(app, pilot)
        assert app.query_one("#center", ContentSwitcher).current == "view-evidence"
        assert app.query_one("#evidence-input", Input).value == chunk_id

        # Back on search, `v` opens Validate prefilled with the same hit.
        app.set_focus(None)
        await pilot.press("2")
        await pilot.pause()
        app.set_focus(None)
        await pilot.press("v")
        await _wait_idle(app, pilot)
        assert app.query_one("#center", ContentSwitcher).current == "view-validate"
        assert app.query_one("#validate-input", Input).value == chunk_id


@_pilot_test
async def test_r_revalidates_and_flips_badge_after_edit(db_path: Path, tmp_path: Path) -> None:
    """Press `r` after editing one source file: that row's badge turns
    content_changed while the untouched rows stay valid."""
    from orbit.engine import OrbitEngine

    root = tmp_path / "reval"
    root.mkdir()
    write(root, "edited.py", "def quaternion_scheduler():\n    return 'quaternion scheduling alpha'\n")
    write(root, "stable.py", "def other_quaternion():\n    return 'quaternion scheduling beta'\n")
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    app = OrbitTUI(cfg)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        results = app.query_one("#results", ListView)
        items = [c for c in results.children if isinstance(c, ResultItem)]
        assert items, "expected hits across both files"
        by_name = {it.hit.filename: it for it in items}
        assert "edited.py" in by_name and "stable.py" in by_name
        # Freshly indexed, so every visible row validates.
        assert all(it.hit.validation == "valid" for it in items)

        # Edit edited.py's cited region out from under the index.
        (root / "edited.py").write_text(
            "PREPENDED LINE A\nPREPENDED LINE B\n"
            + (root / "edited.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        # Press r on the Search view to re-validate every visible hit.
        app.set_focus(None)
        await pilot.press("r")
        await _wait_idle(app, pilot)

        edited = by_name["edited.py"].hit
        stable = by_name["stable.py"].hit
        assert edited.validation in {"content_changed", "chunk_out_of_range"}, edited.validation
        assert edited.validation != "valid"
        assert stable.validation == "valid"
        # The rendered card badge shows the new state as a word, not an icon alone.
        edited_card = _plain(by_name["edited.py"].query_one(".result-card", Static))
        assert edited.validation in edited_card


# --------------------------------------------------------------------------
# conflicts view
# --------------------------------------------------------------------------
@_pilot_test
async def test_conflicts_view_lists_candidates(db_path: Path, tmp_path: Path) -> None:
    root = tmp_path / "conf"
    root.mkdir()
    write(root, "a.cfg", "The deploy timeout = 30 seconds for staging.\n")
    write(root, "b.cfg", "The deploy timeout = 900 seconds for staging.\n")
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    app = OrbitTUI(cfg)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        app.set_focus(None)
        await pilot.press("5")  # Detect Conflicts
        await _wait_idle(app, pilot)
        rendered = _plain(app.query_one("#conflicts-list", Static))
        assert "candidate" in rendered
        assert "inferred" in rendered
        # The view reports its scan coverage.
        assert "scanned" in rendered


@_pilot_test
async def test_conflicts_view_does_not_change_table_row_count(db_path: Path, tmp_path: Path) -> None:
    """Opening Detect Conflicts in the UI must not write to the conflicts table."""
    from orbit.storage.db import OrbitDB
    from orbit.evidence import find_conflicts

    root = tmp_path / "conf"
    root.mkdir()
    write(root, "a.cfg", "The deploy timeout = 30 seconds for staging.\n")
    write(root, "b.cfg", "The deploy timeout = 900 seconds for staging.\n")
    cfg = make_config(db_path)
    eng = OrbitEngine(cfg)
    eng.index([root])
    eng.close()

    def row_count() -> int:
        db = OrbitDB(str(db_path))
        db.open()
        try:
            row = db.query_one("SELECT COUNT(*) AS n FROM conflicts")
            return int(row["n"]) if row else 0
        finally:
            db.close()

    # Seed a persisted scan so the table starts non-empty.
    seed = OrbitDB(str(db_path))
    seed.open()
    try:
        find_conflicts(seed, persist=True)
    finally:
        seed.close()
    before = row_count()
    assert before > 0

    app = OrbitTUI(cfg)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        app.set_focus(None)
        await pilot.press("5")  # Detect Conflicts
        await _wait_idle(app, pilot)
        # Rescan from the view as well (its own refresh path).
        await pilot.press("r")
        await _wait_idle(app, pilot)

    after = row_count()
    assert after == before, f"conflicts table row count changed: {before} -> {after}"


@_pilot_test
async def test_detail_panel_hidden_on_non_detail_views(indexed_config) -> None:
    from orbit.tui.widgets import SourceDetails

    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        # Populate the panel on Search.
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)
        details = app.query_one("#details", SourceDetails)
        assert not details.has_class("detail-hidden")

        # Navigate to Settings; the panel should hide and clear.
        app.set_focus(None)
        await pilot.press("7")
        await _wait_idle(app, pilot)
        assert details.has_class("detail-hidden")
        assert app._selected_chunk_id  # selection is remembered…
        # …but the file-info pane no longer shows a hit path.
        assert "No result selected" in _plain(app.query_one("#detail-fileinfo", Static))

        # Back to Search — the panel is visible again.
        app.set_focus(None)
        await pilot.press("2")
        await pilot.pause()
        assert not details.has_class("detail-hidden")


@_pilot_test
async def test_no_percentage_rendered_anywhere(indexed_config) -> None:
    """The relevance '%' badge is gone from result cards and the detail panel."""
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        search = app.query_one("#search-input", Input)
        app.set_focus(search)
        search.value = "quaternion scheduling"
        await pilot.press("enter")
        await _wait_idle(app, pilot)

        results = app.query_one("#results", ListView)
        items = [c for c in results.children if isinstance(c, ResultItem)]
        assert items
        for it in items:
            card = _plain(it.query_one(".result-card", Static))
            assert "%" not in card, f"result card still shows a percentage: {card!r}"
            assert "relevance" not in card.lower()
        fileinfo = _plain(app.query_one("#detail-fileinfo", Static))
        assert "%" not in fileinfo
        assert "relevance" not in fileinfo.lower()
        # The fused score is shown instead.
        assert "score" in fileinfo.lower()


# --------------------------------------------------------------------------
# mcp view
# --------------------------------------------------------------------------
@_pilot_test
async def test_mcp_view_lists_tools(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        app.set_focus(None)
        await pilot.press("6")  # MCP Server
        await _wait_idle(app, pilot)
        rendered = _plain(app.query_one("#mcp-block", Static))
        assert "orbit_search" in rendered
        assert "mcp" in rendered.lower()


# --------------------------------------------------------------------------
# responsive layout + clean exit
# --------------------------------------------------------------------------
@_pilot_test
async def test_narrow_layout_hides_side_panels(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(60, 30)) as pilot:
        await _wait_idle(app, pilot)
        # Below the narrow threshold the screen carries the 'narrow' class.
        assert app.screen.has_class("narrow")


@_pilot_test
async def test_quit_action_exits(indexed_config) -> None:
    app = OrbitTUI(indexed_config)
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_idle(app, pilot)
        app.set_focus(None)
        await pilot.press("q")
        await pilot.pause()
    # The run_test context exits cleanly once the app has exited.
    assert app.return_code == 0 or app.return_code is None
