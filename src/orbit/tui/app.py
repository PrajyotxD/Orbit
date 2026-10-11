"""Orbit full-screen terminal UI (Textual).

An interactive front end over the existing Orbit engine. It owns no retrieval or
indexing logic of its own: every operation goes to
:class:`orbit.tui.service.OrbitService`, which opens a fresh, thread-local
:class:`~orbit.engine.OrbitEngine` per call so blocking work can run on Textual
worker threads without tripping SQLite's thread affinity.

Layout mirrors the reference design:

    ┌ TopBar (identity · database · repo · file count) ───────────────┐
    │ Sidebar │ Center (context-aware view, ContentSwitcher) │ Details │
    └ StatusBar (keyboard hints · live status) ───────────────────────┘

Everything on screen comes from the real index. An empty or uninitialised index
says so, degraded retrieval is surfaced, and validation and conflict results are
shown exactly as the engine returns them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)
from textual.widgets import (
    ContentSwitcher,
)
from textual.worker import get_current_worker

from ..config import OrbitConfig
from .icons import DEFAULT_ICON_SET, IconSet, get_icon_set
from .service import (
    CATEGORIES,
    ConflictScanSnapshot,
    EvidenceSnapshot,
    IndexSnapshot,
    McpSnapshot,
    OrbitService,
    SearchSnapshot,
    StatusSnapshot,
    ValidationSnapshot,
)
from .widgets import (
    NAV_ITEMS,
    ResultItem,
    Sidebar,
    SourceDetails,
    StatusBar,
    TopBar,
)

__all__ = ["OrbitTUI", "run"]

#: Live-search debounce window, seconds.
_DEBOUNCE = 0.3

#: Context-sensitive bottom-bar hints per view.
_HINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "view-search": (
        ("Up/Dn", "Navigate"), ("Enter", "Open"), ("e", "Evidence"),
        ("v", "Validate"), ("r", "Re-validate"), ("f", "Filter"), ("q", "Quit"),
    ),
    "view-index": (
        ("Enter", "Index"), ("/", "Search"), ("r", "Refresh"), ("q", "Quit"),
    ),
    "view-evidence": (
        ("Enter", "Explore"), ("/", "Search"), ("Esc", "Back"), ("q", "Quit"),
    ),
    "view-validate": (
        ("Enter", "Validate"), ("/", "Search"), ("Esc", "Back"), ("q", "Quit"),
    ),
    "view-conflicts": (
        ("r", "Rescan"), ("/", "Search"), ("Esc", "Back"), ("q", "Quit"),
    ),
    "view-mcp": (("/", "Search"), ("Esc", "Back"), ("q", "Quit")),
    "view-settings": (("r", "Refresh"), ("/", "Search"), ("Esc", "Back"), ("q", "Quit")),
}
_DEFAULT_HINTS = (("/", "Search"), ("Esc", "Back"), ("q", "Quit"))


class OrbitTUI(App):
    """The Orbit terminal application."""

    CSS_PATH = "orbit.tcss"
    TITLE = "Orbit — Knowledge Engine"

    BINDINGS = [
        Binding("slash", "focus_search", "Search", show=False),
        Binding("escape", "escape", "Back", show=False),
        Binding("q", "request_quit", "Quit", show=False),
        Binding("f", "cycle_filter", "Filter", show=False),
        Binding("r", "refresh_view", "Refresh", show=False),
        Binding("e", "explore_selected", "Explore evidence", show=False),
        Binding("v", "validate_selected", "Validate", show=False),
        Binding("1", "nav('view-index')", "Index", show=False),
        Binding("2", "nav('view-search')", "Search", show=False),
        Binding("3", "nav('view-evidence')", "Evidence", show=False),
        Binding("4", "nav('view-validate')", "Validate", show=False),
        Binding("5", "nav('view-conflicts')", "Conflicts", show=False),
        Binding("6", "nav('view-mcp')", "MCP", show=False),
        Binding("7", "nav('view-settings')", "Settings", show=False),
        Binding("0", "nav('view-exit')", "Exit", show=False),
    ]

    #: Views that use the right-hand Source Details panel. On any other view the
    #: panel is cleared and hidden so an old hit can't sit there as stale context.
    _DETAIL_VIEWS = frozenset({"view-search", "view-evidence", "view-validate"})

    def __init__(
        self,
        config: OrbitConfig,
        *,
        default_index_path: str = "",
        icon_set: str | None = None,
    ) -> None:
        super().__init__()
        self.service = OrbitService(config)
        self._config = config
        self._default_index_path = default_index_path
        # Resolve the active icon set once; widgets read it off ``app.icons``.
        self.icons: IconSet = get_icon_set(icon_set or DEFAULT_ICON_SET)
        self._active_view = "view-search"
        self._nav_settled = False
        self._suppress_nav = False
        self._filter = "all"
        self._last_search: SearchSnapshot | None = None
        self._selected_chunk_id = ""
        self._search_timer = None

    # ---------------------------------------------------------------- compose
    def compose(self) -> ComposeResult:
        yield TopBar(id="topbar")
        with Horizontal(id="body"):
            sidebar = Sidebar(id="sidebar")
            yield sidebar
            with ContentSwitcher(initial="view-search", id="center"):
                yield from self._compose_search_view()
                yield from self._compose_index_view()
                yield from self._compose_evidence_view()
                yield from self._compose_validate_view()
                yield from self._compose_conflicts_view()
                yield from self._compose_mcp_view()
                yield from self._compose_settings_view()
            yield SourceDetails(id="details")
        yield StatusBar(id="statusbar")

    # -- center views -------------------------------------------------------
    def _compose_search_view(self) -> ComposeResult:
        with Vertical(id="view-search"):
            yield Static(f"{self.icons.search}  Search Knowledge", classes="view-title")
            yield Static(
                "Find relevant passages from your indexed corpus, each with a verifiable citation.",
                classes="view-subtitle",
            )
            yield Input(placeholder="Type a query and press Enter…", id="search-input")
            yield Static("", id="filter-bar")
            yield Static("Type a query to search.", id="result-meta")
            yield ListView(id="results")

    def _compose_index_view(self) -> ComposeResult:
        with Vertical(id="view-index"):
            yield Static(f"{self.icons.index}  Index Repository", classes="view-title")
            yield Static(
                "Scan files or directories into the index. Re-indexing is incremental: "
                "only changed files are rewritten.",
                classes="view-subtitle",
            )
            yield Input(
                value=self._default_index_path,
                placeholder="path/to/repo  (space-separated paths allowed)",
                id="index-input",
            )
            with Horizontal(id="index-controls"):
                yield Button("Index", id="index-button", variant="success")
            yield Static("", id="index-status", classes="hint")
            yield VerticalScroll(
                Static(self._index_placeholder(), id="index-log"),
                classes="panel-scroll",
            )

    def _compose_evidence_view(self) -> ComposeResult:
        with Vertical(id="view-evidence"):
            yield Static(f"{self.icons.evidence}  Explore Evidence", classes="view-title")
            yield Static(
                "Walk the evidence graph outward from a chunk. Explicit edges are structural "
                "facts; inferred edges are heuristics and are labelled as such.",
                classes="view-subtitle",
            )
            yield Input(placeholder="chunk_id (from a search result)", id="evidence-input")
            yield VerticalScroll(
                Static(self._evidence_placeholder(), id="evidence-block"),
                classes="panel-scroll",
            )

    def _compose_validate_view(self) -> ComposeResult:
        with Vertical(id="view-validate"):
            yield Static(f"{self.icons.validate}  Validate Citations", classes="view-title")
            yield Static(
                "Re-read the cited lines from disk and compare hashes. This checks provenance "
                "(does the pointer still resolve), not whether the passage is true.",
                classes="view-subtitle",
            )
            yield Input(placeholder="chunk_id to validate", id="validate-input")
            yield VerticalScroll(
                Static(self._validate_placeholder(), id="validate-block"),
                classes="panel-scroll",
            )

    def _compose_conflicts_view(self) -> ComposeResult:
        with Vertical(id="view-conflicts"):
            yield Static(f"{self.icons.conflict}  Detect Conflicts", classes="view-title")
            yield Static(
                "Heuristic conflict candidates across the corpus. Treat each as a "
                "lead worth checking, not a finding: every candidate is labelled "
                "'inferred' and names the rule that flagged it.",
                classes="view-subtitle",
            )
            yield VerticalScroll(Static("", id="conflicts-list"), classes="panel-scroll")

    def _compose_mcp_view(self) -> ComposeResult:
        with Vertical(id="view-mcp"):
            yield Static(f"{self.icons.mcp}  MCP Server", classes="view-title")
            yield Static(
                "Orbit exposes its tools over MCP (stdio). The server runs as a separate "
                "process so it stays independent of this UI.",
                classes="view-subtitle",
            )
            yield VerticalScroll(Static("", id="mcp-block"), classes="panel-scroll")

    def _compose_settings_view(self) -> ComposeResult:
        with Vertical(id="view-settings"):
            yield Static(f"{self.icons.settings}  Settings & Status", classes="view-title")
            yield Static(
                "Active configuration and live index status (read-only).",
                classes="view-subtitle",
            )
            yield VerticalScroll(Static("", id="settings-block"), classes="panel-scroll")

    # -- empty-state placeholders (so no view shows a blank bordered box) ----
    @staticmethod
    def _index_placeholder():
        from rich.text import Text

        t = Text()
        t.append("Indexing output appears here.\n\n", style="bold #6e7681")
        t.append("Enter one or more paths above and press ", style="#6e7681")
        t.append("Enter", style="#adbac7")
        t.append(" (or click ", style="#6e7681")
        t.append("Index", style="#3fb950")
        t.append("). The real engine runs on a worker thread, and this panel "
                 "then shows the added / updated / deleted counts and any "
                 "failures.", style="#6e7681")
        return t

    @staticmethod
    def _evidence_placeholder():
        from rich.text import Text

        t = Text()
        t.append("Evidence graph appears here.\n\n", style="bold #6e7681")
        t.append("Paste a ", style="#6e7681")
        t.append("chunk_id", style="#adbac7")
        t.append(" from a search result and press Enter to walk its outgoing "
                 "edges (same-document, shared-symbol, links).", style="#6e7681")
        return t

    @staticmethod
    def _validate_placeholder():
        from rich.text import Text

        t = Text()
        t.append("Validation result appears here.\n\n", style="bold #6e7681")
        t.append("Paste a ", style="#6e7681")
        t.append("chunk_id", style="#adbac7")
        t.append(" and press Enter. Orbit re-reads the cited lines from disk and "
                 "compares content hashes to confirm the citation still resolves.",
                 style="#6e7681")
        return t

    # ------------------------------------------------------------------ mount
    def on_mount(self) -> None:
        # Build the sidebar navigation list, already highlighted on the default
        # view so the first (async) Highlighted message matches self._active_view
        # and the handler no-ops. Setting the highlight via initial_index avoids
        # a mount-time race where the list's default index 0 ("view-index")
        # would switch the view out from under us.
        items = [
            ListItem(Label(f"{key}  {self.icons.glyph(icon_name)}  {label}"), id=f"nav-{view_id}")
            for key, view_id, label, icon_name in NAV_ITEMS
        ]
        default_index = next(
            (i for i, it in enumerate(NAV_ITEMS) if it[1] == self._active_view), 0
        )
        nav = ListView(*items, initial_index=default_index, id="nav")
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.mount(nav, after="#logo")
        self._set_hints(self._active_view)
        self.query_one("#filter-bar", Static).update(self._filter_bar_text())
        self._refresh_top()
        self.set_focus(self.query_one("#search-input", Input))
        self._apply_responsive(self.size.width)
        self.call_after_refresh(self._mark_nav_settled)

    def _mark_nav_settled(self) -> None:
        self._nav_settled = True

    # --------------------------------------------------------------- routing
    def action_nav(self, view_id: str) -> None:
        self.switch_view(view_id)

    def switch_view(self, view_id: str) -> None:
        if view_id == "view-exit":
            self.action_request_quit()
            return
        if view_id == self._active_view and view_id != "view-conflicts":
            # Re-selecting a data view does nothing, except Conflicts, which the
            # user may want to rescan; it has its own explicit refresh key.
            self._focus_primary(view_id)
            return
        self._active_view = view_id
        try:
            self.query_one("#center", ContentSwitcher).current = view_id
        except Exception:  # noqa: BLE001 - view id must exist; defensive only
            return
        self._apply_detail_visibility(view_id)
        self._sync_nav_highlight(view_id)
        self._set_hints(view_id)
        self._focus_primary(view_id)
        self._on_enter_view(view_id)

    def _apply_detail_visibility(self, view_id: str) -> None:
        """Show the Source Details panel only on views that use it.

        On Index, Conflicts, MCP and Settings the panel is cleared and hidden,
        so a hit selected on Search doesn't sit there as stale context beside
        unrelated output. This uses a CSS class rather than an inline
        ``display`` so it composes with the responsive ``Screen.narrow/compact``
        rules instead of overriding them.
        """
        try:
            details = self.query_one("#details", SourceDetails)
        except Exception:  # noqa: BLE001
            return
        if view_id in self._DETAIL_VIEWS:
            details.remove_class("detail-hidden")
        else:
            details.clear()
            details.add_class("detail-hidden")

    def _on_enter_view(self, view_id: str) -> None:
        """Lazily load data when a view that needs it becomes active."""
        if view_id == "view-conflicts":
            self.query_one("#conflicts-list", Static).update("Scanning for conflict candidates…")
            self._set_busy("Scanning conflicts…")
            self._conflicts_worker()
        elif view_id == "view-mcp":
            self._mcp_worker()
        elif view_id == "view-settings":
            self.query_one("#settings-block", Static).update("Loading status…")
            self._set_busy("Loading status…")
            self._settings_worker()
        elif view_id == "view-evidence":
            inp = self.query_one("#evidence-input", Input)
            if not inp.value and self._selected_chunk_id:
                inp.value = self._selected_chunk_id
        elif view_id == "view-validate":
            inp = self.query_one("#validate-input", Input)
            if not inp.value and self._selected_chunk_id:
                inp.value = self._selected_chunk_id

    def _focus_primary(self, view_id: str) -> None:
        focus_map = {
            "view-search": "#search-input",
            "view-index": "#index-input",
            "view-evidence": "#evidence-input",
            "view-validate": "#validate-input",
        }
        sel = focus_map.get(view_id)
        if sel:
            try:
                inp = self.query_one(sel, Input)
                self.set_focus(inp)
                # The index path is often long, so move the cursor to the end
                # and let the field scroll to show the tail of the path instead
                # of cutting it off at the start.
                if view_id == "view-index" and inp.value:
                    inp.cursor_position = len(inp.value)
            except Exception:  # noqa: BLE001
                pass

    def _sync_nav_highlight(self, view_id: str) -> None:
        try:
            nav = self.query_one("#nav", ListView)
        except Exception:  # noqa: BLE001
            return
        target = f"nav-{view_id}"
        self._suppress_nav = True
        try:
            for i, child in enumerate(nav.children):
                if child.id == target:
                    nav.index = i
                    break
        finally:
            self._suppress_nav = False

    @on(ListView.Highlighted, "#nav")
    def _nav_highlighted(self, event: ListView.Highlighted) -> None:
        if self._suppress_nav or event.item is None or event.item.id is None:
            return
        view_id = event.item.id.removeprefix("nav-")
        self.switch_view(view_id)

    @on(ListView.Selected, "#nav")
    def _nav_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return
        view_id = event.item.id.removeprefix("nav-")
        self.switch_view(view_id)

    # ----------------------------------------------------------- status bar
    def _set_hints(self, view_id: str) -> None:
        self.query_one("#statusbar", StatusBar).set_hints(_HINTS.get(view_id, _DEFAULT_HINTS))

    def _set_ready(self, message: str = "Ready") -> None:
        self.query_one("#statusbar", StatusBar).set_status(message, kind="ready")

    def _set_busy(self, message: str) -> None:
        self.query_one("#statusbar", StatusBar).set_status(message, kind="busy")

    def _set_error(self, message: str) -> None:
        self.query_one("#statusbar", StatusBar).set_status(message, kind="error")

    # ----------------------------------------------------------- top/sidebar
    def _refresh_top(self) -> None:
        self._top_worker()

    @work(thread=True, exclusive=True, group="top")
    def _top_worker(self) -> None:
        info = self.service.top_info()
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_top, info)

    def _apply_top(self, info) -> None:
        self.query_one("#topbar", TopBar).update_info(info)
        self.query_one("#sidebar", Sidebar).update_info_box(info)
        if not info.initialized:
            self._set_ready("Index not initialized — press 1 to index")
        else:
            self._set_ready()

    # --------------------------------------------------------------- search
    @on(Input.Changed, "#search-input")
    def _search_changed(self, event: Input.Changed) -> None:
        if self._search_timer is not None:
            self._search_timer.stop()
        query = event.value
        if not query.strip():
            self._last_search = None
            self.query_one("#results", ListView).clear()
            self.query_one("#result-meta", Static).update("Type a query to search.")
            self.query_one("#details", SourceDetails).clear()
            return
        self._search_timer = self.set_timer(_DEBOUNCE, lambda: self._run_search(query))

    @on(Input.Submitted, "#search-input")
    def _search_submitted(self, event: Input.Submitted) -> None:
        if self._search_timer is not None:
            self._search_timer.stop()
        self._run_search(event.value)

    def _run_search(self, query: str) -> None:
        if not query.strip():
            return
        self._set_busy(f"Searching “{query.strip()[:40]}”…")
        self.query_one("#result-meta", Static).update("Searching…")
        self._search_worker(query)

    @work(thread=True, exclusive=True, group="search")
    def _search_worker(self, query: str) -> None:
        snap = self.service.search(query, limit=25)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_search, snap)

    def _apply_search(self, snap: SearchSnapshot) -> None:
        self._last_search = snap
        self._render_results()
        if snap.error:
            self._set_error(f"Search failed: {snap.error[:60]}")
        else:
            self._set_ready(f"{len(snap.hits)} result(s) in {snap.duration_ms:.0f} ms")

    def _render_results(self) -> None:
        results = self.query_one("#results", ListView)
        results.clear()
        meta = self.query_one("#result-meta", Static)
        details = self.query_one("#details", SourceDetails)
        snap = self._last_search
        self.query_one("#filter-bar", Static).update(self._filter_bar_text())
        if snap is None:
            meta.update("Type a query to search.")
            details.clear()
            return
        if snap.error:
            meta.update(f"[error] {snap.error}")
            details.clear()
            return
        hits = snap.hits
        if self._filter != "all":
            hits = tuple(h for h in hits if h.category == self._filter)
        if not snap.hits:
            note = snap.degraded_reason or "no matches"
            meta.update(f"No results for “{snap.query}” — {note}")
            details.clear()
            return
        channels = "+".join(snap.channels_used) or "none"
        shown = f"{len(hits)} of {len(snap.hits)}" if self._filter != "all" else f"{len(snap.hits)}"
        meta_text = f"Found {shown} result(s) in {snap.duration_ms:.0f} ms · via {channels}"
        if snap.degraded_reason:
            meta_text += f" · degraded: {snap.degraded_reason}"
        meta.update(meta_text)
        if not hits:
            details.clear()
            meta.update(meta_text + f"  —  no '{self._filter}' results in this query.")
            return
        for hit in hits:
            results.append(ResultItem(hit))
        # Auto-select the first result to populate the details panel.
        results.index = 0
        self._show_detail(hits[0])

    @on(ListView.Highlighted, "#results")
    def _result_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if isinstance(item, ResultItem):
            self._show_detail(item.hit)

    @on(ListView.Selected, "#results")
    def _result_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ResultItem):
            self._show_detail(item.hit)
            try:
                self.set_focus(self.query_one("#details", SourceDetails))
            except Exception:  # noqa: BLE001
                pass

    def _show_detail(self, hit) -> None:
        self._selected_chunk_id = hit.chunk_id
        details = self.query_one("#details", SourceDetails)
        details.show_hit(hit)
        self._related_worker(hit.chunk_id)

    @work(thread=True, exclusive=True, group="related")
    def _related_worker(self, chunk_id: str) -> None:
        snap = self.service.evidence(chunk_id, max_depth=1)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_related, snap)

    def _apply_related(self, snap: EvidenceSnapshot) -> None:
        # Only paint if this result is still the selected chunk.
        if snap.chunk_id != self._selected_chunk_id:
            return
        self.query_one("#details", SourceDetails).set_related(snap)

    # --------------------------------------------------------------- filters
    def action_cycle_filter(self) -> None:
        if self._active_view != "view-search":
            return
        names = [c[0] for c in CATEGORIES]
        idx = names.index(self._filter) if self._filter in names else 0
        self._filter = names[(idx + 1) % len(names)]
        self._render_results()
        self._set_ready(f"Filter: {self._filter}")

    def _filter_bar_text(self):
        from rich.text import Text

        text = Text(no_wrap=True, overflow="ellipsis")
        for i, (name, _langs) in enumerate(CATEGORIES):
            if i:
                text.append("   ")
            label = name.capitalize()
            if name == self._filter:
                text.append(f" {label} ", style="bold #0a0e14 on #3fb950")
            else:
                text.append(label, style="#6e7681")
        text.append("   ·  f: cycle", style="#484f58")
        return text

    # ---------------------------------------------------------------- index
    @on(Button.Pressed, "#index-button")
    def _index_pressed(self, event: Button.Pressed) -> None:
        self._start_index()

    @on(Input.Submitted, "#index-input")
    def _index_submitted(self, event: Input.Submitted) -> None:
        self._start_index()

    def _start_index(self) -> None:
        raw = self.query_one("#index-input", Input).value.strip()
        paths = tuple(p for p in raw.split() if p) if raw else ()
        if not paths and not self._config.paths:
            self.query_one("#index-status", Static).update(
                "Enter at least one path to index (or configure paths in your config)."
            )
            self._set_error("No path to index")
            return
        label = " ".join(paths) if paths else "configured paths"
        self.query_one("#index-status", Static).update(
            f"{self.icons.arrow} Indexing {label} …"
        )
        self.query_one("#index-button", Button).disabled = True
        self._set_busy(f"Indexing {label}…")
        self._index_worker(paths)

    @work(thread=True, exclusive=True, group="index")
    def _index_worker(self, paths: Sequence[str]) -> None:
        snap = self.service.index(paths)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_index, snap)

    def _apply_index(self, snap: IndexSnapshot) -> None:
        from rich.text import Text

        try:
            self.query_one("#index-button", Button).disabled = False
        except Exception:  # noqa: BLE001
            pass
        text = Text()
        if snap.error:
            text.append("Indexing failed\n\n", style="bold #f85149")
            text.append(snap.error, style="#f85149")
            self.query_one("#index-log", Static).update(text)
            self.query_one("#index-status", Static).update("Index failed.")
            self._set_error("Index failed")
            return
        text.append("Index complete\n\n", style="bold #3fb950")

        def row(label: str, value: object, style: str = "#adbac7") -> None:
            text.append(f"{label:<16}", style="#6e7681")
            text.append(f"{value}\n", style=style)

        row("Scanned", snap.scanned)
        row("Added", snap.added, "#3fb950")
        row("Updated", snap.updated)
        row("Unchanged", snap.unchanged)
        row("Deleted", snap.deleted)
        row("Renamed", snap.renamed)
        row("Skipped", snap.skipped)
        row("Failed", snap.failed, "#f85149" if snap.failed else "#adbac7")
        row("Chunks written", snap.chunks_written)
        row("Embeddings", snap.embeddings_written)
        row("Duration", f"{snap.duration_ms:.0f} ms")
        if snap.failures:
            text.append("\nFailures:\n", style="bold #f85149")
            for p, reason, detail in snap.failures[:20]:
                text.append(f"  {p}: {reason} {detail}\n", style="#f85149")
        if snap.skips:
            text.append("\nSkipped:\n", style="bold #d29922")
            for p, reason, detail in snap.skips[:20]:
                text.append(f"  {p}: {reason} {detail}\n", style="#d29922")
        self.query_one("#index-log", Static).update(text)
        self.query_one("#index-status", Static).update(
            f"Done: +{snap.added} ~{snap.updated} -{snap.deleted} in {snap.duration_ms:.0f} ms"
        )
        self._set_ready(f"Indexed: +{snap.added} ~{snap.updated} -{snap.deleted}")
        # Show the new counts in the top bar and sidebar.
        self._refresh_top()

    # ------------------------------------------------------------- evidence
    @on(Input.Submitted, "#evidence-input")
    def _evidence_submitted(self, event: Input.Submitted) -> None:
        chunk_id = event.value.strip()
        if not chunk_id:
            return
        self.query_one("#evidence-block", Static).update("Walking the evidence graph…")
        self._set_busy("Exploring evidence…")
        self._evidence_worker(chunk_id)

    @work(thread=True, exclusive=True, group="evidence-view")
    def _evidence_worker(self, chunk_id: str) -> None:
        snap = self.service.evidence(chunk_id, max_depth=2)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_evidence_view, snap)

    def _apply_evidence_view(self, snap: EvidenceSnapshot) -> None:
        from rich.text import Text

        text = Text()
        if snap.error:
            text.append(snap.error, style="#f85149")
            self.query_one("#evidence-block", Static).update(text)
            self._set_error(snap.error[:60])
            return
        text.append("chunk ", style="#6e7681")
        text.append(f"{snap.chunk_id}", style="bold #e6edf3")
        text.append("   found " if snap.found else "   not found ",
                     style="#3fb950" if snap.found else "#d29922")
        text.append("\n\n")
        text.append(f"{snap.step_count} edge(s), {snap.visited_count} node(s) visited\n", style="#adbac7")
        if snap.counts_by_kind:
            parts = ", ".join(f"{k}: {v}" for k, v in sorted(snap.counts_by_kind.items()))
            text.append(f"by kind — {parts}\n", style="#6e7681")
        text.append("\nRelated passages:\n", style="bold #adbac7")
        if not snap.related:
            text.append("  none (no outgoing evidence edges)\n", style="#6e7681")
        for rel in snap.related:
            kind_style = "#3fb950" if rel.kind == "explicit" else "#d29922"
            # Keep the bullet, path, line range and relation on one line.
            text.append(f"  {self.icons.bullet} ", style="#3fb950")
            text.append(rel.path, style="#58a6ff")
            text.append(f"  L{rel.line_range}  ", style="#6e7681")
            text.append(rel.relation, style="#8b949e")
            text.append(" [", style="#30363d")
            text.append(rel.kind, style=kind_style)
            text.append("]\n", style="#30363d")
        self.query_one("#evidence-block", Static).update(text)
        self._set_ready(f"{snap.step_count} edge(s) from {snap.chunk_id}")

    # ------------------------------------------------------------- validate
    @on(Input.Submitted, "#validate-input")
    def _validate_submitted(self, event: Input.Submitted) -> None:
        chunk_id = event.value.strip()
        if not chunk_id:
            return
        self.query_one("#validate-block", Static).update("Re-reading source from disk…")
        self._set_busy("Validating…")
        self._validate_worker(chunk_id)

    @work(thread=True, exclusive=True, group="validate")
    def _validate_worker(self, chunk_id: str) -> None:
        snap = self.service.validate(chunk_id)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_validate, snap)

    def _apply_validate(self, snap: ValidationSnapshot) -> None:
        from rich.text import Text

        text = Text()
        colour = {
            "valid": "#3fb950",
            "content_changed": "#d29922",
            "path_missing": "#f85149",
            "chunk_out_of_range": "#f85149",
            "document_deleted": "#f85149",
            "not_validated": "#6e7681",
        }.get(snap.state, "#d29922")
        glyph = {
            "valid": self.icons.valid,
            "content_changed": self.icons.changed,
        }.get(snap.state, self.icons.missing if not snap.found else self.icons.warn)

        text.append("chunk_id   ", style="#6e7681")
        text.append(f"{snap.chunk_id}\n", style="#e6edf3")
        if snap.source_path:
            text.append("source     ", style="#6e7681")
            text.append(f"{snap.source_path}:{snap.line_range}\n", style="#adbac7")
        # State line: icon + the state WORD (so it never rides on icon or colour
        # alone) + a plain-language outcome.
        outcome = "citation resolves" if snap.is_valid else "citation does NOT resolve"
        text.append("state      ", style="#6e7681")
        text.append(f"{glyph} {snap.state}", style=f"bold {colour}")
        text.append(f"  ({outcome})\n", style=colour)
        if snap.detail:
            text.append("detail     ", style="#6e7681")
            text.append(f"{snap.detail}\n", style="#adbac7")
        if snap.current_hash:
            text.append("hash       ", style="#6e7681")
            text.append(f"{snap.current_hash}\n", style="#8b949e")
        if snap.current_revision is not None:
            text.append("revision   ", style="#6e7681")
            text.append(f"{snap.current_revision}\n", style="#adbac7")
        self.query_one("#validate-block", Static).update(text)
        if snap.error:
            self._set_error(snap.error[:60])
        else:
            self._set_ready(f"Validation: {snap.state}")

    # ------------------------------------------------------------- conflicts
    @work(thread=True, exclusive=True, group="conflicts")
    def _conflicts_worker(self) -> None:
        scan = self.service.conflicts(limit=100)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_conflicts, scan)

    def _apply_conflicts(self, scan: ConflictScanSnapshot) -> None:
        from rich.text import Text

        rows = scan.rows
        text = Text()
        if scan.error:
            text.append("Conflict scan failed\n\n", style="bold #f85149")
            text.append(scan.error, style="#f85149")
            self.query_one("#conflicts-list", Static).update(text)
            self._set_error("Conflict scan failed")
            return
        # Scan coverage: find_conflicts caps the scan at scan_limit chunks, so
        # report how much of the corpus this pass actually covered.
        if scan.partial:
            text.append(
                f"scanned {scan.scanned:,} of {scan.total:,} chunks "
                "(scan is capped, so some candidates may be missed)\n\n",
                style="#d29922",
            )
        else:
            text.append(f"scanned {scan.scanned:,} of {scan.total:,} chunks\n\n", style="#6e7681")
        if not rows:
            text.append("No conflict candidates found.\n\n", style="#3fb950")
            text.append(
                "Conflict detection is a heuristic over the current index "
                "(same attribute, different number; near-duplicate with one side negated).",
                style="#6e7681",
            )
            self.query_one("#conflicts-list", Static).update(text)
            self._set_ready("No conflicts")
            return
        text.append(f"{len(rows)} candidate(s) to check (leads, not findings).\n\n", style="bold #d29922")
        for c in rows:
            text.append(f"[{c.kind}] ", style="#d29922")
            text.append(f"{c.rule}", style="bold #e6edf3")
            text.append(f"  score={c.score:.3f}\n", style="#6e7681")
            text.append("   A ", style="#6e7681")
            text.append(f"{c.path_a}:{c.line_range_a}", style="#58a6ff")
            text.append(f"  ({c.chunk_id_a})\n", style="#484f58")
            text.append("   B ", style="#6e7681")
            text.append(f"{c.path_b}:{c.line_range_b}", style="#58a6ff")
            text.append(f"  ({c.chunk_id_b})\n", style="#484f58")
            if c.detail:
                text.append(f"   {c.detail}\n", style="#8b949e")
            text.append("\n")
        self.query_one("#conflicts-list", Static).update(text)
        self._set_ready(f"{len(rows)} conflict candidate(s)")

    # ------------------------------------------------------------------- mcp
    @work(thread=True, exclusive=True, group="mcp")
    def _mcp_worker(self) -> None:
        snap = self.service.mcp_info()
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_mcp, snap)

    def _apply_mcp(self, snap: McpSnapshot) -> None:
        from rich.text import Text

        text = Text()
        if snap.sdk_available:
            text.append(f"{self.icons.ok} MCP SDK available\n", style="#3fb950")
        else:
            text.append(f"{self.icons.warn} MCP SDK not installed\n", style="#d29922")
        text.append(f"{snap.detail}\n\n", style="#6e7681")
        text.append("Launch the server (separate process):\n", style="#adbac7")
        text.append(f"  {snap.launch_command}\n\n", style="bold #e6edf3")
        text.append("Tools exposed:\n", style="bold #adbac7")
        for name, summary in snap.tools:
            text.append(f"  {name}\n", style="#58a6ff")
            if summary:
                text.append(f"      {summary}\n", style="#6e7681")
        self.query_one("#mcp-block", Static).update(text)
        self._set_ready("MCP info")

    # -------------------------------------------------------------- settings
    @work(thread=True, exclusive=True, group="settings")
    def _settings_worker(self) -> None:
        status = self.service.status(with_freshness=True)
        rows = self.service.config_rows()
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_settings, status, rows)

    def _apply_settings(self, status: StatusSnapshot, rows: list[tuple[str, str]]) -> None:
        from rich.text import Text

        text = Text()
        text.append("Configuration\n", style="bold #e6edf3")
        for label, value in rows:
            text.append(f"  {label:<22}", style="#6e7681")
            text.append(f"{value}\n", style="#adbac7")
        text.append("\nIndex status\n", style="bold #e6edf3")
        if not status.initialized:
            text.append("  index not initialized\n", style="#d29922")
            if status.error:
                text.append(f"  {status.error}\n", style="#f85149")
        else:
            def row(label: str, value: object) -> None:
                text.append(f"  {label:<22}", style="#6e7681")
                text.append(f"{value}\n", style="#adbac7")

            row("Schema version", status.schema_version)
            row("Documents (current)", status.documents_current)
            row("Documents (deleted)", status.documents_deleted)
            row("Chunks", status.chunks)
            row("Embeddings", status.embeddings)
            row("Embedding provider", status.embedding_provider)
            row("Embedding dim", status.embedding_dim)
            if status.languages:
                langs = ", ".join(f"{k}: {v}" for k, v in sorted(status.languages.items()))
                row("Languages", langs)
            if status.freshness:
                fresh = ", ".join(f"{k}: {v}" for k, v in sorted(status.freshness.items()))
                row("Freshness", fresh)
        self.query_one("#settings-block", Static).update(text)
        self._set_ready("Status loaded")

    # --------------------------------------------------------------- actions
    def action_focus_search(self) -> None:
        if self._active_view != "view-search":
            self.switch_view("view-search")
        self.set_focus(self.query_one("#search-input", Input))

    def action_escape(self) -> None:
        focused = self.focused
        if isinstance(focused, Input):
            self.set_focus(None)
            return
        if self._active_view != "view-search":
            self.switch_view("view-search")

    def action_refresh_view(self) -> None:
        if self._active_view == "view-search":
            # On Search, `r` re-reads the visible hits from disk and re-validates
            # them, rather than re-running retrieval (which wouldn't pick up a
            # source edit in the badges). If nothing is shown, fall through to a
            # plain refresh.
            if self._last_search is not None and self._last_search.hits:
                self._revalidate_visible()
                return
        self._on_enter_view(self._active_view)
        if self._active_view == "view-search" and self._last_search is not None:
            self._run_search(self._last_search.query)
        self._refresh_top()

    def action_explore_selected(self) -> None:
        """`e` on the Search view: open Explore Evidence for the selected hit.

        Prefills the chunk id from the current selection and runs the lookup, so
        the view opens already populated for that hit. Manual entry still works.
        """
        if self._active_view != "view-search" or not self._selected_chunk_id:
            return
        chunk_id = self._selected_chunk_id
        self.query_one("#evidence-input", Input).value = chunk_id
        self.switch_view("view-evidence")
        self.query_one("#evidence-block", Static).update("Walking the evidence graph…")
        self._set_busy("Exploring evidence…")
        self._evidence_worker(chunk_id)

    def action_validate_selected(self) -> None:
        """`v` on the Search view: open Validate Citations for the selected hit.

        Prefills the chunk id from the current selection and runs validation, so
        the view opens already populated for that hit. Manual entry still works.
        """
        if self._active_view != "view-search" or not self._selected_chunk_id:
            return
        chunk_id = self._selected_chunk_id
        self.query_one("#validate-input", Input).value = chunk_id
        self.switch_view("view-validate")
        self.query_one("#validate-block", Static).update("Re-reading source from disk…")
        self._set_busy("Validating…")
        self._validate_worker(chunk_id)

    # ------------------------------------------------------- re-validate (r)
    def _revalidate_visible(self) -> None:
        results = self.query_one("#results", ListView)
        chunk_ids = [c.hit.chunk_id for c in results.children if isinstance(c, ResultItem)]
        if not chunk_ids:
            return
        self._set_busy(f"Re-validating {len(chunk_ids)} hit(s)…")
        self._revalidate_worker(tuple(chunk_ids))

    @work(thread=True, exclusive=True, group="revalidate")
    def _revalidate_worker(self, chunk_ids: Sequence[str]) -> None:
        results: dict[str, str] = {}
        for cid in chunk_ids:
            snap = self.service.validate(cid)
            results[cid] = snap.state
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.call_from_thread(self._apply_revalidation, results)

    def _apply_revalidation(self, states: dict[str, str]) -> None:
        results = self.query_one("#results", ListView)
        updated = 0
        selected_hit = None
        for item in results.children:
            if not isinstance(item, ResultItem):
                continue
            new_state = states.get(item.hit.chunk_id)
            if new_state is None or new_state == item.hit.validation:
                if item.hit.chunk_id == self._selected_chunk_id:
                    selected_hit = item.hit
                continue
            # HitSnapshot is frozen, so rebuild it with the fresh validation
            # state and re-render the card badge in place.
            fresh = replace(item.hit, validation=new_state)
            item.hit = fresh
            item.refresh_card()
            updated += 1
            if fresh.chunk_id == self._selected_chunk_id:
                selected_hit = fresh
        # Keep the detail panel's badge matching the selected row.
        if selected_hit is not None:
            self.query_one("#details", SourceDetails).show_hit(selected_hit)
            self._related_worker(selected_hit.chunk_id)
        self._set_ready(f"Re-validated {len(states)} hit(s); {updated} changed")

    def action_request_quit(self) -> None:
        self.exit()

    # ------------------------------------------------------------- responsive
    def on_resize(self, event) -> None:
        self._apply_responsive(event.size.width)

    def _apply_responsive(self, width: int) -> None:
        screen = self.screen
        screen.remove_class("narrow")
        screen.remove_class("compact")
        if width < 70:
            screen.add_class("narrow")
        elif width < 100:
            screen.add_class("compact")


def run(config: OrbitConfig, *, default_index_path: str = "", icon_set: str | None = None) -> None:
    """Launch the Orbit TUI against ``config``. Blocks until the user exits."""
    OrbitTUI(config, default_index_path=default_index_path, icon_set=icon_set).run()
