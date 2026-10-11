"""Presentational widgets for the Orbit TUI.

These widgets render snapshots from :mod:`orbit.tui.service`. They hold no
database handle and do no I/O; the :class:`~orbit.tui.app.OrbitTUI` app drives
them. Dynamic text (file paths, code snippets, validation detail) goes through
Rich ``Text`` objects instead of markup strings, so a bracket in a path or a
code sample won't be misread as console markup.
"""

from __future__ import annotations

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, ListItem, Static

from .icons import DEFAULT_ICON_SET, IconSet, get_icon_set
from .service import (
    EvidenceSnapshot,
    HitSnapshot,
    TopInfo,
)

__all__ = [
    "TopBar",
    "StatusBar",
    "Sidebar",
    "ResultItem",
    "SourceDetails",
    "NAV_ITEMS",
    "lexer_for_language",
]


def _icons_for(widget) -> IconSet:
    """Return the app's active icon set, falling back to the default.

    Widgets read the active set from the running app so one ``--icons`` choice
    drives every glyph. During early mount (or in isolated widget tests) the app
    may not expose one yet, so fall back to the default set.
    """
    app = getattr(widget, "app", None)
    icons = getattr(app, "icons", None)
    if isinstance(icons, IconSet):
        return icons
    return get_icon_set(DEFAULT_ICON_SET)


# --------------------------------------------------------------------------
# Navigation model. (key, view_id, label, icon_name)
#
# Order and numbering follow the reference layout: Index is 1, Search is 2,
# Exit is 0. The number doubles as a key binding handled by the app. The fourth
# field is a semantic icon name (see orbit.tui.icons) that the active icon set
# resolves to a glyph, rather than a hardcoded symbol.
# --------------------------------------------------------------------------
NAV_ITEMS: tuple[tuple[str, str, str, str], ...] = (
    ("1", "view-index", "Index Repository", "index"),
    ("2", "view-search", "Search Knowledge", "search"),
    ("3", "view-evidence", "Explore Evidence", "evidence"),
    ("4", "view-validate", "Validate Citations", "validate"),
    ("5", "view-conflicts", "Detect Conflicts", "conflict"),
    ("6", "view-mcp", "MCP Server", "mcp"),
    ("7", "view-settings", "Settings", "settings"),
    ("0", "view-exit", "Exit", "exit"),
)

# pygments lexer names keyed by Orbit's detected-language label. Anything not
# listed falls back to plain text (see :func:`lexer_for_language`).
_LEXER_BY_LANGUAGE: dict[str, str] = {
    "python": "python",
    "java": "java",
    "kotlin": "kotlin",
    "c": "c",
    "cpp": "cpp",
    "csharp": "csharp",
    "go": "go",
    "rust": "rust",
    "javascript": "javascript",
    "typescript": "typescript",
    "ruby": "ruby",
    "php": "php",
    "shell": "bash",
    "powershell": "powershell",
    "sql": "sql",
    "json": "json",
    "xml": "xml",
    "html": "html",
    "yaml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "css": "css",
    "scss": "scss",
    "markdown": "markdown",
    "restructuredtext": "rst",
    "make": "make",
    "dockerfile": "docker",
    "gradle": "groovy",
    "protobuf": "protobuf",
    "text": "text",
}


def lexer_for_language(language: str) -> str:
    """Map an Orbit language label to a pygments lexer name, defaulting to text."""
    return _LEXER_BY_LANGUAGE.get((language or "").lower(), "text")


# --------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------
class TopBar(Static):
    """The window title bar: product identity on the left, index facts right."""

    def update_info(self, info: TopInfo | None, *, status: str = "") -> None:
        icons = _icons_for(self)
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(f"{icons.logo} orbit", style="bold #3fb950")
        if info is not None:
            text.append(f"  v{info.version}", style="#6e7681")
        text.append("  —  Knowledge Engine", style="#6e7681")
        if info is not None:
            text.append("     ")
            text.append("Database: ", style="#6e7681")
            text.append(info.db_name, style="bold #e6edf3")
            text.append("   ·   ", style="#30363d")
            text.append("Repository: ", style="#6e7681")
            text.append(info.repository, style="#adbac7")
            text.append("   ·   ", style="#30363d")
            text.append(f"{icons.ok} ", style="#3fb950")
            text.append(f"{info.file_count:,} files indexed", style="#adbac7")
            if not info.initialized:
                text.append("   ·   ", style="#30363d")
                text.append("not initialized", style="bold #d29922")
        self.update(text)


# --------------------------------------------------------------------------
# Bottom status bar (custom footer: context hints left, status dot right)
# --------------------------------------------------------------------------
class StatusBar(Static):
    """Context-sensitive keyboard hints plus a live operation/status indicator."""

    #: (kind -> (dot colour, label colour))
    _KIND_STYLE = {
        "ready": ("#3fb950", "#adbac7"),
        "busy": ("#d29922", "#d29922"),
        "error": ("#f85149", "#f85149"),
    }

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._hints: tuple[tuple[str, str], ...] = ()
        self._status = "Ready"
        self._kind = "ready"

    def set_hints(self, hints: tuple[tuple[str, str], ...]) -> None:
        self._hints = hints
        self._render()

    def set_status(self, message: str, *, kind: str = "ready") -> None:
        self._status = message or ""
        self._kind = kind if kind in self._KIND_STYLE else "ready"
        self._render()

    def _render(self) -> None:
        icons = _icons_for(self)
        left = Text(no_wrap=True, overflow="ellipsis")
        for i, (key, label) in enumerate(self._hints):
            if i:
                left.append("   ")
            left.append(f" {key} ", style="bold #0a0e14 on #3fb950")
            left.append(f" {label}", style="#6e7681")
        dot_colour, label_colour = self._KIND_STYLE[self._kind]
        right = Text()
        right.append(f"{icons.bullet} ", style=dot_colour)
        right.append(self._status, style=label_colour)
        # Pad so the status sits on the right edge; Static wraps a single line.
        width = max(self.size.width, 0)
        pad = width - left.cell_len - right.cell_len
        if pad < 1:
            pad = 1
        combined = Text.assemble(left, " " * pad, right, no_wrap=True, overflow="ellipsis")
        self.update(combined)

    def on_resize(self) -> None:  # re-pad when the terminal width changes
        self._render()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
class Sidebar(Vertical):
    """Left navigation column: logo, numbered menu, index facts, tagline."""

    def compose(self) -> ComposeResult:
        icons = _icons_for(self)
        logo = Text()
        logo.append(f"{icons.logo}  ORBIT\n", style="bold #3fb950")
        logo.append("Knowledge Engine", style="#6e7681")
        yield Static(logo, id="logo")
        # The app builds the nav list; it owns selection and routing.
        yield Static("", id="info-box")
        quote = Text()
        quote.append('"Evidence over assumption."\n', style="italic #6e7681")
        quote.append("                      — ORBIT", style="#484f58")
        yield Static(quote, id="quote")

    def update_info_box(self, info: TopInfo | None) -> None:
        box = Text()
        if info is None:
            box.append("Current DB   —\n", style="#6e7681")
            self.query_one("#info-box", Static).update(box)
            return

        def row(label: str, value: str) -> None:
            box.append(f"{label:<13}", style="#6e7681")
            box.append(f"{value}\n", style="#adbac7")

        row("Current DB", info.db_name)
        row("Repository", info.repository)
        row("Indexed", f"{info.file_count:,}")
        row("Embedder", info.embedding_provider)
        box.append(f"{'Last Indexed':<13}", style="#6e7681")
        box.append(info.last_indexed, style="#adbac7")
        self.query_one("#info-box", Static).update(box)


# --------------------------------------------------------------------------
# A single result card inside the results ListView
# --------------------------------------------------------------------------
class ResultItem(ListItem):
    """A ListView row carrying its :class:`HitSnapshot` and rendering a card."""

    def __init__(self, hit: HitSnapshot) -> None:
        super().__init__()
        self.hit = hit

    def compose(self) -> ComposeResult:
        yield Static(self._card_text(self.hit, _icons_for(self)), classes="result-card")

    def refresh_card(self) -> None:
        """Re-render this row's card from the current ``self.hit``.

        Called after a re-validation pass swaps in a HitSnapshot with a new
        validation state, so the badge updates in place without rebuilding the
        whole results list.
        """
        try:
            self.query_one(".result-card", Static).update(
                self._card_text(self.hit, _icons_for(self))
            )
        except Exception:  # noqa: BLE001 - not yet mounted; nothing to refresh
            pass

    @staticmethod
    def _card_text(hit: HitSnapshot, icons: IconSet) -> Text:
        text = Text()
        # Header: rank + filename, then the fused score (no relevance %).
        text.append(f"{hit.rank}. ", style="#6e7681")
        text.append(hit.filename, style="bold #e6edf3")
        text.append(f"   score {hit.score:.5f}", style="bold #3fb950")
        if hit.lexical_score is not None:
            text.append(f"  lex {hit.lexical_score:.5f}", style="#6e7681")
        if hit.vector_score is not None:
            text.append(f"  vec {hit.vector_score:.5f}", style="#6e7681")
        text.append("\n")
        text.append(hit.path, style="#4a9960")
        text.append("\n")
        validation_style = "#3fb950" if hit.validation == "valid" else "#d29922"
        text.append(f"L{hit.start_line}–L{hit.end_line}", style="#6e7681")
        text.append(f"  {icons.bullet}  ", style="#30363d")
        text.append(hit.retrieval_method, style="#6e7681")
        text.append(f"  {icons.bullet}  ", style="#30363d")
        text.append(hit.validation, style=validation_style)
        text.append("\n")
        # Snippet: line breaks are kept (the service already capped it at 3 lines).
        text.append(hit.snippet, style="#8b949e")
        return text


# --------------------------------------------------------------------------
# Right panel: source details
# --------------------------------------------------------------------------
class SourceDetails(VerticalScroll):
    """The right column: file info, syntax-highlighted passage, context, related."""

    BORDER_TITLE = "Source Details"

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder(), id="detail-fileinfo")
        with VerticalScroll(id="detail-code-wrap"):
            yield Static("", id="detail-code")
        with Collapsible(title="Context", collapsed=False, id="detail-context-box"):
            yield Static("", id="detail-context")
        with Collapsible(title="Related Files", collapsed=False, id="detail-related-box"):
            yield Static("", id="detail-related")

    @staticmethod
    def _placeholder() -> Text:
        t = Text()
        t.append("No result selected.\n\n", style="bold #6e7681")
        t.append("Run a search and highlight a result to inspect its ", style="#6e7681")
        t.append("source passage, provenance and related files.", style="#6e7681")
        return t

    def clear(self) -> None:
        self.query_one("#detail-fileinfo", Static).update(self._placeholder())
        self.query_one("#detail-code", Static).update("")
        self.query_one("#detail-context", Static).update("")
        self.query_one("#detail-related", Static).update(
            Text("—", style="#6e7681")
        )

    def show_hit(self, hit: HitSnapshot) -> None:
        icons = _icons_for(self)
        # File-info header.
        info = Text()
        info.append(f"{icons.file} ", style="#3fb950")
        info.append(hit.path, style="bold #adbac7")
        info.append("\n")
        info.append(f"L{hit.start_line}–L{hit.end_line}", style="#6e7681")
        info.append(f"   {icons.bullet}   ", style="#30363d")
        info.append(f"score {hit.score:.5f}", style="#3fb950")
        info.append(f"   {icons.bullet}   ", style="#30363d")
        vstyle = "#3fb950" if hit.validation == "valid" else "#d29922"
        info.append(hit.validation, style=vstyle)
        self.query_one("#detail-fileinfo", Static).update(info)

        # Syntax-highlighted passage with absolute line numbers.
        self.query_one("#detail-code", Static).update(self._syntax(hit))

        # Context: the provenance metadata that makes the citation checkable.
        ctx = Text()
        ctx.append("chunk_id   ", style="#6e7681")
        ctx.append(f"{hit.chunk_id}\n", style="#adbac7")
        ctx.append("doc_id     ", style="#6e7681")
        ctx.append(f"{hit.doc_id}\n", style="#adbac7")
        ctx.append("language   ", style="#6e7681")
        ctx.append(f"{hit.language or '—'}\n", style="#adbac7")
        ctx.append("score      ", style="#6e7681")
        ctx.append(f"{hit.score:.5f}\n", style="#adbac7")
        if hit.lexical_score is not None:
            ctx.append("lexical    ", style="#6e7681")
            ctx.append(f"{hit.lexical_score:.5f}\n", style="#adbac7")
        if hit.vector_score is not None:
            ctx.append("vector     ", style="#6e7681")
            ctx.append(f"{hit.vector_score:.5f}\n", style="#adbac7")
        ctx.append("validation ", style="#6e7681")
        ctx.append(hit.validation, style=vstyle)
        self.query_one("#detail-context", Static).update(ctx)

        # Related files fill in asynchronously; show a loading note until then.
        self.query_one("#detail-related", Static).update(
            Text("loading related files…", style="italic #6e7681")
        )

    def _syntax(self, hit: HitSnapshot) -> Syntax | Text:
        lexer = lexer_for_language(hit.language)
        try:
            return Syntax(
                hit.text,
                lexer,
                line_numbers=True,
                start_line=hit.start_line,
                theme="github-dark",
                word_wrap=True,
                indent_guides=False,
            )
        except Exception:  # noqa: BLE001 - unknown lexer: fall back to plain text
            return Text(hit.text, style="#adbac7")

    def set_related(self, evidence: EvidenceSnapshot) -> None:
        icons = _icons_for(self)
        text = Text()
        if evidence.error:
            text.append(evidence.error, style="#d29922")
            self.query_one("#detail-related", Static).update(text)
            return
        if not evidence.related:
            text.append("No related passages.\n", style="#6e7681")
            text.append(
                "Related files come from Orbit's evidence graph "
                "(same-document, shared-symbol and link edges).",
                style="#484f58",
            )
            self.query_one("#detail-related", Static).update(text)
            return
        for rel in evidence.related:
            kind_style = "#3fb950" if rel.kind == "explicit" else "#d29922"
            name = rel.path.replace("\\", "/").rsplit("/", 1)[-1]
            text.append(f"{icons.bullet} ", style="#3fb950")
            text.append(name, style="underline #58a6ff")
            text.append(f"  L{rel.line_range}", style="#6e7681")
            text.append("\n   ")
            text.append(rel.relation, style="#8b949e")
            text.append("  [", style="#30363d")
            text.append(rel.kind, style=kind_style)
            text.append("]\n", style="#30363d")
        self.query_one("#detail-related", Static).update(text)
