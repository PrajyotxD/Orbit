"""Thread-safe data-access layer for the Orbit TUI.

Every method opens its OWN short-lived :class:`~orbit.engine.OrbitEngine` in the
calling thread and returns plain, immutable snapshots. That's on purpose.

:class:`~orbit.storage.db.OrbitDB` wraps a *thread-affine* SQLite connection: a
connection may only be used from the thread that created it. The TUI runs every
blocking operation (search, indexing, validation, conflict scans) on a Textual
worker thread, so a single shared engine would raise

    SQLite objects created in a thread can only be used in that same thread

the moment a worker touched it. Opening a fresh engine inside each worker steps
around that; WAL mode (configured by ``OrbitDB``) keeps concurrent readers and a
single writer safe on the same file.

The snapshots returned here carry no live database handle, so handing one back
to the UI thread is safe. The UI-facing formatting (score formatting, category
buckets, human timestamps) happens here too, which keeps the widgets dumb and
lets this whole layer be unit-tested without a terminal.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ..config import OrbitConfig
from ..errors import OrbitError

__all__ = [
    "OrbitService",
    "TopInfo",
    "HitSnapshot",
    "SearchSnapshot",
    "RelatedSnapshot",
    "EvidenceSnapshot",
    "IndexSnapshot",
    "ValidationSnapshot",
    "ConflictSnapshot",
    "ConflictScanSnapshot",
    "StatusSnapshot",
    "McpSnapshot",
    "CATEGORIES",
    "category_for_language",
]

# --------------------------------------------------------------------------
# Filter categories.
#
# Orbit's index records a language per document (see
# orbit.config.LANGUAGE_BY_EXTENSION); it has no concept of a "comment" span,
# so there's no honest "comments" bucket to offer. The filter tabs bucket only
# the real languages the index can actually tell apart.
# --------------------------------------------------------------------------
_CODE_LANGS = frozenset(
    {
        "python", "java", "kotlin", "c", "cpp", "csharp", "go", "rust",
        "javascript", "typescript", "ruby", "php", "shell", "powershell",
        "sql", "protobuf", "gradle", "make", "dockerfile",
    }
)
_DOC_LANGS = frozenset({"markdown", "restructuredtext", "text", "html"})
_CONFIG_LANGS = frozenset({"json", "xml", "yaml", "toml", "ini", "css", "scss"})

#: Tab order shown in the center panel. The value is the set of languages that
#: belong to the bucket; an empty set means "match everything" (the ``all`` tab).
CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("all", frozenset()),
    ("code", _CODE_LANGS),
    ("docs", _DOC_LANGS),
    ("config", _CONFIG_LANGS),
)


def category_for_language(language: str) -> str:
    """Return the primary category bucket for a detected ``language``."""
    lang = (language or "").lower()
    if lang in _CODE_LANGS:
        return "code"
    if lang in _DOC_LANGS:
        return "docs"
    if lang in _CONFIG_LANGS:
        return "config"
    return "other"


# --------------------------------------------------------------------------
# Snapshots (plain, immutable, no DB handle).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TopInfo:
    version: str
    db_name: str
    db_path: str
    repository: str
    file_count: int
    embedding_provider: str
    last_indexed: str
    initialized: bool


@dataclass(frozen=True)
class HitSnapshot:
    rank: int
    chunk_id: str
    doc_id: str
    path: str
    start_line: int
    end_line: int
    line_range: str
    score: float
    lexical_score: float | None
    vector_score: float | None
    validation: str
    language: str
    category: str
    snippet: str
    text: str
    retrieval_method: str

    @property
    def filename(self) -> str:
        return self.path.replace("\\", "/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class SearchSnapshot:
    query: str
    hits: tuple[HitSnapshot, ...]
    duration_ms: float
    channels_used: tuple[str, ...]
    retrieval_method: str
    total_candidates: int
    degraded_reason: str
    error: str = ""


@dataclass(frozen=True)
class RelatedSnapshot:
    chunk_id: str
    path: str
    line_range: str
    relation: str
    kind: str


@dataclass(frozen=True)
class EvidenceSnapshot:
    chunk_id: str
    found: bool
    step_count: int
    visited_count: int
    counts_by_kind: dict[str, int]
    related: tuple[RelatedSnapshot, ...]
    error: str = ""


@dataclass(frozen=True)
class IndexSnapshot:
    paths: tuple[str, ...]
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    renamed: int = 0
    failed: int = 0
    skipped: int = 0
    chunks_written: int = 0
    embeddings_written: int = 0
    duration_ms: float = 0.0
    failures: tuple[tuple[str, str, str], ...] = ()
    skips: tuple[tuple[str, str, str], ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.failed == 0


@dataclass(frozen=True)
class ValidationSnapshot:
    chunk_id: str
    found: bool
    state: str
    is_valid: bool
    detail: str
    current_hash: str | None
    current_revision: int | None
    source_path: str = ""
    line_range: str = ""
    error: str = ""


@dataclass(frozen=True)
class ConflictSnapshot:
    rule: str
    score: float
    kind: str
    path_a: str
    line_range_a: str
    chunk_id_a: str
    path_b: str
    line_range_b: str
    chunk_id_b: str
    detail: str


@dataclass(frozen=True)
class ConflictScanSnapshot:
    """Conflict candidates plus how much of the index actually got scanned.

    ``find_conflicts`` loads at most ``scan_limit`` chunks (``LIMIT ?`` in
    :func:`orbit.evidence.freshness._load_chunks`), so when the index holds more
    current chunks than that, the scan only covers part of the corpus.
    ``scanned`` and ``total`` let the view say "scanned N of M chunks" rather
    than imply it looked at everything.
    """

    rows: tuple[ConflictSnapshot, ...]
    scanned: int
    total: int
    error: str = ""

    @property
    def partial(self) -> bool:
        return self.total > self.scanned


@dataclass(frozen=True)
class StatusSnapshot:
    initialized: bool
    schema_version: int
    documents_current: int
    documents_deleted: int
    chunks: int
    embeddings: int
    embedding_provider: str
    embedding_dim: int
    db_path: str
    languages: dict[str, int]
    freshness: dict[str, int] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class McpSnapshot:
    sdk_available: bool
    detail: str
    tools: tuple[tuple[str, str], ...]
    launch_command: str


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
class OrbitService:
    """Open-per-call, snapshot-returning data layer over an Orbit index.

    Construct with an :class:`OrbitConfig`. Every public method is safe to call
    from any thread because it never shares an engine or connection across the
    call boundary.
    """

    def __init__(self, config: OrbitConfig) -> None:
        self.config = config

    # ----------------------------------------------------------- engine access
    @contextmanager
    def _engine(self) -> Iterator["object"]:
        # Imported lazily so importing the service doesn't pull in the full
        # engine stack until an operation actually runs.
        from ..engine import OrbitEngine

        eng = OrbitEngine(self.config)
        try:
            yield eng
        finally:
            try:
                eng.close()
            except Exception:
                pass

    @property
    def db_path(self) -> str:
        return self.config.db_path

    @property
    def db_name(self) -> str:
        return Path(self.config.db_path).name

    # --------------------------------------------------------------- db switch
    def use_db(self, db_path: str) -> None:
        """Repoint every future operation at ``db_path``.

        The service holds no open connection between calls (each method opens
        its own short-lived engine), so switching the active DB is just swapping
        the stored config. Nothing else has to be torn down.
        """
        db_path = (db_path or "").strip()
        if db_path:
            self.config = self.config.with_db_path(db_path)

    def discover_databases(self) -> tuple[tuple[str, str], ...]:
        """Find Orbit DB files near the active one, newest first.

        Returns ``(display_name, absolute_path)`` pairs for every ``*.db`` in the
        active DB's directory, so the UI can offer them in a switcher. The active
        DB is always included even if it doesn't exist on disk yet.
        """
        active = Path(self.config.db_path)
        directory = active.parent if str(active.parent) else Path(".")
        found: dict[str, float] = {}
        try:
            for p in directory.glob("*.db"):
                try:
                    found[str(p.resolve())] = p.stat().st_mtime
                except OSError:
                    found[str(p.resolve())] = 0.0
        except OSError:
            pass
        try:
            found.setdefault(str(active.resolve()), active.stat().st_mtime if active.exists() else 0.0)
        except OSError:
            found.setdefault(str(active), 0.0)
        ordered = sorted(found.items(), key=lambda kv: kv[1], reverse=True)
        return tuple((Path(path).name, path) for path, _mtime in ordered)

    @staticmethod
    def suggest_db_for_index_path(index_path: str) -> str:
        """Suggest a per-repo DB path for an index target.

        A folder maps to ``.orbit/<folder-name>.db`` so each repo lands in its
        own DB by default instead of everything piling into one shared index.
        Returns an empty string when no sensible name can be derived (so the
        caller keeps whatever DB is already selected).
        """
        raw = (index_path or "").strip().split()
        if not raw:
            return ""
        first = Path(raw[0]).expanduser()
        name = first.name or first.anchor.strip("\\/").replace(":", "")
        if not name:
            return ""
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
        return str(Path(".orbit") / f"{safe}.db")

    def live_counts(self) -> tuple[int, int]:
        """Current (documents, chunks) in the active DB; (0, 0) if not ready.

        Cheap enough to poll on a timer while an index run writes: WAL lets this
        reader see each per-document transaction as the writer commits it, so a
        caller can show real progress without the engine emitting callbacks.
        """
        try:
            with self._engine() as eng:
                if not self._is_initialized(eng):
                    return (0, 0)
                docs = int(eng.db.scalar("SELECT count(*) FROM documents WHERE status='current'") or 0)  # type: ignore[attr-defined]
                chunks = int(eng.db.scalar("SELECT count(*) FROM chunks") or 0)  # type: ignore[attr-defined]
                return (docs, chunks)
        except (OrbitError, sqlite3.Error):
            return (0, 0)

    # ------------------------------------------------------------------- probes
    def _is_initialized(self, eng: "object") -> bool:
        """True when the schema exists (an index has been created)."""
        try:
            return eng.db.schema_version() > 0  # type: ignore[attr-defined]
        except (OrbitError, sqlite3.Error):
            return False

    # ----------------------------------------------------------------- top bar
    def top_info(self) -> TopInfo:
        version = _orbit_version()
        with self._engine() as eng:
            if not self._is_initialized(eng):
                return TopInfo(
                    version=version,
                    db_name=self.db_name,
                    db_path=str(self.config.db_path),
                    repository="—",
                    file_count=0,
                    embedding_provider="—",
                    last_indexed="never",
                    initialized=False,
                )
            try:
                stats = eng.stats()  # type: ignore[attr-defined]
                repo = self._repository_label(eng)
                last = eng.db.get_meta("updated_at") or eng.db.get_meta("last_sync")  # type: ignore[attr-defined]
            except (OrbitError, sqlite3.Error) as exc:
                return TopInfo(
                    version=version,
                    db_name=self.db_name,
                    db_path=str(self.config.db_path),
                    repository="—",
                    file_count=0,
                    embedding_provider="—",
                    last_indexed="error",
                    initialized=False,
                )
            return TopInfo(
                version=version,
                db_name=self.db_name,
                db_path=str(self.config.db_path),
                repository=repo,
                file_count=stats.documents_current,
                embedding_provider=stats.embedding_provider or "—",
                last_indexed=_format_epoch(last),
                initialized=True,
            )

    def _repository_label(self, eng: "object") -> str:
        """Repository name shown in the top bar: the most recently indexed root.

        One DB can hold more than one repo. Picking the most *frequent* root
        makes the label stick on whichever repo has the most files, so indexing
        a smaller second repo into the same DB leaves the wrong name on screen.
        Ordering by indexed_at and taking the newest root makes the label follow
        the latest index run.
        """
        import json

        try:
            rows = eng.db.query(  # type: ignore[attr-defined]
                "SELECT metadata, indexed_at FROM documents "
                "WHERE status='current' ORDER BY indexed_at DESC, rowid DESC"
            )
        except (OrbitError, sqlite3.Error):
            return "—"
        roots: list[str] = []
        for r in rows:
            try:
                meta = json.loads(r["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            root = meta.get("root")
            if root:
                roots.append(str(root))
        if not roots:
            return "—"
        latest_root = roots[0]
        distinct = len(set(roots))
        name = Path(latest_root).name or latest_root or "—"
        # Flag a mixed DB so the single name isn't mistaken for the whole index.
        if distinct > 1:
            name = f"{name} (+{distinct - 1} more)"
        return name

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        limit: int,
        modes: Sequence[str] = ("lexical", "vector"),
    ) -> SearchSnapshot:
        query = query or ""
        with self._engine() as eng:
            if not self._is_initialized(eng):
                return SearchSnapshot(
                    query=query,
                    hits=(),
                    duration_ms=0.0,
                    channels_used=(),
                    retrieval_method="lexical",
                    total_candidates=0,
                    degraded_reason="index not initialized — run Index Repository first",
                    error="",
                )
            try:
                result = eng.search(query, limit=limit, modes=tuple(modes))  # type: ignore[attr-defined]
            except OrbitError as exc:
                return SearchSnapshot(
                    query=query,
                    hits=(),
                    duration_ms=0.0,
                    channels_used=(),
                    retrieval_method="lexical",
                    total_candidates=0,
                    degraded_reason="",
                    error=str(exc),
                )
            hits: list[HitSnapshot] = []
            for h in result.hits:
                ev = h.evidence
                lang = str(h.metadata.get("language", "") or "")
                hits.append(
                    HitSnapshot(
                        rank=ev.rank,
                        chunk_id=ev.chunk_id,
                        doc_id=ev.doc_id,
                        path=ev.source_path,
                        start_line=ev.start_line,
                        end_line=ev.end_line,
                        line_range=ev.line_range,
                        score=ev.score,
                        lexical_score=ev.lexical_score,
                        vector_score=ev.vector_score,
                        validation=_enum_value(ev.validation),
                        language=lang,
                        category=category_for_language(lang),
                        snippet=_snippet_lines(h.text, max_lines=3),
                        text=h.text,
                        retrieval_method=_enum_value(ev.retrieval_method),
                    )
                )
            return SearchSnapshot(
                query=result.query,
                hits=tuple(hits),
                duration_ms=result.duration_ms,
                channels_used=tuple(result.channels_used),
                retrieval_method=_enum_value(result.retrieval_method),
                total_candidates=result.total_candidates,
                degraded_reason=result.degraded_reason,
            )

    # ------------------------------------------------------------ evidence graph
    def evidence(self, chunk_id: str, *, max_depth: int = 1) -> EvidenceSnapshot:
        from ..evidence import traverse

        chunk_id = (chunk_id or "").strip()
        if not chunk_id:
            return EvidenceSnapshot(
                chunk_id="", found=False, step_count=0, visited_count=0,
                counts_by_kind={}, related=(), error="no chunk selected",
            )
        with self._engine() as eng:
            if not self._is_initialized(eng):
                return EvidenceSnapshot(
                    chunk_id=chunk_id, found=False, step_count=0, visited_count=0,
                    counts_by_kind={}, related=(), error="index not initialized",
                )
            try:
                found = eng.get_chunk(chunk_id) is not None  # type: ignore[attr-defined]
                steps, visited = traverse(eng.db, chunk_id, max_depth=max_depth)  # type: ignore[attr-defined]
            except OrbitError as exc:
                return EvidenceSnapshot(
                    chunk_id=chunk_id, found=False, step_count=0, visited_count=0,
                    counts_by_kind={}, related=(), error=str(exc),
                )
            counts: dict[str, int] = {}
            for s in steps:
                k = _enum_value(s.kind)
                counts[k] = counts.get(k, 0) + 1
            # Resolve non-document chunk nodes to a path + line range.
            related = self._resolve_related(eng, steps)
            return EvidenceSnapshot(
                chunk_id=chunk_id,
                found=found,
                step_count=len(steps),
                visited_count=len(visited),
                counts_by_kind=counts,
                related=related,
            )

    def _resolve_related(self, eng: "object", steps: Sequence["object"]) -> tuple[RelatedSnapshot, ...]:
        rel_by_id: dict[str, tuple[str, str]] = {}
        order: list[tuple[str, str, str]] = []  # (chunk_id, relation, kind)
        seen: set[str] = set()
        for s in steps:
            target = s.to_id  # type: ignore[attr-defined]
            if target.startswith("doc:"):
                continue
            if target in seen:
                continue
            seen.add(target)
            rel_by_id[target] = (s.relation, _enum_value(s.kind))  # type: ignore[attr-defined]
            order.append((target, s.relation, _enum_value(s.kind)))  # type: ignore[attr-defined]
        if not order:
            return ()
        ids = [cid for cid, _, _ in order]
        ph = ",".join("?" for _ in ids)
        try:
            rows = eng.db.query(  # type: ignore[attr-defined]
                "SELECT c.chunk_id, d.path, c.start_line, c.end_line "
                "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id "
                f"WHERE c.chunk_id IN ({ph})",
                ids,
            )
        except (OrbitError, sqlite3.Error):
            return ()
        path_lines = {
            r["chunk_id"]: (r["path"], f"{r['start_line']}-{r['end_line']}") for r in rows
        }
        out: list[RelatedSnapshot] = []
        for cid, relation, kind in order:
            path, lines = path_lines.get(cid, ("?", "?"))
            out.append(
                RelatedSnapshot(
                    chunk_id=cid, path=path, line_range=lines, relation=relation, kind=kind
                )
            )
        return tuple(out)

    # ------------------------------------------------------------------- index
    def index(self, paths: Sequence[str], *, prune: bool = True) -> IndexSnapshot:
        targets = tuple(str(p) for p in paths if str(p).strip())
        with self._engine() as eng:
            try:
                report = eng.index(targets or None, prune=prune)  # type: ignore[attr-defined]
            except OrbitError as exc:
                return IndexSnapshot(paths=targets, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - surface the message, don't crash the UI
                return IndexSnapshot(paths=targets, error=f"{type(exc).__name__}: {exc}")
            return IndexSnapshot(
                paths=targets,
                scanned=report.scanned,
                added=report.added,
                updated=report.updated,
                unchanged=report.unchanged,
                deleted=report.deleted,
                renamed=report.renamed,
                failed=report.failed,
                skipped=report.skipped,
                chunks_written=report.chunks_written,
                embeddings_written=report.embeddings_written,
                duration_ms=report.duration_ms,
                failures=tuple(report.failures),
                skips=tuple(report.skips),
            )

    # ---------------------------------------------------------------- validate
    def validate(self, chunk_id: str) -> ValidationSnapshot:
        chunk_id = (chunk_id or "").strip()
        if not chunk_id:
            return ValidationSnapshot(
                chunk_id="", found=False, state="not_validated", is_valid=False,
                detail="no chunk id given", current_hash=None, current_revision=None,
                error="no chunk id given",
            )
        with self._engine() as eng:
            if not self._is_initialized(eng):
                return ValidationSnapshot(
                    chunk_id=chunk_id, found=False, state="not_validated", is_valid=False,
                    detail="index not initialized", current_hash=None, current_revision=None,
                    error="index not initialized",
                )
            try:
                result = eng.validate_citation(chunk_id)  # type: ignore[attr-defined]
                hit = eng.get_chunk(chunk_id)  # type: ignore[attr-defined]
            except OrbitError as exc:
                return ValidationSnapshot(
                    chunk_id=chunk_id, found=False, state="not_validated", is_valid=False,
                    detail=str(exc), current_hash=None, current_revision=None, error=str(exc),
                )
            source_path = hit.evidence.source_path if hit else ""
            line_range = hit.evidence.line_range if hit else ""
            return ValidationSnapshot(
                chunk_id=chunk_id,
                found=hit is not None,
                state=_enum_value(result.state),
                is_valid=result.is_valid,
                detail=result.detail or ("chunk is not present in the index" if hit is None else ""),
                current_hash=result.current_hash,
                current_revision=result.current_revision,
                source_path=source_path,
                line_range=line_range,
            )

    def freshness(self) -> dict[str, int]:
        from ..evidence import freshness_summary

        with self._engine() as eng:
            if not self._is_initialized(eng):
                return {}
            try:
                return dict(freshness_summary(eng.db))  # type: ignore[attr-defined]
            except OrbitError:
                return {}

    # --------------------------------------------------------------- conflicts
    def conflicts(self, *, limit: int = 50, scan_limit: int = 5000) -> ConflictScanSnapshot:
        from ..evidence import find_conflicts

        with self._engine() as eng:
            if not self._is_initialized(eng):
                return ConflictScanSnapshot(rows=(), scanned=0, total=0)
            try:
                # Count current chunks so the view can report scan coverage.
                total = self._current_chunk_count(eng)
                # persist=False: viewing conflict candidates shouldn't write
                # anything to the conflicts table. find_conflicts caps the scan
                # at scan_limit chunks (LIMIT ? in _load_chunks).
                found = find_conflicts(eng.db, limit=scan_limit, persist=False)  # type: ignore[attr-defined]
            except OrbitError as exc:
                return ConflictScanSnapshot(rows=(), scanned=0, total=0, error=str(exc))
            out: list[ConflictSnapshot] = []
            for c in found[: max(1, limit)]:
                out.append(
                    ConflictSnapshot(
                        rule=c.rule,
                        score=c.score,
                        kind=_enum_value(c.kind),
                        path_a=c.path_a,
                        line_range_a=c.line_range_a,
                        chunk_id_a=c.chunk_id_a,
                        path_b=c.path_b,
                        line_range_b=c.line_range_b,
                        chunk_id_b=c.chunk_id_b,
                        detail=c.detail,
                    )
                )
            scanned = min(scan_limit, total) if total else 0
            return ConflictScanSnapshot(
                rows=tuple(out), scanned=scanned, total=total
            )

    @staticmethod
    def _current_chunk_count(eng: "object") -> int:
        try:
            row = eng.db.query_one(  # type: ignore[attr-defined]
                "SELECT COUNT(*) AS n FROM chunks c "
                "JOIN documents d ON d.doc_id=c.doc_id WHERE d.status='current'"
            )
        except (OrbitError, sqlite3.Error):
            return 0
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ status
    def status(self, *, with_freshness: bool = False) -> StatusSnapshot:
        with self._engine() as eng:
            if not self._is_initialized(eng):
                return StatusSnapshot(
                    initialized=False, schema_version=0, documents_current=0,
                    documents_deleted=0, chunks=0, embeddings=0,
                    embedding_provider="—", embedding_dim=0,
                    db_path=str(self.config.db_path), languages={},
                )
            try:
                stats = eng.stats()  # type: ignore[attr-defined]
            except OrbitError as exc:
                return StatusSnapshot(
                    initialized=False, schema_version=0, documents_current=0,
                    documents_deleted=0, chunks=0, embeddings=0,
                    embedding_provider="—", embedding_dim=0,
                    db_path=str(self.config.db_path), languages={}, error=str(exc),
                )
            fresh: dict[str, int] = {}
            if with_freshness:
                try:
                    from ..evidence import freshness_summary

                    fresh = dict(freshness_summary(eng.db))  # type: ignore[attr-defined]
                except OrbitError:
                    fresh = {}
            return StatusSnapshot(
                initialized=True,
                schema_version=stats.schema_version,
                documents_current=stats.documents_current,
                documents_deleted=stats.documents_deleted,
                chunks=stats.chunks,
                embeddings=stats.embeddings,
                embedding_provider=stats.embedding_provider or "—",
                embedding_dim=stats.embedding_dim,
                db_path=stats.db_path,
                languages=dict(stats.languages),
                freshness=fresh,
            )

    # --------------------------------------------------------------------- mcp
    def mcp_info(self) -> McpSnapshot:
        """Report the MCP tool surface and launch command without starting a server.

        The TUI doesn't run the stdio MCP server in-process — stdio is the MCP
        protocol channel and would collide with the terminal UI. It reports the
        real tool set and the exact command to launch the standalone server, so
        the two stay in step.
        """
        from ..mcp_server import build_tools

        sdk_available = True
        detail = "mcp SDK found; the server can be launched over stdio."
        try:
            from ..mcp_server import _import_server_class

            _import_server_class()
        except OrbitError as exc:
            sdk_available = False
            detail = f"mcp SDK not installed: {exc}. Install with: pip install 'orbit-rag[mcp]'"

        tools: list[tuple[str, str]] = []
        with self._engine() as eng:
            try:
                for name, fn in build_tools(eng).items():  # type: ignore[attr-defined]
                    doc = (fn.__doc__ or "").strip().splitlines()
                    first = doc[0].strip() if doc else ""
                    tools.append((name, first))
            except Exception:  # noqa: BLE001
                tools = []
        abs_db = str(Path(self.config.db_path).resolve())
        launch = f'orbit --db "{abs_db}" mcp'
        return McpSnapshot(
            sdk_available=sdk_available,
            detail=detail,
            tools=tuple(tools),
            launch_command=launch,
        )

    # ------------------------------------------------------------------ config
    def config_rows(self) -> list[tuple[str, str]]:
        """A safe, read-only view of the active configuration for Settings."""
        c = self.config
        return [
            ("Database", str(c.db_path)),
            ("Configured paths", ", ".join(c.paths) or "(none)"),
            ("Embedding provider", c.embedding.provider),
            ("Embedding dimension", str(c.embedding.dimension)),
            ("Retrieval limit", str(c.retrieval.limit)),
            ("Fusion", c.retrieval.fusion),
            ("Candidate multiplier", str(c.retrieval.candidate_multiplier)),
            ("Chunk max tokens", str(c.chunking.max_tokens)),
            ("Chunk overlap tokens", str(c.chunking.overlap_tokens)),
            ("Context token budget", str(c.context.token_budget)),
        ]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _orbit_version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _enum_value(value: "object") -> str:
    return str(getattr(value, "value", value))


def _trim(text: str, width: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= width:
        return text
    return text[:width].rsplit(" ", 1)[0] + " …"


def _snippet_lines(text: str, *, max_lines: int = 3, width: int = 120) -> str:
    """Return up to ``max_lines`` lines of ``text`` with line breaks kept.

    Unlike :func:`_trim`, this doesn't collapse whitespace: internal newlines
    survive so a multi-line passage renders as multiple lines in a result card.
    Leading and trailing blank lines are dropped, an over-long single line is
    hard-cut to ``width`` with a trailing ellipsis, and if the passage runs past
    ``max_lines`` the extra lines become a trailing ellipsis.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not raw:
        return ""
    lines = raw.split("\n")
    kept = lines[:max_lines]
    out: list[str] = []
    for line in kept:
        if len(line) > width:
            line = line[:width].rstrip() + " …"
        out.append(line)
    result = "\n".join(out)
    if len(lines) > max_lines:
        result += "\n…"
    return result


def _format_epoch(raw: "object") -> str:
    if not raw:
        return "never"
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return "unknown"
    if ts <= 0:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
