"""SQLite connection management.

Every connection goes through :class:`OrbitDB`; nothing else opens one directly.

WAL so readers don't block the indexer, ``synchronous=NORMAL`` because this is a
rebuildable derived index, foreign keys on so the chunk/embedding cascades on
delete actually fire, and a busy timeout so a second CLI invocation waits instead
of erroring.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..errors import IndexCorruptError, StorageError
from .schema import SCHEMA_VERSION, migration_statements

__all__ = ["OrbitDB"]


class OrbitDB:
    """Thin, explicit wrapper around a single SQLite connection."""

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self._timeout = timeout
        self._conn: sqlite3.Connection | None = None

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> OrbitDB:
        if self._conn is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self.path), timeout=self._timeout, isolation_level=None)
        except sqlite3.Error as exc:  # pragma: no cover - environment specific
            raise StorageError(f"could not open database {self.path}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        self._conn = conn
        try:
            self._apply_pragmas(conn)
        except sqlite3.DatabaseError as exc:
            # A corrupt file surfaces here as sqlite3.DatabaseError ("file is not
            # a database"). Keep it inside the Orbit error hierarchy so callers'
            # `except OrbitError` handlers actually catch it.
            conn.close()
            self._conn = None
            raise IndexCorruptError(f"{self.path} is not a valid Orbit index: {exc}") from exc
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> OrbitDB:
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def _apply_pragmas(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(30_000)}")
        conn.execute("PRAGMA temp_store=MEMORY")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("database is not open")
        return self._conn

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    # ---------------------------------------------------------------- migrations
    def migrate(self) -> int:
        """Apply any pending migrations. Returns the resulting schema version."""
        self._ensure_meta_table()
        current = self.schema_version()
        target = SCHEMA_VERSION
        if current > target:
            raise StorageError(
                f"index schema version {current} is newer than this build supports "
                f"({target}); upgrade Orbit or rebuild the index"
            )
        for version in range(current + 1, target + 1):
            for stmt in migration_statements(version):
                try:
                    self.conn.execute(stmt)
                except sqlite3.Error as exc:
                    raise StorageError(
                        f"migration to schema v{version} failed: {exc}", detail=stmt[:200]
                    ) from exc
        self.set_meta("schema_version", str(target))
        if self.get_meta("created_at") is None:
            self.set_meta("created_at", str(time.time()))
        self.set_meta("updated_at", str(time.time()))
        return target

    def _ensure_meta_table(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS orbit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = self.conn.execute(
            "SELECT value FROM orbit_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT OR REPLACE INTO orbit_meta(key, value) VALUES('schema_version', '0')"
            )

    def schema_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM orbit_meta WHERE key='schema_version'"
        ).fetchone()
        return int(row["value"]) if row and row["value"] else 0

    def verify_fts(self) -> None:
        """Ensure the FTS index is usable; rebuild it if it is out of sync."""
        try:
            self.conn.execute("SELECT count(*) FROM chunks_fts LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise IndexCorruptError(f"FTS index unavailable: {exc}") from exc

    def rebuild_fts(self) -> None:
        """Rebuild the FTS external-content index from ``chunks``."""
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    def fts_count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])

    # ---------------------------------------------------------------- meta k/v
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM orbit_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO orbit_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ---------------------------------------------------------------- tx helpers
    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Explicit transaction. Commits on success, rolls back on any exception."""
        conn = self.conn
        already = conn.in_transaction
        if already:
            yield conn
            return
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ---------------------------------------------------------------- helpers
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        try:
            return self.conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise StorageError(f"SQL error: {exc}", detail=sql[:200]) from exc

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
        seq = list(seq)
        try:
            return self.conn.executemany(sql, seq)
        except sqlite3.Error as exc:
            raise StorageError(f"SQL error: {exc}", detail=sql[:200]) from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        return row[0] if row else None

    def fts_available(self) -> bool:
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.__orbit_fts_probe USING fts5(x)"
            )
            self.conn.execute("DROP TABLE IF EXISTS temp.__orbit_fts_probe")
            return True
        except sqlite3.Error:
            return False