"""SQLite schema.

Versioned through ``orbit_meta.schema_version``. Migrations apply in order and
every statement is idempotent, so an interrupted one is safe to re-run.

``chunks`` holds the text; ``chunks_fts`` is an FTS5 external-content index over
it, so the text is stored once and kept in sync inside the chunk transaction.

``evidence_edges`` and ``conflicts`` have no FK to ``chunks`` -- the indexer
purges them explicitly when a document goes away.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

# Tokenizer choice: unicode61 with diacritics folded, ASCII-folding off so
# non-Latin scripts survive. bm25() ranking uses SQLite's built-in BM25 which
# implements the Robertson/Sparck-Jones IDF weighting.
FTS_TOKENIZER = "unicode61 remove_diacritics 2"

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS orbit_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id       TEXT PRIMARY KEY,
            path         TEXT NOT NULL,
            abs_path     TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL,
            revision     INTEGER NOT NULL DEFAULT 1,
            size_bytes   INTEGER NOT NULL DEFAULT 0,
            mtime        REAL    NOT NULL DEFAULT 0,
            language     TEXT    NOT NULL DEFAULT 'text',
            status       TEXT    NOT NULL DEFAULT 'current',
            indexed_at   REAL    NOT NULL DEFAULT 0,
            metadata     TEXT    NOT NULL DEFAULT '{}'
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path
            ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(status);
        CREATE INDEX IF NOT EXISTS idx_documents_hash
            ON documents(content_hash);

        CREATE TABLE IF NOT EXISTS chunks (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id       TEXT NOT NULL UNIQUE,
            doc_id         TEXT NOT NULL,
            ordinal        INTEGER NOT NULL,
            text           TEXT NOT NULL,
            content_hash   TEXT NOT NULL,
            start_line     INTEGER NOT NULL,
            end_line       INTEGER NOT NULL,
            start_offset   INTEGER NOT NULL,
            end_offset     INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL DEFAULT 0,
            metadata       TEXT    NOT NULL DEFAULT '{}',
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc
            ON chunks(doc_id, ordinal);

        -- FTS5 external-content index. 'content=chunks' means the text lives in
        -- the chunks table; we keep the two in sync inside the same transaction.
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='rowid',
            tokenize=FTS_TOKENIZER_PLACEHOLDER
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT PRIMARY KEY,
            dim      INTEGER NOT NULL,
            vector   BLOB    NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chunk_symbols (
            chunk_id TEXT NOT NULL,
            symbol   TEXT NOT NULL,
            kind     TEXT NOT NULL DEFAULT 'symbol',
            PRIMARY KEY (chunk_id, symbol),
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_symbols_symbol
            ON chunk_symbols(symbol);

        CREATE TABLE IF NOT EXISTS evidence_edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            kind      TEXT NOT NULL,
            relation  TEXT NOT NULL,
            weight    REAL NOT NULL DEFAULT 1.0,
            detail    TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (source_id, target_id, relation)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_source ON evidence_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON evidence_edges(target_id);

        CREATE TABLE IF NOT EXISTS conflicts (
            conflict_id   TEXT PRIMARY KEY,
            chunk_id_a    TEXT NOT NULL,
            chunk_id_b    TEXT NOT NULL,
            rule          TEXT NOT NULL,
            score         REAL NOT NULL,
            kind          TEXT NOT NULL DEFAULT 'inferred',
            detail        TEXT NOT NULL DEFAULT '',
            detected_at   REAL NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_conflicts_a ON conflicts(chunk_id_a);
        CREATE INDEX IF NOT EXISTS idx_conflicts_b ON conflicts(chunk_id_b);
        """,
    ),
)


def _split_statements(script: str) -> list[str]:
    """Split a SQL script into statements, ignoring ``;`` inside quotes/comments."""
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(script)
    while i < n:
        ch = script[i]
        nxt = script[i + 1] if i + 1 < n else ""
        # Line comment
        if ch == "-" and nxt == "-":
            j = script.find("\n", i)
            if j == -1:
                break
            buf.append(script[i:j])
            i = j
            continue
        # Block comment
        if ch == "/" and nxt == "*":
            j = script.find("*/", i + 2)
            j = n if j == -1 else j + 2
            buf.append(script[i:j])
            i = j
            continue
        # Quoted string (single or double or backtick); '' escapes
        if ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
            i += 1
            while i < n:
                buf.append(script[i])
                if script[i] == quote:
                    # doubled quote is an escape, not a terminator
                    if i + 1 < n and script[i + 1] == quote:
                        buf.append(script[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def migration_statements(version: int) -> list[str]:
    """Return the DDL statements for a single schema version."""
    for ver, script in MIGRATIONS:
        if ver == version:
            resolved = script.replace("FTS_TOKENIZER_PLACEHOLDER", repr(FTS_TOKENIZER))
            return _split_statements(resolved)
    raise ValueError(f"unknown schema version {version}")


def render_script(version: int) -> str:
    """Return the raw migration script (used for docs/tests)."""
    for ver, script in MIGRATIONS:
        if ver == version:
            return script.replace("FTS_TOKENIZER_PLACEHOLDER", repr(FTS_TOKENIZER))
    raise ValueError(f"unknown schema version {version}")