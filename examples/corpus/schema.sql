-- Example schema for the demo corpus.
--
-- This file exists to be indexed and searched, not to be executed.

CREATE TABLE sessions (
    id            TEXT PRIMARY KEY,
    caller        TEXT NOT NULL,
    issued_at     INTEGER NOT NULL,
    expires_at    INTEGER NOT NULL,
    revoked_at    INTEGER
);

CREATE INDEX idx_sessions_caller ON sessions (caller, expires_at);

CREATE TABLE idempotency_keys (
    key         TEXT PRIMARY KEY,
    route       TEXT NOT NULL,
    response_id TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

-- NOTE: this default disagrees with the value stated in operations.md.
-- `orbit conflicts` is expected to flag the disagreement as a candidate.
CREATE TABLE settings (
    name             TEXT PRIMARY KEY,
    session_ttl_seconds INTEGER NOT NULL DEFAULT 1800,
    retry_limit      INTEGER NOT NULL DEFAULT 3,
    max_file_bytes   INTEGER NOT NULL DEFAULT 2000000
);

-- Sessions older than 30 days are removed by the nightly job.
CREATE TABLE retention_policy (
    resource      TEXT PRIMARY KEY,
    retain_days   INTEGER NOT NULL,
    enforce_daily INTEGER NOT NULL DEFAULT 1
);

INSERT INTO retention_policy (resource, retain_days) VALUES ('sessions', 30);
INSERT INTO retention_policy (resource, retain_days) VALUES ('idempotency_keys', 7);