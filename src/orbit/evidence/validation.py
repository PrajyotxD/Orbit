"""Citation validation.

Re-reads the cited lines from the live file and compares hashes. This answers
one question -- does the pointer still resolve -- and nothing about whether the
sentence built on top of it is true.
"""

from __future__ import annotations

from pathlib import Path

from ..models import ValidationResult, ValidationState, sha256_text

__all__ = ["validate_evidence", "validate_citation", "chunk_matches_source"]


def _chunk_hash(text: str, start_line: int) -> str:
    return sha256_text(f"{start_line}:{text}")


def validate_evidence(db, chunk_id: str) -> ValidationResult:
    """Validate one chunk against the live filesystem."""
    row = db.query_one(
        "SELECT c.chunk_id, c.text, c.content_hash, c.start_line, c.end_line, "
        "c.doc_id, d.path, d.abs_path, d.status AS doc_status, d.content_hash AS doc_hash, "
        "d.revision "
        "FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE c.chunk_id=?",
        (chunk_id,),
    )
    if row is None:
        return ValidationResult(ValidationState.DOCUMENT_DELETED, "chunk not in index")

    revision = int(row["revision"] or 0)

    if row["doc_status"] != "current":
        return ValidationResult(
            ValidationState.DOCUMENT_DELETED,
            "document is marked deleted in the index",
            current_revision=revision,
        )

    path = Path(row["abs_path"]) if row["abs_path"] else None
    if path is None or not path.exists():
        return ValidationResult(
            ValidationState.PATH_MISSING,
            f"source file missing: {row['path']}",
            current_revision=revision,
        )

    try:
        live = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ValidationResult(
            ValidationState.PATH_MISSING,
            f"cannot read source: {exc}",
            current_revision=revision,
        )

    live = live.replace("\r\n", "\n").replace("\r", "\n")
    lines = live.split("\n")
    start, end = row["start_line"], row["end_line"]
    if end > len(lines):
        return ValidationResult(
            ValidationState.CHUNK_OUT_OF_RANGE,
            f"cited lines {start}-{end} exceed file length {len(lines)}",
            current_revision=revision,
        )

    region = "\n".join(lines[start - 1 : end]).rstrip("\n")
    live_hash = _chunk_hash(region, start)
    if live_hash == row["content_hash"]:
        return ValidationResult(
            ValidationState.VALID,
            "cited region matches the live source",
            current_hash=live_hash,
            current_revision=revision,
        )
    return ValidationResult(
        ValidationState.CONTENT_CHANGED,
        "source was edited since indexing; cited region no longer matches",
        current_hash=live_hash,
        current_revision=revision,
    )


def validate_citation(db, chunk_id: str) -> ValidationResult:
    """Alias with a clearer public name."""
    return validate_evidence(db, chunk_id)


def chunk_matches_source(db, chunk_id: str) -> bool:
    return validate_evidence(db, chunk_id).state is ValidationState.VALID


def resolve_source_path(db, chunk_id: str) -> str | None:
    row = db.query_one(
        "SELECT d.path FROM chunks c JOIN documents d ON d.doc_id=c.doc_id WHERE c.chunk_id=?",
        (chunk_id,),
    )
    return row["path"] if row else None


def verify_document(db, doc_id: str) -> ValidationResult:
    """Validate that a document's stored content hash matches the live file."""
    row = db.query_one(
        "SELECT path, abs_path, content_hash, status FROM documents WHERE doc_id=?", (doc_id,)
    )
    if row is None:
        return ValidationResult(ValidationState.DOCUMENT_DELETED, "document not in index")
    if row["status"] != "current":
        return ValidationResult(ValidationState.DOCUMENT_DELETED, "document is deleted")
    path = Path(row["abs_path"]) if row["abs_path"] else None
    if path is None or not path.exists():
        return ValidationResult(ValidationState.PATH_MISSING, f"missing: {row['path']}")
    try:
        live = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ValidationResult(ValidationState.PATH_MISSING, f"cannot read: {exc}")
    live = live.replace("\r\n", "\n").replace("\r", "\n")
    if sha256_text(live) == row["content_hash"]:
        return ValidationResult(ValidationState.VALID, "document content matches", current_hash=row["content_hash"])
    return ValidationResult(
        ValidationState.CONTENT_CHANGED, "document content differs from index"
    )