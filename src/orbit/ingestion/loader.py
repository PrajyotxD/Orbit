"""Document loading: discovery, safety filters, decoding.

Indexed files are treated as untrusted data throughout -- decoded to text,
never executed. A file we cannot use becomes a :class:`LoadFailure` rather than
an exception, so one bad file never kills a run.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    LANGUAGE_BY_EXTENSION,
    SECRET_CONTENT_PATTERNS,
    SECRET_LIKE_FILENAMES,
    IngestionConfig,
)
from ..errors import PathSafetyError
from ..models import LoadFailure, LoadFailureReason, SourceDoc, sha256_text

__all__ = ["discover_files", "load_file", "iter_source_docs", "is_secret_filename"]


@dataclass
class _Candidate:
    path: Path
    root: Path


def is_secret_filename(name: str) -> bool:
    base = name.lower()
    if base in {s.lower() for s in SECRET_LIKE_FILENAMES}:
        return True
    # Catch variants like .env.local, *.pem, *.key, id_rsa.pub
    if base.endswith((".pem", ".key", ".p12", ".pfx", ".jks", ".keystore")):
        return True
    if base.startswith(".env"):
        return True
    return False


def _matches_any(name: str, patterns: set[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def _is_excluded_dir(name: str, excluded: set[str]) -> bool:
    return name in excluded or name.startswith(".") and name not in (".", "..")


def _within_root(path: Path, root: Path) -> bool:
    """Guard against path traversal / symlink escapes."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


_GLOB_CHARS = ("*", "?", "[")


def is_excluded_by(path: Path, root: Path, patterns: Sequence[str]) -> bool:
    """True when ``path`` is covered by one of the operator's ``exclude_paths``.

    Each pattern is either a glob or a plain path:

    * a glob matches the root-relative path, the absolute path, or the file name;
    * a plain path matches when it names the file, names an ancestor directory,
      or is itself the absolute location of either.

    Matching is done on resolved paths so ``./a/../b`` and symlinked spellings
    behave the same as the obvious form.
    """
    if not patterns:
        return False

    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        rel = path.as_posix()
    absolute = path.as_posix()

    for raw in patterns:
        pattern = str(raw).strip()
        if not pattern:
            continue

        if any(ch in pattern for ch in _GLOB_CHARS):
            if (
                fnmatch.fnmatch(rel, pattern)
                or fnmatch.fnmatch(absolute, pattern)
                or fnmatch.fnmatch(path.name, pattern)
            ):
                return True
            continue

        # Plain path: compare positionally, absolute vs relative, and treat a
        # directory entry as excluding everything beneath it.
        candidate = Path(pattern).expanduser()
        candidates = {candidate.as_posix()}
        if candidate.is_absolute():
            candidates.add(candidate.as_posix().rstrip("/"))
        for text in candidates:
            if text in (rel, absolute, path.name):
                return True
            if rel == text or rel.startswith(text.rstrip("/") + "/"):
                return True
            if absolute == text or absolute.startswith(text.rstrip("/") + "/"):
                return True
    return False


def discover_files(
    roots: list[str] | list[Path],
    config: IngestionConfig | None = None,
    exclude: Sequence[str] = (),
) -> tuple[list[Path], list[LoadFailure]]:
    """Return ``(candidates, failures)`` for all indexable files under ``roots``.

    ``exclude`` holds the operator's ``exclude_paths``: globs and paths that must
    never be indexed. Ordering is deterministic -- roots in the order given,
    files sorted by their path relative to their root.
    """
    config = config or IngestionConfig()
    excluded_dirs = config.effective_excluded_dirs()
    excluded_globs = config.effective_excluded_globs()
    allowed_ext = config.allowed_extensions()
    exclude = tuple(exclude)

    candidates: list[Path] = []
    failures: list[LoadFailure] = []

    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            failures.append(
                LoadFailure(str(root), LoadFailureReason.UNREADABLE, "path does not exist")
            )
            continue
        if root.is_file():
            if not is_excluded_by(root, root.parent, exclude):
                candidates.append(root)
            continue

        for dirpath, dirnames, filenames in os.walk(root, followlinks=config.follow_symlinks):
            here = Path(dirpath)
            # Prune excluded dirs (and hidden dirs) in place.
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in excluded_dirs
                and not (d.startswith(".") and d not in (".github",))
                and not is_excluded_by(here / d, root, exclude)
            )
            for filename in sorted(filenames):
                fp = here / filename
                if not _within_root(fp, root):
                    failures.append(
                        LoadFailure(str(fp), LoadFailureReason.EXCLUDED, "outside root")
                    )
                    continue
                if is_excluded_by(fp, root, exclude):
                    continue
                if _matches_any(filename, excluded_globs):
                    continue
                if config.skip_secret_like_files and is_secret_filename(filename):
                    failures.append(
                        LoadFailure(str(fp), LoadFailureReason.SECRET_PATTERN, "secret-like name")
                    )
                    continue
                ext = fp.suffix.lower()
                if allowed_ext is not None:
                    if ext not in allowed_ext:
                        continue
                elif ext and ext not in LANGUAGE_BY_EXTENSION:
                    # Unknown extension: skip silently (not an error).
                    continue
                candidates.append(fp)

    return candidates, failures


def _looks_binary(path: Path, sniff_bytes: int) -> bool:
    try:
        with path.open("rb") as fh:
            blob = fh.read(sniff_bytes)
    except OSError:
        return False
    if b"\x00" in blob:
        return True
    # Heuristic: high ratio of non-text bytes.
    if not blob:
        return False
    nontext = sum(1 for b in blob if b < 9 or (13 < b < 32) or b > 126)
    return nontext / len(blob) > 0.30


def _decode(raw: bytes, config: IngestionConfig) -> str:
    for enc in (config.encoding, "utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc, errors=config.decode_errors if enc == config.encoding else "strict")
        except (UnicodeDecodeError, LookupError):
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "no usable encoding")


def _language_for(path: Path) -> str:
    name = path.name.lower()
    if name in ("dockerfile", "containerfile"):
        return "dockerfile"
    if name in ("makefile",):
        return "make"
    ext = path.suffix.lower()
    return LANGUAGE_BY_EXTENSION.get(ext, "text")


def load_file(
    path: Path,
    root: Path,
    config: IngestionConfig | None = None,
) -> tuple[SourceDoc | None, LoadFailure | None]:
    """Load one file into a :class:`SourceDoc`, or return a :class:`LoadFailure`."""
    config = config or IngestionConfig()
    try:
        if not _within_root(path, root):
            raise PathSafetyError(f"{path} escapes root {root}")
        stat = path.stat()
        if not path.is_file():
            raise OSError("not a regular file")
        if config.max_file_bytes and stat.st_size > config.max_file_bytes:
            return None, LoadFailure(
                str(path), LoadFailureReason.TOO_LARGE, f"{stat.st_size} bytes"
            )
        if config.skip_secret_like_files and is_secret_filename(path.name):
            return None, LoadFailure(str(path), LoadFailureReason.SECRET_PATTERN, "secret-like name")
        binary_looking = _looks_binary(path, config.text_sniff_bytes)

        raw = path.read_bytes()

        if config.scan_content_for_secrets:
            head = raw[: config.secret_content_scan_bytes]
            for pattern in SECRET_CONTENT_PATTERNS:
                if _re_search(pattern, head):
                    return None, LoadFailure(
                        str(path), LoadFailureReason.SECRET_PATTERN, "secret-like content"
                    )

        try:
            text = _decode(raw, config)
        except (UnicodeDecodeError, LookupError) as exc:
            # A file that only *looked* binary because of a few stray high bytes
            # is really a decode problem, and saying so is far more useful to an
            # operator than a generic "unsupported binary".
            reason = (
                LoadFailureReason.DECODE_ERROR
                if binary_looking
                else LoadFailureReason.UNSUPPORTED
            )
            detail = str(exc)[:120] if binary_looking else "binary content"
            return None, LoadFailure(str(path), reason, detail)
        if binary_looking:
            return None, LoadFailure(str(path), LoadFailureReason.UNSUPPORTED, "binary content")

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return None, LoadFailure(str(path), LoadFailureReason.EMPTY, "no text content")

        language = _language_for(path)
        return (
            SourceDoc(
                path=_relpath(path, root),
                abs_path=str(path),
                text=text,
                language=language,
                content_hash=sha256_text(text),
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                metadata={"suffix": path.suffix.lower()},
            ),
            None,
        )
    except PathSafetyError as exc:
        return None, LoadFailure(str(path), LoadFailureReason.EXCLUDED, str(exc)[:120])
    except PermissionError as exc:
        return None, LoadFailure(str(path), LoadFailureReason.UNREADABLE, str(exc)[:120])
    except (OSError, ValueError) as exc:
        return None, LoadFailure(str(path), LoadFailureReason.UNREADABLE, str(exc)[:120])


def _relpath(path: Path, root: Path) -> str:
    """Root-relative display path for a source file.

    When a *file* is passed as its own root, ``relative_to`` yields ``"."``,
    which would collapse every individually-indexed file onto the same path (and
    therefore the same document id). In that case the file's own name is used,
    with a parent prefix only when the file sits below its root's directory, so
    two files indexed individually never collide.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return path.name
    text = rel.as_posix()
    if text in (".", "", "/"):
        return path.name
    return text


def _re_search(pattern: str, data: bytes) -> bool:
    import re

    try:
        return re.search(pattern.encode("latin-1", "ignore"), data) is not None
    except re.error:
        return False


def iter_source_docs(
    roots: list[str] | list[Path],
    config: IngestionConfig | None = None,
    exclude: Sequence[str] = (),
) -> Iterator[SourceDoc | LoadFailure]:
    """Discover and load every file under ``roots``, yielding docs and failures."""
    config = config or IngestionConfig()
    candidates, failures = discover_files(roots, config, exclude)
    yield from failures
    for path in candidates:
        root = _root_for(path, roots)
        doc, failure = load_file(path, root, config)
        if doc is not None:
            yield doc
        elif failure is not None:
            yield failure


def _root_for(path: Path, roots: list[str] | list[Path]) -> Path:
    best = Path(roots[0])
    best_len = -1
    for raw in roots:
        r = Path(raw).expanduser()
        try:
            resolved = r.resolve()
        except OSError:
            continue
        if _within_root(path, resolved):
            n = len(str(resolved))
            if n > best_len:
                best_len = n
                best = r
    return best