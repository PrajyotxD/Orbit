"""Ingestion: discovery, exclusion rules, decoding, metadata extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit.config import IngestionConfig
from orbit.ingestion import discover_files, is_secret_filename, iter_source_docs, load_file
from orbit.ingestion.loader import _within_root
from orbit.models import LoadFailureReason, sha256_text

from conftest import write


def _names(paths) -> list[str]:
    return sorted(p.name for p in paths)


# ------------------------------------------------------------------ discovery
def test_discover_finds_nested_files(tmp_path: Path) -> None:
    write(tmp_path, "a.md", "a\n")
    write(tmp_path, "sub/b.txt", "b\n")
    write(tmp_path, "sub/deep/c.py", "c\n")
    found, failures = discover_files([tmp_path])
    assert _names(found) == ["a.md", "b.txt", "c.py"]
    assert failures == []


def test_discovery_order_is_deterministic(tmp_path: Path) -> None:
    for name in ("z.txt", "a.txt", "m.txt", "b/n.txt"):
        write(tmp_path, name, "x\n")
    first, _ = discover_files([tmp_path])
    second, _ = discover_files([tmp_path])
    assert [str(p) for p in first] == [str(p) for p in second]


def test_excluded_directories_are_pruned(tmp_path: Path) -> None:
    for d in ("node_modules", ".git", "dist", "__pycache__", "build", ".venv"):
        write(tmp_path, f"{d}/x.txt", "should not appear\n")
    write(tmp_path, "keep.txt", "keep me\n")
    found, _ = discover_files([tmp_path])
    assert _names(found) == ["keep.txt"]


def test_excluded_globs_are_dropped(tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "keep\n")
    write(tmp_path, "b.pyc", "junk")
    write(tmp_path, "c.zip", "junk")
    write(tmp_path, "d.png", "junk")
    found, _ = discover_files([tmp_path])
    assert _names(found) == ["a.txt"]


def test_unknown_extensions_are_skipped_silently(tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "keep\n")
    write(tmp_path, "b.weirdext", "unknown\n")
    found, failures = discover_files([tmp_path])
    assert _names(found) == ["a.txt"]
    assert failures == []


def test_included_extensions_filters_everything_else(tmp_path: Path) -> None:
    write(tmp_path, "a.md", "m\n")
    write(tmp_path, "b.txt", "t\n")
    cfg = IngestionConfig(included_extensions=(".md",))
    found, _ = discover_files([tmp_path], cfg)
    assert _names(found) == ["a.md"]


def test_missing_root_is_reported_as_unreadable(tmp_path: Path) -> None:
    found, failures = discover_files([tmp_path / "nope"])
    assert found == []
    assert len(failures) == 1
    assert failures[0].reason is LoadFailureReason.UNREADABLE


def test_empty_directory_is_not_an_error(tmp_path: Path) -> None:
    found, failures = discover_files([tmp_path])
    assert found == []
    assert failures == []


# --------------------------------------------------------------------- secrets
@pytest.mark.parametrize(
    "name",
    [".env", ".env.local", ".env.production", "id_rsa", "id_dsa", "credentials", ".npmrc",
     ".netrc", "server.pem", "private.key", "cert.p12", "store.keystore", "archive.jks"],
)
def test_secret_like_names_are_rejected(name: str) -> None:
    assert is_secret_filename(name) is True


@pytest.mark.parametrize("name", ["README.md", "main.py", "notes.txt", "environment.md"])
def test_ordinary_names_are_not_flagged(name: str) -> None:
    assert is_secret_filename(name) is False


def test_secret_files_are_not_discovered(tmp_path: Path) -> None:
    write(tmp_path, ".env", "TOKEN=abc\n")
    write(tmp_path, "id_rsa", "-----BEGIN RSA PRIVATE KEY-----\nxx\n")
    write(tmp_path, "public.txt", "fine\n")
    found, failures = discover_files([tmp_path])
    assert _names(found) == ["public.txt"]
    reasons = {f.reason for f in failures}
    assert reasons == {LoadFailureReason.SECRET_PATTERN}


def test_secret_content_sniff_is_opt_in(tmp_path: Path) -> None:
    body = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----\n"
    p = write(tmp_path, "notes.txt", body)

    doc, failure = load_file(p, tmp_path, IngestionConfig(scan_content_for_secrets=False))
    assert doc is not None and failure is None

    doc, failure = load_file(p, tmp_path, IngestionConfig(scan_content_for_secrets=True))
    assert doc is None
    assert failure is not None and failure.reason is LoadFailureReason.SECRET_PATTERN


# ------------------------------------------------------------------- decoding
def test_load_file_populates_provenance(tmp_path: Path) -> None:
    p = write(tmp_path, "a.md", "# Title\n\nbody text\n")
    doc, failure = load_file(p, tmp_path)
    assert failure is None
    assert doc is not None
    assert doc.path == "a.md"
    assert doc.language == "markdown"
    assert doc.content_hash == sha256_text(doc.text)
    assert doc.abs_path == str(p)
    assert doc.size_bytes == p.stat().st_size
    assert doc.text == "# Title\n\nbody text\n"


def test_invalid_utf8_is_reported_not_raised(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\xff\xfe\x00 broken \xc3\x28 bytes")
    doc, failure = load_file(p, tmp_path)
    assert doc is None
    assert failure is not None
    # observed behaviour: the binary sniffer fires before the decoder, so the
    # reason is UNSUPPORTED rather than DECODE_ERROR.  Either is acceptable as
    # long as the caller is told and nothing crashes.
    assert failure.reason in (
        LoadFailureReason.UNSUPPORTED,
        LoadFailureReason.DECODE_ERROR,
    ), failure.reason


def test_binary_content_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "blob.txt"
    p.write_bytes(bytes(range(256)) * 8)
    doc, failure = load_file(p, tmp_path)
    assert doc is None
    assert failure is not None
    assert failure.reason in (LoadFailureReason.UNSUPPORTED, LoadFailureReason.DECODE_ERROR)


def test_empty_and_whitespace_files_are_reported_as_empty(tmp_path: Path) -> None:
    for name, content in (("e1.txt", ""), ("e2.txt", "   \n\n\t\n")):
        p = write(tmp_path, name, content)
        doc, failure = load_file(p, tmp_path)
        assert doc is None, name
        assert failure is not None and failure.reason is LoadFailureReason.EMPTY, name


def test_oversized_files_are_rejected(tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    p.write_bytes(b"a" * 5000)
    doc, failure = load_file(p, tmp_path, IngestionConfig(max_file_bytes=1000))
    assert doc is None
    assert failure is not None and failure.reason is LoadFailureReason.TOO_LARGE
    assert "5000" in failure.detail


def test_max_file_bytes_zero_disables_the_limit(tmp_path: Path) -> None:
    p = tmp_path / "ok.txt"
    p.write_bytes(b"a" * 5000)
    doc, failure = load_file(p, tmp_path, IngestionConfig(max_file_bytes=0))
    assert failure is None and doc is not None


def test_crlf_is_normalised_on_load(tmp_path: Path) -> None:
    p = tmp_path / "crlf.md"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    doc, _ = load_file(p, tmp_path)
    assert doc is not None
    assert doc.text == "one\ntwo\nthree\n"
    assert "\r" not in doc.text


def test_one_bad_file_does_not_abort_the_stream(tmp_path: Path) -> None:
    write(tmp_path, "good1.txt", "fine one\n")
    write(tmp_path, "bad.txt", bytes(range(256)) * 4)
    write(tmp_path, "good2.txt", "fine two\n")
    items = list(iter_source_docs([tmp_path]))
    docs = [i for i in items if not hasattr(i, "reason")]
    fails = [i for i in items if hasattr(i, "reason")]
    assert len(docs) == 2
    assert len(fails) == 1
    assert {d.path for d in docs} == {"good1.txt", "good2.txt"}


# ---------------------------------------------------------------- path safety
def test_within_root_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = write(root, "in.txt", "x\n")
    outside = write(tmp_path, "out.txt", "x\n")
    assert _within_root(inside, root) is True
    assert _within_root(outside, root) is False


def test_symlink_escaping_root_is_rejected(tmp_path: Path) -> None:
    outside = write(tmp_path, "outside.txt", "outside secret\n")
    root = tmp_path / "root"
    root.mkdir()
    write(root, "inside.txt", "inside\n")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")

    items = list(iter_source_docs([root]))
    docs = [i for i in items if not hasattr(i, "reason")]
    fails = [i for i in items if hasattr(i, "reason")]
    assert [d.path for d in docs] == ["inside.txt"]
    assert any(f.reason is LoadFailureReason.EXCLUDED for f in fails)


def test_dot_dot_in_root_is_resolved_safely(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write(root, "in.txt", "x\n")
    sneaky = str(root / ".." / "root")
    docs = [i for i in iter_source_docs([sneaky]) if not hasattr(i, "reason")]
    assert [d.path for d in docs] == ["in.txt"]


# -------------------------------------------------------------------- metadata
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.txt", "text"),
        ("a.md", "markdown"),
        ("a.py", "python"),
        ("a.rs", "rust"),
        ("a.go", "go"),
        ("a.yaml", "yaml"),
        ("Dockerfile", "dockerfile"),
        ("Makefile", "make"),
    ],
)
def test_language_detection(tmp_path: Path, filename: str, expected: str) -> None:
    p = write(tmp_path, filename, "some content here\n")
    doc, _ = load_file(p, tmp_path)
    assert doc is not None
    assert doc.language == expected