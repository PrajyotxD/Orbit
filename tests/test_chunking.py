"""Chunking: determinism, budget discipline, and *real* line provenance.

The central adversarial claim under test is the one Orbit makes in its own
docs: ``start_line``/``end_line`` always point at real lines in the source
file, so a cited passage can always be re-read from disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orbit.chunking import chunk_document, estimate_tokens, iter_chunks
from orbit.config import ChunkingConfig
from orbit.errors import ChunkingError

from conftest import read_lines, write


def _assert_line_provenance(text: str, chunks) -> None:
    """Every chunk's text must really live inside its own cited line range."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for ch in chunks:
        assert 1 <= ch.start_line <= ch.end_line, (ch.start_line, ch.end_line)
        assert ch.end_line <= len(lines), (
            f"chunk {ch.ordinal} cites end_line={ch.end_line} but file has {len(lines)} lines"
        )
        region = "\n".join(lines[ch.start_line - 1 : ch.end_line])
        # 1. the cited region really contains the chunk verbatim
        assert ch.text in region, (
            f"chunk {ch.ordinal} text not found in its cited region "
            f"[{ch.start_line}-{ch.end_line}]: {ch.text!r} vs {region!r}"
        )
        # 2. the chunk text exists verbatim somewhere in the real file
        assert ch.text in text, f"chunk {ch.ordinal} text is not verbatim source text"
        # 3. locating the chunk's meaningful content in the file yields lines
        #    that fall inside the cited range
        stripped = ch.text.strip()
        if stripped:
            idx = text.index(stripped)
            first = text[:idx].count("\n") + 1
            last = first + stripped.count("\n")
            assert ch.start_line <= first, (
                f"chunk {ch.ordinal} claims start_line={ch.start_line} but its text "
                f"actually begins on line {first}"
            )
            assert last <= ch.end_line, (
                f"chunk {ch.ordinal} claims end_line={ch.end_line} but its text "
                f"actually ends on line {last}"
            )


# ------------------------------------------------------------------ estimator
def test_estimate_tokens_is_monotonic_and_sane() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 40, 4.0) == 10
    assert estimate_tokens("a" * 41, 4.0) == 11
    assert estimate_tokens("a" * 400, 4.0) > estimate_tokens("a" * 100, 4.0)


# ------------------------------------------------------------------ edge cases
def test_empty_and_whitespace_input_yield_no_chunks() -> None:
    assert chunk_document("") == []
    assert chunk_document("   \n\n\t\n") == []
    assert list(iter_chunks("")) == []


def test_invalid_overlap_raises_chunking_error() -> None:
    # Build a config that is valid, then force the invalid state to reach
    # chunk_document's own guard rather than ChunkingConfig.__post_init__.
    cfg = ChunkingConfig(max_tokens=10, overlap_tokens=1)
    object.__setattr__(cfg, "overlap_tokens", 10)
    with pytest.raises(ChunkingError):
        chunk_document("some text here", cfg)


def test_single_line_file_produces_one_chunk() -> None:
    chunks = chunk_document("just one line", ChunkingConfig())
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 1)
    assert chunks[0].text == "just one line"


def test_oversized_single_line_is_not_infinite_loop() -> None:
    cfg = ChunkingConfig(max_tokens=5, overlap_tokens=1)
    chunks = chunk_document("x" * 200_000, cfg)
    assert len(chunks) >= 1
    _assert_line_provenance("x" * 200_000, chunks)


def test_no_trailing_newline() -> None:
    text = "alpha\nbravo\ncharlie"
    chunks = chunk_document(text, ChunkingConfig(max_tokens=4, overlap_tokens=1, min_chunk_tokens=0))
    _assert_line_provenance(text, chunks)
    assert chunks[-1].end_line == 3


def test_crlf_file_is_normalised_consistently(tmp_path: Path) -> None:
    raw = "alpha line\r\nbravo line\r\ncharlie line\r\n"
    p = tmp_path / "crlf.txt"
    p.write_bytes(raw.encode("utf-8"))
    text = raw.replace("\r\n", "\n")
    chunks = chunk_document(text, ChunkingConfig(max_tokens=6, overlap_tokens=1, min_chunk_tokens=0))
    _assert_line_provenance(text, chunks)
    # line count as the validator sees it
    assert read_lines(p)[0] == "alpha line"


# ----------------------------------------------------------------- properties
def test_chunking_is_deterministic() -> None:
    text = "\n".join(f"def function_{i}():\n    return {i}\n" for i in range(200))
    cfg = ChunkingConfig(max_tokens=60, overlap_tokens=10)
    assert chunk_document(text, cfg) == chunk_document(text, cfg)


def test_ordinals_are_dense_and_increasing() -> None:
    text = "\n".join(f"line number {i} with some padding text" for i in range(1, 500))
    chunks = chunk_document(text, ChunkingConfig(max_tokens=30, overlap_tokens=5))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_line_ranges_always_make_forward_progress() -> None:
    text = "\n".join(f"line number {i} with some padding text" for i in range(1, 500))
    chunks = chunk_document(text, ChunkingConfig(max_tokens=30, overlap_tokens=5))
    assert chunks[0].start_line == 1
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.start_line > prev.start_line, (prev, cur)
        assert cur.start_line <= prev.end_line + 1 or cur.start_line <= prev.end_line


def test_chunks_respect_the_token_budget_except_for_unbreakable_lines() -> None:
    text = "\n".join(f"filler sentence number {i} with a reasonable amount of text" for i in range(400))
    cfg = ChunkingConfig(max_tokens=40, overlap_tokens=5, min_chunk_tokens=0)
    chunks = chunk_document(text, cfg)
    assert chunks
    for ch in chunks:
        # a chunk may exceed the budget only when it is a single line
        if ch.start_line != ch.end_line:
            assert ch.token_estimate <= cfg.max_tokens + 8, ch.token_estimate


def test_token_estimate_matches_the_documented_formula() -> None:
    text = "a" * 40 + "\n" + "b" * 40
    cfg = ChunkingConfig(max_tokens=400, tokens_per_char=4.0)
    chunks = chunk_document(text, cfg)
    assert len(chunks) == 1
    assert chunks[0].token_estimate == estimate_tokens(chunks[0].text, 4.0)


# ------------------------------------------------------- real files on disk
@pytest.mark.parametrize(
    "name",
    [
        "alpha.md",
        "beta.txt",
        "gamma.py",
        "delta.rs",
        "epsilon.json",
        "zeta.sql",
        "eta.yaml",
        "theta.go",
        "iota.js",
        "kappa.css",
    ],
)
def test_line_ranges_are_real_for_real_files(tmp_path: Path, name: str) -> None:
    """Write a multi-section file, chunk it, and re-read every cited range."""
    parts = []
    for i in range(1, 121):
        if i % 5 == 0:
            parts.append(f"# Section {i}")
        else:
            parts.append(f"prose line {i} describing topic {i % 7} in detail")
    text = "\n".join(parts) + "\n"
    p = write(tmp_path, name, text)
    on_disk = p.read_text(encoding="utf-8")

    cfg = ChunkingConfig(max_tokens=50, overlap_tokens=8, min_chunk_tokens=5)
    chunks = chunk_document(on_disk, cfg)
    assert len(chunks) > 1, "fixture must produce multiple chunks to be meaningful"

    lines = read_lines(p)
    for ch in chunks:
        assert ch.end_line <= len(lines)
        region = "\n".join(lines[ch.start_line - 1 : ch.end_line])
        assert ch.text in region, (
            f"{name} chunk {ch.ordinal}: cited [{ch.start_line}-{ch.end_line}] "
            f"does not contain {ch.text[:60]!r}"
        )
        stripped = ch.text.strip()
        if stripped:
            idx = on_disk.index(stripped)
            first = on_disk[:idx].count("\n") + 1
            last = first + stripped.count("\n")
            assert ch.start_line <= first <= last <= ch.end_line


def test_line_provenance_survives_a_blank_line_leading_chunk(tmp_path: Path) -> None:
    """Chunks whose first line is blank still cite a range that contains them.

    Documenting real behaviour: the chunker's ``start_line`` points at the first
    line of its slice, which may be an empty line; the first line carrying
    *content* can therefore be later than ``start_line`` but never earlier.
    """
    text = "\n".join(f"content line {i}" for i in range(1, 40))
    cfg = ChunkingConfig(max_tokens=12, overlap_tokens=2, min_chunk_tokens=0)
    chunks = chunk_document(text, cfg)
    _assert_line_provenance(text, chunks)


def test_offsets_are_ascending() -> None:
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_document(text, ChunkingConfig(max_tokens=25, overlap_tokens=5))
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.start_offset >= prev.start_offset
        assert cur.end_offset >= cur.start_offset


def test_unicode_content_is_preserved_verbatim(tmp_path: Path) -> None:
    text = "héllo wörld\nこんにちは世界\nΩ≈ç√∫˜µ≤≥÷\n"
    p = write(tmp_path, "unicode.txt", text)
    chunks = chunk_document(p.read_text(encoding="utf-8"), ChunkingConfig(max_tokens=6, overlap_tokens=1))
    _assert_line_provenance(text, chunks)
    joined = "\n".join(c.text for c in chunks)
    assert "こんにちは世界" in joined


def test_boundary_respecting_flags_can_be_disabled() -> None:
    text = "\n\ndef a():\n\ndef b():\n\ndef c():\n"
    a = chunk_document(text, ChunkingConfig(max_tokens=8, overlap_tokens=1, min_chunk_tokens=0,
                                           respect_blank_lines=True, respect_symbol_boundaries=True))
    b = chunk_document(text, ChunkingConfig(max_tokens=8, overlap_tokens=1, min_chunk_tokens=0,
                                           respect_blank_lines=False, respect_symbol_boundaries=False))
    _assert_line_provenance(text, a)
    _assert_line_provenance(text, b)


def test_metadata_records_hard_cut_flag() -> None:
    text = "\n".join(f"x{i}" for i in range(100))
    chunks = chunk_document(text, ChunkingConfig(max_tokens=10, overlap_tokens=1, min_chunk_tokens=0))
    assert all("hard_cut" in c.metadata for c in chunks)
    assert all(isinstance(c.metadata["hard_cut"], bool) for c in chunks)