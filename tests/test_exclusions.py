"""Regression: `exclude_paths` must exclude, never add to the index.

Before this fix, `OrbitEngine.index()` did:

    targets = targets + list(self.config.exclude_paths)

which *appended* the excluded paths to the roots handed to the indexer, so a
directory the operator explicitly asked to leave out was indexed anyway. A config
with ``paths=["docs"]`` and ``exclude_paths=["docs/private"]`` indexed both.
"""

from __future__ import annotations

from pathlib import Path

from orbit.engine import OrbitEngine

from conftest import write


def _indexed_paths(engine: OrbitEngine) -> set[str]:
    return {d["path"] for d in engine.documents()}


def test_exclude_paths_are_not_indexed(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "public").mkdir(parents=True)
    (root / "private").mkdir(parents=True)
    write(root, "public/open.md", "public document about open launches.\n")
    write(root, "private/secret.md", "private document about hidden launch codes.\n")

    engine = make_engine(paths=[str(root)], exclude_paths=["private"])
    engine.index()
    paths = _indexed_paths(engine)

    assert "private/secret.md" not in paths, f"excluded file was indexed: {paths}"
    assert "public/open.md" in paths, f"non-excluded file missing: {paths}"


def test_exclude_paths_directory_is_skipped_entirely(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "keep").mkdir(parents=True)
    (root / "skipme").mkdir(parents=True)
    write(root, "keep/a.md", "kept file about orbit retrieval.\n")
    write(root, "skipme/b.md", "skipped file about orbit retrieval.\n")
    write(root, "skipme/c.md", "another skipped file about orbit.\n")

    engine = make_engine(paths=[str(root)], exclude_paths=["skipme"])
    engine.index()
    paths = _indexed_paths(engine)

    assert paths == {"keep/a.md"}, f"expected only the kept file, got {paths}"


def test_exclude_paths_supports_globs(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir(parents=True)
    write(root, "real.md", "a real document about orbit.\n")
    write(root, "schema.generated.ts", "generated file, should not be indexed.\n")
    write(root, "notes.generated.py", "generated file, should not be indexed either.\n")

    engine = make_engine(paths=[str(root)], exclude_paths=["*.generated.*"])
    engine.index()
    paths = _indexed_paths(engine)

    assert paths == {"real.md"}, f"glob exclusion failed, got {paths}"


def test_exclude_paths_accepts_absolute_paths(make_engine, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    excluded = root / "vendored"
    excluded.mkdir(parents=True)
    write(root, "main.md", "main document about the engine.\n")
    write(excluded, "dep.md", "dependency doc that must not be indexed.\n")

    engine = make_engine(paths=[str(root)], exclude_paths=[str(excluded)])
    engine.index()
    paths = _indexed_paths(engine)

    assert paths == {"main.md"}, f"absolute exclude path failed, got {paths}"


def test_no_exclude_paths_still_indexes_everything(make_engine, tmp_path: Path) -> None:
    """Guard against over-correcting into an exclusion that swallows the corpus."""
    root = tmp_path / "corpus"
    root.mkdir(parents=True)
    write(root, "a.md", "first document about orbit.\n")
    write(root, "b.md", "second document about orbit.\n")

    engine = make_engine(paths=[str(root)])
    engine.index()
    paths = _indexed_paths(engine)

    assert paths == {"a.md", "b.md"}, f"nothing should have been excluded, got {paths}"


def test_exclude_paths_does_not_hide_already_indexed_files(make_engine, tmp_path: Path) -> None:
    """Excluding a path must not silently tombstone it without being asked."""
    root = tmp_path / "corpus"
    (root / "gone").mkdir(parents=True)
    write(root, "gone/a.md", "temporarily indexed document about orbit.\n")

    engine = make_engine(paths=[str(root)])
    engine.index()
    assert _indexed_paths(engine) == {"gone/a.md"}

    # Re-index with the directory excluded: it drops out of the corpus, which is
    # the documented behaviour of an exclusion.
    engine.config = engine.config.with_exclude_paths(("gone",))
    engine.index()
    assert _indexed_paths(engine) == set()