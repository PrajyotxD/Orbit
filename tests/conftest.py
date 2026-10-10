"""Shared pytest fixtures for the ORBIT QA suite.

Design rules obeyed here:
  * every filesystem artefact lives under the per-test ``tmp_path`` fixture;
  * nothing is ever written inside the repository;
  * the default (hashing) embedder is used everywhere, so the whole suite is
    offline, deterministic and needs no API key;
  * no network access is ever performed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from orbit import OrbitConfig
from orbit.engine import OrbitEngine


# --------------------------------------------------------------------------
# filesystem helpers
# --------------------------------------------------------------------------
def write(root: Path, rel: str, content: str | bytes, *, encoding: str = "utf-8") -> Path:
    """Write ``content`` at ``root/rel``, creating parents. Returns the path."""
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding=encoding, newline="")
    return path


def read_lines(path: Path) -> list[str]:
    """Read a file the same way Orbit's validator does (CRLF normalised)."""
    live = Path(path).read_text(encoding="utf-8", errors="replace")
    live = live.replace("\r\n", "\n").replace("\r", "\n")
    return live.split("\n")


def tree_snapshot(root: Path) -> dict[str, str]:
    """Map of relative path -> sha256 for every file under ``root``."""
    out: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            rel = fp.relative_to(root).as_posix()
            out[rel] = hashlib.sha256(fp.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------
# corpus builders
# --------------------------------------------------------------------------
#: Distinct, low-frequency marker words so lexical tests are not ambiguous.
DOC_ALPHA = (
    "# Alpha Guide\n\n"
    "The alpha subsystem coordinates quaternion scheduling across the cluster.\n"
    "It retries idempotent operations and records latency histograms.\n\n"
    "## Tuning\n\n"
    "Set scheduler_threads to a value that divides evenly into the ring size.\n"
)

DOC_BETA = (
    "Beta notes about vectorised storage.\n"
    "Row-group pruning reduces the number of segments scanned per query.\n"
    "Compaction rewrites small segments once they exceed the threshold.\n"
)

DOC_GAMMA = (
    "def schedule_ring(threads: int) -> int:\n"
    "    \"\"\"Return a scheduler width.\"\"\"\n"
    "    return threads\n\n\n"
    "class QuatAllocator:\n"
    "    def allocate(self):\n"
    "        return None\n"
)

DOC_DELTA = (
    "# Delta\n\n"
    "Telemetry export writes newline delimited json to object storage.\n"
    "Sampling reduces cardinality before the encoder runs.\n"
)

DEFAULT_CORPUS: dict[str, str] = {
    "alpha.md": DOC_ALPHA,
    "notes/beta.txt": DOC_BETA,
    "src/gamma.py": DOC_GAMMA,
    "ops/delta.md": DOC_DELTA,
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A four-document corpus at ``tmp_path/corpus``."""
    root = tmp_path / "corpus"
    root.mkdir()
    for rel, text in DEFAULT_CORPUS.items():
        write(root, rel, text)
    return root


@pytest.fixture
def empty_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "empty"
    root.mkdir()
    return root


# --------------------------------------------------------------------------
# engine helpers
# --------------------------------------------------------------------------
@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "orbit.db"


def make_config(db_path: Path, **overrides) -> OrbitConfig:
    """Build an ``OrbitConfig`` pointed at ``db_path`` with the offline default."""
    cfg = OrbitConfig(db_path=str(db_path))
    if "embedding_provider" in overrides:
        cfg = cfg.with_embedding_provider(overrides.pop("embedding_provider"))
    if "paths" in overrides:
        cfg = cfg.with_paths(overrides.pop("paths"))
    if "exclude_paths" in overrides:
        cfg = replace(cfg, exclude_paths=tuple(overrides.pop("exclude_paths")))
    return cfg


@pytest.fixture
def make_engine(db_path: Path):
    """Factory returning open ``OrbitEngine`` instances bound to ``tmp_path``."""
    created: list[OrbitEngine] = []

    def _factory(**overrides) -> OrbitEngine:
        eng = OrbitEngine(make_config(db_path, **overrides))
        created.append(eng)
        return eng

    yield _factory
    for eng in created:
        try:
            eng.close()
        except Exception:
            pass


@pytest.fixture
def indexed_engine(make_engine, corpus: Path) -> OrbitEngine:
    """An engine that has already indexed the default corpus."""
    eng = make_engine()
    eng.index([corpus])
    return eng


# --------------------------------------------------------------------------
# environment hygiene
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch):
    """Make an accidental network call loud instead of silent.

    The default configuration is fully offline; if any code path ever tries to
    reach the network during the suite we want a failure, not a hang.
    """
    import socket

    real_socket = socket.socket

    class _BlockedSocket(real_socket):  # type: ignore[misc, valid-type]
        def connect(self, *a, **kw):  # pragma: no cover - only on regression
            raise AssertionError(f"network access attempted: {a!r}")

        def connect_ex(self, *a, **kw):  # pragma: no cover - only on regression
            raise AssertionError(f"network access attempted: {a!r}")

    monkeypatch.setattr(socket, "socket", _BlockedSocket)
    yield