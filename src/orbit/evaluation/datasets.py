"""Benchmark dataset loading.

A benchmark is only as good as its ground truth, so this module is deliberately
pedantic:

* a line range must be a pair of 1-based inclusive integers;
* ``relevant_paths`` and ``relevant_line_ranges`` must line up one-to-one (or
  name a single file that owns every range);
* :func:`verify_cases` re-reads the corpus and reports any label that points
  outside a real file, and any label whose lines are blank. A label that does
  not contain the answer is a defect, so it is surfaced rather than averaged in.

Two relevance modes
-------------------
``chunk level``
    ``relevant_chunks`` holds ``chunk_id`` values. Precise, but only valid for
    the chunking configuration that produced them. Reading them back from a real
    index (rather than inventing them) is what keeps this mode honest.

``location level``
    ``relevant_paths`` + ``relevant_line_ranges``. A retrieved hit is relevant
    when it comes from the same file *and* its line range overlaps a gold range.
    This survives a change to the chunker, so it is the number to trust when
    comparing two builds of Orbit.

A case may carry either or both. A run reports both numbers and says which basis
each one used; it never silently substitutes one for the other.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DatasetError",
    "BenchmarkCase",
    "load_cases",
    "parse_case",
    "verify_cases",
    "normalise_path",
    "location_key",
]

from .metrics import location_key as _location_key
from .metrics import normalise_path as _normalise_path


class DatasetError(ValueError):
    """The dataset is malformed. Raised loudly, never swallowed."""


def normalise_path(path: str) -> str:
    """Re-exported from :mod:`orbit.evaluation.metrics` for convenience."""
    return _normalise_path(path)


def location_key(path: str, start: int, end: int) -> str:
    """Re-exported from :mod:`orbit.evaluation.metrics` for convenience."""
    return _location_key(path, start, end)


@dataclass(frozen=True)
class BenchmarkCase:
    """One labelled question.

    ``relevant_paths`` and ``relevant_line_ranges`` are parallel lists. Either
    ``relevant_paths[i]`` owns ``relevant_line_ranges[i]``, or there is exactly one
    path and it owns every range.
    """

    id: str
    question: str
    relevant_paths: tuple[str, ...] = ()
    relevant_line_ranges: tuple[tuple[int, int], ...] = ()
    relevant_chunks: tuple[str, ...] = ()
    difficulty: str = "easy"
    notes: str = ""
    graded: Mapping[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def gold_locations(self) -> list[tuple[str, int, int]]:
        """``[(normalised_path, start, end), ...]`` for every gold location."""
        out: list[tuple[str, int, int]] = []
        for path, rng in zip(self.relevant_paths, self.relevant_line_ranges):
            out.append((normalise_path(path), int(rng[0]), int(rng[1])))
        return out

    @property
    def gold_files(self) -> list[str]:
        seen: list[str] = []
        for p in self.relevant_paths:
            n = normalise_path(p)
            if n not in seen:
                seen.append(n)
        return seen

    @property
    def has_chunk_labels(self) -> bool:
        return bool(self.relevant_chunks)

    @property
    def has_location_labels(self) -> bool:
        return bool(self.relevant_paths and self.relevant_line_ranges)

    def location_key(self) -> str:
        """Single-line human label, e.g. ``fusion.md:18-22`` (first gold loc)."""
        locs = self.gold_locations
        if not locs:
            return "<no location label>"
        return location_key(*locs[0])

    def contains_location(self, path: str, start: int, end: int) -> bool:
        """Overlap test against every gold location."""
        p = normalise_path(path)
        lo, hi = (start, end) if start <= end else (end, start)
        for gp, gs, ge in self.gold_locations:
            if gp != p:
                continue
            if min(gs, ge) <= hi and lo <= max(gs, ge):
                return True
        return False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "relevant_paths": list(self.relevant_paths),
            "relevant_line_ranges": [list(r) for r in self.relevant_line_ranges],
            "relevant_chunks": list(self.relevant_chunks),
            "difficulty": self.difficulty,
            "notes": self.notes,
            "graded": dict(self.graded),
        }


def _as_pair(value: object, case_id: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DatasetError(
            f"case {case_id!r}: line range must be a [start, end] pair, got {value!r}"
        )
    try:
        start, end = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"case {case_id!r}: non-integer line range {value!r}") from exc
    if start < 1:
        raise DatasetError(f"case {case_id!r}: line numbers are 1-based, got {value!r}")
    if start > end:
        raise DatasetError(
            f"case {case_id!r}: line range start {start} is after end {end}"
        )
    return start, end


def parse_case(raw: object, *, index: int = 0) -> BenchmarkCase:
    """Build a :class:`BenchmarkCase` from one JSON object."""
    if not isinstance(raw, Mapping):
        raise DatasetError(f"question #{index} is not a JSON object")
    case_id = str(raw.get("id") or f"q{index + 1:02d}")
    question = str(raw.get("question") or "").strip()
    if not question:
        raise DatasetError(f"case {case_id!r}: missing 'question'")

    paths = tuple(str(p) for p in (raw.get("relevant_paths") or ()))
    ranges_raw = raw.get("relevant_line_ranges") or ()
    ranges = tuple(_as_pair(r, case_id) for r in ranges_raw)

    if paths and ranges and len(paths) not in (1, len(ranges)):
        raise DatasetError(
            f"case {case_id!r}: relevant_paths has {len(paths)} entries but "
            f"relevant_line_ranges has {len(ranges)}; they must pair up, or a "
            f"single path must own every range"
        )
    if len(ranges) > 1 and len(paths) == 1:
        paths = paths * len(ranges)
    if len(paths) != len(ranges):
        raise DatasetError(
            f"case {case_id!r}: every relevant path needs a line range "
            f"({len(paths)} paths, {len(ranges)} ranges)"
        )
    if not paths and not (raw.get("relevant_chunk_ids") or raw.get("relevant_chunks")):
        raise DatasetError(
            f"case {case_id!r}: no relevance labels "
            f"(need relevant_paths+relevant_line_ranges or relevant_chunk_ids)"
        )

    chunk_ids = tuple(
        str(c) for c in (raw.get("relevant_chunk_ids") or raw.get("relevant_chunks") or ())
    )

    graded_raw = raw.get("graded") or {}
    graded: dict[str, int] = {}
    if graded_raw and not isinstance(graded_raw, Mapping):
        raise DatasetError(f"case {case_id!r}: 'graded' must be an object")
    for key, value in (graded_raw or {}).items():
        try:
            graded[str(key)] = int(value)
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"case {case_id!r}: bad grade for {key!r}: {value!r}") from exc

    return BenchmarkCase(
        id=case_id,
        question=question,
        relevant_paths=paths,
        relevant_line_ranges=ranges,
        relevant_chunks=chunk_ids,
        difficulty=str(raw.get("difficulty") or "easy"),
        notes=str(raw.get("notes") or ""),
        graded=graded,
    )


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    """Read a ``questions.json`` file into :class:`BenchmarkCase` objects."""
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"questions file not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{p} is not valid JSON: {exc}") from exc

    if isinstance(raw, Mapping):
        raw = raw.get("questions", raw.get("cases"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DatasetError(f"{p} must contain a JSON array of questions")

    cases = [parse_case(item, index=i) for i, item in enumerate(raw)]

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"duplicate question id {case.id!r}")
        seen.add(case.id)
    return cases


def verify_cases(
    cases: Sequence[BenchmarkCase], corpus_root: str | Path
) -> list[dict]:
    """Check every location label against the files on disk.

    Returns one report dict per case::

        {"id", "ok", "problems": [...], "locations": [...]}

    A problem is raised when the file is missing, the range runs past end of
    file, the range selects only blank lines, or the file simply is not in the
    corpus. Any problem means the *label* is suspect; the benchmark still runs,
    but the runner prints the report rather than hiding it.
    """
    root = Path(corpus_root)
    reports: list[dict] = []
    for case in cases:
        problems: list[str] = []
        locations: list[dict] = []
        for path, (start, end) in zip(case.relevant_paths, case.relevant_line_ranges):
            fp = root / path
            entry = {"path": path, "start": start, "end": end, "ok": False, "chars": 0}
            if not fp.is_file():
                problems.append(f"{path}: file not found under corpus root {root}")
                locations.append(entry)
                continue
            try:
                lines = fp.read_text(encoding="utf-8").split("\n")
            except (OSError, UnicodeDecodeError) as exc:
                problems.append(f"{path}: unreadable ({exc})")
                locations.append(entry)
                continue
            if end > len(lines):
                problems.append(
                    f"{path}: lines {start}-{end} run past end of file ({len(lines)} lines)"
                )
                locations.append(entry)
                continue
            body = "\n".join(lines[start - 1 : end])
            entry["chars"] = len(body.strip())
            if not body.strip():
                problems.append(f"{path}: lines {start}-{end} contain only blank lines")
            else:
                entry["ok"] = True
                entry["preview"] = body.strip().split("\n")[0][:120]
            locations.append(entry)
        reports.append(
            {"id": case.id, "ok": not problems, "problems": problems, "locations": locations}
        )
    return reports