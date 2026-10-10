"""Benchmark runner.

Runs a labelled question set against a live :class:`~orbit.engine.OrbitEngine`
in several retrieval modes and reports, per mode:

* recall@k, precision@k, hit-rate@k, MRR and nDCG@k at **both** relevance
  levels (chunk-id and source-location);
* citation-location accuracy — the Orbit-specific question "did the retrieved
  passage point at the right place in the right file?";
* per-query wall-clock latency (mean / p50 / p95);
* optionally, index throughput on a throwaway database.

Configuration isolation
-----------------------
:class:`~orbit.config.OrbitConfig` is a frozen dataclass, and nothing here mutates
the caller's engine. Each mode gets a *new* config built with
:func:`dataclasses.replace` and its own :class:`~orbit.retrieval.Retriever` that
shares the engine's read-only database handle. After a run,
``engine.config`` is still the object it was before.

Honesty rules
-------------
1. A mode whose channel did not run is reported as ``UNAVAILABLE`` with the
   reason. It gets no numbers, ever.
2. A mode that ran but silently lost a channel is reported as ``DEGRADED`` with
   the reason, and its numbers are labelled as degraded.
3. Missing measurements are ``None``, which :func:`~orbit.evaluation.metrics.aggregate`
   excludes from means. They are never counted as zero.
"""

from __future__ import annotations

import platform
import sqlite3
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import OrbitConfig
from .datasets import BenchmarkCase, location_key, normalise_path, verify_cases
from .metrics import (
    aggregate,
    citation_location_accuracy,
    hit_rate_at_k,
    hit_rate_at_k_relevant,
    latency_summary,
    mrr,
    mrr_relevant,
    ndcg_at_k,
    ndcg_at_k_relevant,
    percentile,
    precision_at_k,
    precision_at_k_relevant,
    ranges_overlap,
    recall_at_k,
    recall_at_k_relevant,
)

__all__ = [
    "MODES",
    "run_benchmark",
    "measure_index_throughput",
    "format_report",
    "environment_block",
]

MODES: tuple[str, ...] = ("lexical", "vector", "hybrid")

_STATUS_OK = "ok"
_STATUS_UNAVAILABLE = "UNAVAILABLE"
_STATUS_DEGRADED = "DEGRADED"


# ---
# environment
# ---
def _orbit_version() -> str:
    try:  # installed as a distribution
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("orbit-rag")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("version"):
                return s.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "unknown"


def environment_block(config: OrbitConfig, stats: Any = None) -> dict[str, Any]:
    """Everything needed to reproduce a run on another machine."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "orbit_version": _orbit_version(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": _cpu_count(),
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_module_version": sqlite3.version,
        "embedding_provider": getattr(stats, "embedding_provider", None)
        or config.embedding.provider,
        "embedding_dimension": getattr(stats, "embedding_dim", None)
        or config.embedding.dimension,
        "embedding_model": config.embedding.model if config.embedding.provider == "hf" else "",
        "db_path": getattr(stats, "db_path", None) or config.db_path,
        "chunking": {
            "max_tokens": config.chunking.max_tokens,
            "overlap_tokens": config.chunking.overlap_tokens,
            "max_chars": config.chunking.max_chars_effective,
            "tokens_per_char": config.chunking.tokens_per_char,
            "min_chunk_tokens": config.chunking.min_chunk_tokens,
            "respect_blank_lines": config.chunking.respect_blank_lines,
            "respect_symbol_boundaries": config.chunking.respect_symbol_boundaries,
        },
        "fusion": {
            "method": config.retrieval.fusion,
            "rrf_k": config.retrieval.rrf_k,
            "lexical_weight": config.retrieval.lexical_weight,
            "vector_weight": config.retrieval.vector_weight,
            "lexical_k1": config.retrieval.lexical_k1,
            "lexical_b": config.retrieval.lexical_b,
            "candidate_multiplier": config.retrieval.candidate_multiplier,
            "dedupe": config.retrieval.dedupe,
        },
    }


def _cpu_count() -> int:
    import os

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 0


# ---
# mode machinery
# ---
_SEARCH_MODES = {
    "lexical": ("lexical",),
    "vector": ("vector",),
    "hybrid": ("lexical", "vector"),
}

_MODE_LABEL = {
    "lexical": "lexical only (FTS5 BM25)",
    "vector": "vector only (exact scan)",
    "hybrid": "hybrid (lexical + vector, fused)",
}


def _mode_config(base: OrbitConfig, limit: int) -> OrbitConfig:
    """A NEW frozen config for a benchmark run. The caller's config is untouched."""
    return replace(
        base,
        retrieval=replace(base.retrieval, limit=int(limit)),
    )


def _vector_usable(stats: Any) -> tuple[bool, str]:
    provider = (getattr(stats, "embedding_provider", "") or "").strip().lower()
    count = int(getattr(stats, "embeddings", 0) or 0)
    if provider in ("none", ""):
        return False, (
            f"no embedding provider is configured (index reports provider="
            f"{provider or 'unset'!r}); the index was built with embedding.provider=none"
        )
    if count <= 0:
        return False, (
            f"the index contains 0 embeddings (provider={provider!r}); "
            f"the vector channel has nothing to scan"
        )
    return True, ""


def _latency_key(name: str, k: int) -> str:
    return f"{name}@{k}"


# ---
# main entry point
# ---
def run_benchmark(
    engine: Any,
    cases: Sequence[BenchmarkCase],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    modes: Sequence[str] = ("lexical", "vector", "hybrid"),
    repeats: int = 1,
    validate: bool = False,
    measure_index: bool = False,
    corpus: str | Path | Sequence[str | Path] | None = None,
    corpus_root: str | Path | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Run ``cases`` against ``engine`` in each of ``modes``.

    Parameters
    ----------
    engine
        An initialised :class:`~orbit.engine.OrbitEngine`. Its config is read,
        never written.
    cases
        Labelled questions from :func:`~orbit.evaluation.datasets.load_cases`.
    k_values
        Cut-offs to report recall/precision/hit-rate/nDCG at. Sorted, deduped,
        all >= 1.
    modes
        Any subset of ``("lexical", "vector", "hybrid")``.
    repeats
        How many times each query is issued. Metrics are computed from the first
        repetition's ranking (retrieval is deterministic); latency samples from
        every repetition are pooled.
    measure_index
        Also time a cold index build of ``corpus`` on a throwaway database.
        Slow, and therefore opt-in.
    """
    ks = sorted({int(k) for k in k_values if int(k) >= 1}) or [1, 3, 5, 10]
    mode_list = [m for m in modes if m in _SEARCH_MODES]
    unknown = [m for m in modes if m not in _SEARCH_MODES]
    repeats = max(1, int(repeats))

    base_config: OrbitConfig = engine.config
    stats_before = engine.stats()
    vector_ok, vector_reason = _vector_usable(stats_before)

    bench_config = _mode_config(base_config, max(ks))
    max_k = max(ks)

    ground_truth = verify_cases(cases, corpus_root) if corpus_root else None

    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "k_values": ks,
        "modes_requested": list(modes),
        "unknown_modes_ignored": unknown,
        "repeats": repeats,
        "validate": validate,
        "n_cases": len(cases),
        "environment": environment_block(bench_config, stats_before),
        "engine": {
            "db_path": str(getattr(stats_before, "db_path", bench_config.db_path)),
            "documents_current": int(getattr(stats_before, "documents_current", 0) or 0),
            "chunks": int(getattr(stats_before, "chunks", 0) or 0),
            "embeddings": int(getattr(stats_before, "embeddings", 0) or 0),
            "embedding_provider": getattr(stats_before, "embedding_provider", ""),
            "embedding_dim": int(getattr(stats_before, "embedding_dim", 0) or 0),
            "languages": dict(getattr(stats_before, "languages", {}) or {}),
        },
        "labels": {
            "chunk_level_available": any(c.has_chunk_labels for c in cases),
            "location_level_available": any(c.has_location_labels for c in cases),
            "cases_with_chunk_labels": sum(1 for c in cases if c.has_chunk_labels),
            "cases_with_location_labels": sum(1 for c in cases if c.has_location_labels),
        },
        "ground_truth": ground_truth,
        "results": {},
    }

    db = engine.db
    shared_embedder = getattr(getattr(engine, "indexer", None), "_embedder", None)

    for mode in mode_list:
        out["results"][mode] = _run_mode(
            engine=engine,
            db=db,
            config=bench_config,
            shared_embedder=shared_embedder,
            cases=cases,
            mode=mode,
            ks=ks,
            max_k=max_k,
            repeats=repeats,
            validate=validate,
            vector_ok=vector_ok,
            vector_reason=vector_reason,
        )

    if measure_index:
        if corpus is None:
            out["index_throughput"] = {
                "measured": False,
                "reason": "measure_index=True but no corpus path was supplied",
            }
        else:
            out["index_throughput"] = measure_index_throughput(
                bench_config, corpus, k_values=ks
            )
    else:
        out["index_throughput"] = {"measured": False, "reason": "not requested"}

    # The caller's engine must be exactly as we found it.
    out["engine_config_unchanged"] = engine.config is base_config

    after = engine.stats()
    out["engine"]["documents_after"] = int(getattr(after, "documents_current", 0) or 0)
    out["engine"]["chunks_after"] = int(getattr(after, "chunks", 0) or 0)
    out["engine"]["index_mutated_by_benchmark"] = (
        out["engine"]["documents_after"] != out["engine"]["documents_current"]
        or out["engine"]["chunks_after"] != out["engine"]["chunks"]
    )

    return out


def _run_mode(
    *,
    engine: Any,
    db: Any,
    config: OrbitConfig,
    shared_embedder: Any,
    cases: Sequence[BenchmarkCase],
    mode: str,
    ks: Sequence[int],
    max_k: int,
    repeats: int,
    validate: bool,
    vector_ok: bool,
    vector_reason: str,
) -> dict[str, Any]:
    label = _MODE_LABEL[mode]
    needs_vector = mode in ("vector", "hybrid")

    if needs_vector and not vector_ok:
        if mode == "vector":
            return {
                "mode": mode,
                "label": label,
                "status": _STATUS_UNAVAILABLE,
                "available": False,
                "degraded": False,
                "reason": vector_reason,
                "search_modes": list(_SEARCH_MODES[mode]),
                "metrics": None,
                "per_query": [],
                "latency_ms": latency_summary([]),
                "candidates": {"mean": 0.0, "mean_observed": 0.0},
            }
        # hybrid still runs, but only the lexical channel can contribute.

    from ..retrieval.retriever import Retriever

    retriever = Retriever(db, config, embedder=shared_embedder)
    search_modes = _SEARCH_MODES[mode]

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    degraded_reasons: list[str] = []
    channels_seen: set[str] = set()
    method_counts: dict[str, int] = {}
    candidates: list[int] = []
    vector_failures: list[str] = []

    for case in cases:
        result = None
        for _ in range(repeats):
            start = time.perf_counter()
            result = retriever.search(
                case.question, limit=max_k, modes=search_modes, validate=validate
            )
            latencies.append((time.perf_counter() - start) * 1000.0)

        assert result is not None
        hits = list(result.hits)
        channels_seen.update(result.channels_used)
        method_key = getattr(result.retrieval_method, "value", str(result.retrieval_method))
        method_counts[method_key] = method_counts.get(method_key, 0) + 1
        candidates.append(int(result.total_candidates))

        if result.degraded_reason:
            degraded_reasons.append(result.degraded_reason)
            if "vector channel unavailable" in result.degraded_reason:
                vector_failures.append(result.degraded_reason)
        if "vector" in search_modes and result.degraded_reason and "vector channel unavailable" not in result.degraded_reason:
            # e.g. "lexical channel returned nothing" / "no results" — not a
            # missing channel, just an empty one. Recorded, not treated as an outage.
            pass

        chunk_ids = [h.evidence.chunk_id for h in hits]
        loc_keys = [
            location_key(h.evidence.source_path, h.evidence.start_line, h.evidence.end_line)
            for h in hits
        ]
        paths = [normalise_path(h.evidence.source_path) for h in hits]
        spans = [(h.evidence.start_line, h.evidence.end_line) for h in hits]

        case_record: dict[str, Any] = {
            "id": case.id,
            "question": case.question,
            "difficulty": case.difficulty,
            "mode": mode,
            "n_hits": len(hits),
            "top_citation": hits[0].citation if hits else "",
            "retrieval_method": method_key,
            "channels_used": ",".join(result.channels_used),
            "candidates": int(result.total_candidates),
            "degraded_reason": result.degraded_reason or None,
        }

        # --- location level -------------------------------------------------
        if case.has_location_labels:
            # Location-level relevance is an *overlap* property: a retrieved
            # chunk that spans the gold line range counts as a hit even though
            # its own range differs. Comparing "path:start-end" strings for
            # equality would report zero recall for a correct retrieval, so the
            # relevance vector is built with the overlap test instead.
            rel_flags: list[bool] = []
            rel_grades: list[float] = []
            for p, (s, e) in zip(paths, spans):
                hit = case.contains_location(p, s, e)
                rel_flags.append(hit)
                grade = 0.0
                if hit:
                    for loc in case.gold_locations:
                        gk = location_key(*loc)
                        grade = max(grade, float(case.graded.get(gk, 1)) if case.graded else 1.0)
                rel_grades.append(grade)
            total_gold = len(case.gold_locations)

            for k in ks:
                case_record[_latency_key("recall", k)] = recall_at_k_relevant(
                    rel_flags, total_gold, k
                )
                case_record[_latency_key("precision", k)] = precision_at_k_relevant(rel_flags, k)
                case_record[_latency_key("hit_rate", k)] = hit_rate_at_k_relevant(rel_flags, k)
                case_record[_latency_key("ndcg", k)] = ndcg_at_k_relevant(rel_grades, k)
            case_record["mrr_location"] = mrr_relevant(rel_flags)
            cla = citation_location_accuracy(hits, case)
            case_record["citation_location_accuracy"] = cla["accuracy"]
            case_record["citation_file_accuracy"] = cla["file_accuracy"]
            case_record["citation_top1_location"] = 1.0 if cla["top1_location_hit"] else 0.0
            case_record["citation_top1_file"] = 1.0 if cla["top1_file_hit"] else 0.0
        else:
            for k in ks:
                case_record[_latency_key("recall", k)] = None
                case_record[_latency_key("precision", k)] = None
                case_record[_latency_key("hit_rate", k)] = None
                case_record[_latency_key("ndcg", k)] = None
            case_record["mrr_location"] = None
            case_record["citation_location_accuracy"] = None
            case_record["citation_file_accuracy"] = None
            case_record["citation_top1_location"] = None
            case_record["citation_top1_file"] = None

        # --- chunk level ----------------------------------------------------
        if case.has_chunk_labels:
            gold_chunks = list(case.relevant_chunks)
            graded_chunks = {cid: float(case.graded.get(cid, 1)) for cid in gold_chunks}
            for k in ks:
                case_record[_latency_key("chunk_recall", k)] = recall_at_k(chunk_ids, gold_chunks, k)
                case_record[_latency_key("chunk_precision", k)] = precision_at_k(chunk_ids, gold_chunks, k)
                case_record[_latency_key("chunk_hit_rate", k)] = hit_rate_at_k(chunk_ids, gold_chunks, k)
                case_record[_latency_key("chunk_ndcg", k)] = ndcg_at_k(chunk_ids, graded_chunks, k)
            case_record["mrr_chunk"] = mrr(chunk_ids, gold_chunks)
        else:
            for k in ks:
                case_record[_latency_key("chunk_recall", k)] = None
                case_record[_latency_key("chunk_precision", k)] = None
                case_record[_latency_key("chunk_hit_rate", k)] = None
                case_record[_latency_key("chunk_ndcg", k)] = None
            case_record["mrr_chunk"] = None

        per_query.append(case_record)

    status = _STATUS_OK
    reason = ""
    degraded = False
    if vector_failures:
        status = _STATUS_UNAVAILABLE
        reason = vector_failures[0]
    elif mode == "hybrid" and not vector_ok:
        status = _STATUS_DEGRADED
        degraded = True
        reason = (
            f"vector channel did not run: {vector_reason}; "
            f"these numbers are the lexical channel only"
        )
    elif degraded_reasons:
        distinct = sorted(set(degraded_reasons))
        if len(distinct) > 1 or len(distinct[0]) > len("no results"):
            status = _STATUS_DEGRADED
            degraded = True
            reason = "; ".join(distinct[:3])

    mean = aggregate(per_query)
    metrics = {
        "location_level": {
            "recall": {str(k): mean.get(f"recall@{k}") for k in ks},
            "precision": {str(k): mean.get(f"precision@{k}") for k in ks},
            "hit_rate": {str(k): mean.get(f"hit_rate@{k}") for k in ks},
            "ndcg": {str(k): mean.get(f"ndcg@{k}") for k in ks},
            "mrr": mean.get("mrr_location"),
            "citation_location_accuracy": mean.get("citation_location_accuracy"),
            "citation_file_accuracy": mean.get("citation_file_accuracy"),
            "citation_top1_location": mean.get("citation_top1_location"),
            "citation_top1_file": mean.get("citation_top1_file"),
        },
        "chunk_level": {
            "recall": {str(k): mean.get(f"chunk_recall@{k}") for k in ks},
            "precision": {str(k): mean.get(f"chunk_precision@{k}") for k in ks},
            "hit_rate": {str(k): mean.get(f"chunk_hit_rate@{k}") for k in ks},
            "ndcg": {str(k): mean.get(f"chunk_ndcg@{k}") for k in ks},
            "mrr": mean.get("mrr_chunk"),
        },
    }

    return {
        "mode": mode,
        "label": label,
        "status": status,
        "available": status != _STATUS_UNAVAILABLE,
        "degraded": degraded,
        "reason": reason,
        "search_modes": list(search_modes),
        "channels_used": sorted(channels_seen),
        "retrieval_methods": method_counts,
        "metrics": None if status == _STATUS_UNAVAILABLE else metrics,
        "per_query": per_query,
        "latency_ms": latency_summary(latencies),
        "candidates": {
            "mean": statistics.fmean(candidates) if candidates else 0.0,
            "min": min(candidates) if candidates else 0,
            "max": max(candidates) if candidates else 0,
        },
    }


# ---
# index throughput
# ---
def measure_index_throughput(
    config: OrbitConfig,
    corpus: str | Path | Sequence[str | Path],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Time a cold index build on a throwaway database.

    The caller's engine and its database are never touched: a fresh
    :class:`~orbit.engine.OrbitEngine` is built against a temporary ``db_path``
    inside a temporary directory, which is removed afterwards.
    """
    import shutil
    import tempfile

    from ..engine import OrbitEngine

    targets = [corpus] if isinstance(corpus, (str, Path)) else list(corpus)
    tmpdir = tempfile.mkdtemp(prefix="orbit-bench-index-")
    tmpdb = str(Path(tmpdir) / "index.db")
    try:
        cfg = replace(config, db_path=tmpdb, paths=tuple(str(t) for t in targets))
        engine = OrbitEngine(cfg)
        try:
            start = time.perf_counter()
            report = engine.index([str(t) for t in targets])
            elapsed = time.perf_counter() - start
            stats = engine.stats()
            docs = max(1, int(report.scanned))
            chunks = int(report.chunks_written)
            return {
                "measured": True,
                "cold_build": True,
                "isolated_db": True,
                "corpus": [str(t) for t in targets],
                "duration_s": elapsed,
                "scanned_documents": int(report.scanned),
                "documents_added": int(report.added),
                "documents_updated": int(report.updated),
                "documents_failed": int(report.failed),
                "documents_skipped": int(report.skipped),
                "chunks_written": chunks,
                "embeddings_written": int(report.embeddings_written),
                "embeddings_total": int(getattr(stats, "embeddings", 0) or 0),
                "embedding_provider": getattr(stats, "embedding_provider", ""),
                "embedding_dim": int(getattr(stats, "embedding_dim", 0) or 0),
                "bytes": int(getattr(Path(tmpdb), "stat")().st_size)
                if Path(tmpdb).is_file()
                else 0,
                "docs_per_second": report.scanned / elapsed if elapsed > 0 else 0.0,
                "chunks_per_second": chunks / elapsed if elapsed > 0 else 0.0,
                "failures": [list(f) for f in report.failures],
                "skips": [list(s) for s in report.skips],
            }
        finally:
            engine.close()
    except Exception as exc:  # a failed cold build is a result, not a crash
        return {"measured": False, "reason": f"cold index build failed: {exc}"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---
# human-readable output
# ---
def _fmt(value: Any, width: int, *, pct: bool = False) -> str:
    if value is None:
        return "-".rjust(width)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value).rjust(width)
    if pct:
        return f"{v * 100:.{max(0, width - 3)}f}".rjust(width)
    return f"{v:.{max(0, width - 4)}f}".rjust(width)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), sep]
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def format_report(results: Mapping[str, Any]) -> str:
    """Render a full human-readable report from :func:`run_benchmark`."""
    lines: list[str] = []
    env = results.get("environment", {})
    eng = results.get("engine", {})
    ks = list(results.get("k_values", []))

    lines.append("=" * 78)
    lines.append("ORBIT RETRIEVAL BENCHMARK")
    lines.append("=" * 78)
    lines.append(f"generated (UTC)   : {env.get('timestamp_utc')}")
    lines.append(f"orbit version      : {env.get('orbit_version')}")
    lines.append(f"python             : {env.get('python')} ({env.get('python_implementation')})")
    lines.append(f"platform           : {env.get('platform')}")
    lines.append(f"cpu                : {env.get('processor')} / {env.get('cpu_count')} logical")
    lines.append(f"sqlite             : {sqlite3.sqlite_version}")
    lines.append(
        f"embedding provider : {env.get('embedding_provider')} "
        f"dim={env.get('embedding_dimension')}"
        + (f" model={env.get('embedding_model')}" if env.get("embedding_model") else "")
    )
    ck = env.get("chunking", {})
    lines.append(
        "chunking           : "
        f"max_tokens={ck.get('max_tokens')} overlap={ck.get('overlap_tokens')} "
        f"max_chars={ck.get('max_chars')} tokens_per_char={ck.get('tokens_per_char')}"
    )
    fu = env.get("fusion", {})
    lines.append(
        "fusion             : "
        f"{fu.get('method')} rrf_k={fu.get('rrf_k')} "
        f"w_lex={fu.get('lexical_weight')} w_vec={fu.get('vector_weight')} "
        f"cand_mult={fu.get('candidate_multiplier')}"
    )
    lines.append(
        f"index              : {eng.get('documents_current')} docs, {eng.get('chunks')} chunks, "
        f"{eng.get('embeddings')} embeddings ({eng.get('embedding_provider')})"
    )
    lines.append(
        f"cases              : {results.get('n_cases')}  repeats={results.get('repeats')}  "
        f"k={','.join(str(k) for k in ks)}"
    )
    labels = results.get("labels", {})
    lines.append(
        f"labels             : chunk-level on "
        f"{labels.get('cases_with_chunk_labels')}/{results.get('n_cases')} cases, "
        f"location-level on {labels.get('cases_with_location_labels')}/{results.get('n_cases')} cases"
    )
    lines.append(f"note               : {results.get('note') or '(none)'}")
    lines.append("")

    # ---- ground truth check ------------------------------------------------
    gt = results.get("ground_truth")
    if gt:
        bad = [g for g in gt if not g["ok"]]
        lines.append(
            f"GROUND TRUTH CHECK : {len(gt) - len(bad)}/{len(gt)} questions have "
            f"labels that resolve to real, non-blank lines in the corpus"
        )
        for g in bad:
            for problem in g["problems"]:
                lines.append(f"  !! {g['id']}: {problem}")
        lines.append("")

    # ---- mode status -------------------------------------------------------
    lines.append("MODE STATUS")
    lines.append("-" * 78)
    for mode, res in results.get("results", {}).items():
        status = res["status"]
        extra = ""
        if status == _STATUS_UNAVAILABLE:
            extra = f"  ->  {res['reason']}"
        elif status == _STATUS_DEGRADED:
            extra = f"  ->  {res['reason']}"
        lines.append(
            f"  {mode:<8} {status:<12} channels={','.join(res.get('channels_used') or []) or '-':<15}"
            f"{extra}"
        )
    lines.append("")

    for basis, title in (
        ("location_level", "LOCATION-LEVEL METRICS  (source_path + overlapping line range)"),
        ("chunk_level", "CHUNK-LEVEL METRICS  (chunk_id, valid only for this chunking config)"),
    ):
        block = results.get("labels", {})
        if basis == "chunk_level" and not block.get("chunk_level_available"):
            lines.append("CHUNK-LEVEL METRICS   : not computed - no case carries chunk-id labels")
            lines.append("")
            continue
        lines.append(title)
        lines.append("-" * 78)
        headers = ["mode", "status"] + [f"R@{k}" for k in ks] + [f"P@{k}" for k in ks] + [
            f"HR@{k}" for k in ks
        ] + [f"nDCG@{k}" for k in ks] + ["MRR"]
        rows = []
        for mode, res in results.get("results", {}).items():
            metrics = res.get("metrics")
            if not metrics:
                rows.append([mode, res["status"]] + ["-"] * (len(headers) - 2))
                continue
            m = metrics[basis]
            row = [mode, res["status"]]
            for name in ("recall", "precision", "hit_rate"):
                row += [_fmt(m[name].get(str(k)), 5, pct=True) for k in ks]
            row += [_fmt(m["ndcg"].get(str(k)), 6) for k in ks]
            row.append(_fmt(m["mrr"], 5))
            rows.append(row)
        lines.append(_table(headers, rows))
        lines.append("")

    # ---- citation location -------------------------------------------------
    lines.append("CITATION-LOCATION ACCURACY  (right file AND overlapping line range)")
    lines.append("-" * 78)
    headers = ["mode", "status", "CLA", "file-level", "top1 loc", "top1 file", "mean cand."]
    rows = []
    for mode, res in results.get("results", {}).items():
        metrics = res.get("metrics")
        if not metrics:
            rows.append([mode, res["status"], "-", "-", "-", "-", "-"])
            continue
        m = metrics["location_level"]
        rows.append(
            [
                mode,
                res["status"],
                _fmt(m["citation_location_accuracy"], 5, pct=True),
                _fmt(m["citation_file_accuracy"], 5, pct=True),
                _fmt(m["citation_top1_location"], 5, pct=True),
                _fmt(m["citation_top1_file"], 5, pct=True),
                _fmt(res["candidates"]["mean"], 5),
            ]
        )
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- latency -----------------------------------------------------------
    lines.append("QUERY LATENCY (ms, wall clock around engine call)")
    lines.append("-" * 78)
    headers = ["mode", "status", "n", "mean", "p50", "p95", "min", "max"]
    rows = []
    for mode, res in results.get("results", {}).items():
        lat = res["latency_ms"]
        rows.append(
            [
                mode,
                res["status"],
                str(lat["n"]),
                f"{lat['mean']:.3f}",
                f"{lat['p50']:.3f}",
                f"{lat['p95']:.3f}",
                f"{lat['min']:.3f}",
                f"{lat['max']:.3f}",
            ]
        )
    lines.append(_table(headers, rows))
    lines.append("")

    # ---- index throughput --------------------------------------------------
    lines.append("INDEX THROUGHPUT")
    lines.append("-" * 78)
    thr = results.get("index_throughput") or {}
    if not thr.get("measured"):
        lines.append(f"  not measured: {thr.get('reason', 'unknown')}")
    else:
        lines.append(f"  corpus                : {', '.join(thr['corpus'])}")
        lines.append(f"  cold build            : {thr['duration_s']:.3f} s on a throwaway db")
        lines.append(
            f"  documents             : {thr['scanned_documents']} scanned, "
            f"{thr['documents_added']} added, {thr['documents_updated']} updated, "
            f"{thr['documents_failed']} failed, {thr['documents_skipped']} skipped"
        )
        lines.append(
            f"  chunks / embeddings   : {thr['chunks_written']} / {thr['embeddings_written']} "
            f"({thr['embedding_provider']}, dim={thr['embedding_dim']})"
        )
        lines.append(f"  throughput            : {thr['docs_per_second']:.1f} docs/s, "
                     f"{thr['chunks_per_second']:.1f} chunks/s")
        lines.append(f"  db size               : {thr['bytes']} bytes")
    lines.append("")

    lines.append("INDEX ISOLATION")
    lines.append("-" * 78)
    lines.append(f"  engine.config object unchanged      : {results.get('engine_config_unchanged')}")
    lines.append(
        f"  benchmark mutated the engine index   : {results.get('engine', {}).get('index_mutated_by_benchmark')}"
    )
    lines.append("")
    return "\n".join(lines)