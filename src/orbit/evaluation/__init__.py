"""Evaluation harness for Orbit.

The evaluation package answers one question: *did retrieval find the right
passage, in the right place, in the right file?*

Three pieces, kept separable on purpose:

:mod:`orbit.evaluation.metrics`
    Pure functions. No engine, no database, no I/O. Unit-testable with nothing
    but lists of strings.

:mod:`orbit.evaluation.datasets`
    Loading and validating labelled questions, with an explicit ground-truth
    check against the corpus on disk.

:mod:`orbit.evaluation.benchmark`
    The runner. Owns the engine-facing concerns: mode isolation, latency,
    degraded channels, and never reporting a number for a channel that did not
    run.

Relevance is measured at two levels — chunk id and source location — and both
are always reported. See :mod:`orbit.evaluation.metrics` for why one of them is
not enough.
"""

from __future__ import annotations

from .benchmark import (
    MODES,
    environment_block,
    format_report,
    measure_index_throughput,
    run_benchmark,
)
from .datasets import (
    BenchmarkCase,
    DatasetError,
    load_cases,
    parse_case,
    verify_cases,
)
from .metrics import (
    aggregate,
    citation_location_accuracy,
    hit_rate_at_k,
    latency_summary,
    location_key,
    mean_reciprocal_rank,
    mrr,
    ndcg_at_k,
    normalise_path,
    percentile,
    precision_at_k,
    ranges_overlap,
    recall_at_k,
)

__all__ = [
    # metrics
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "hit_rate_at_k",
    "citation_location_accuracy",
    "aggregate",
    "latency_summary",
    "percentile",
    "normalise_path",
    "location_key",
    "ranges_overlap",
    # datasets
    "BenchmarkCase",
    "DatasetError",
    "load_cases",
    "parse_case",
    "verify_cases",
    # benchmark
    "MODES",
    "run_benchmark",
    "measure_index_throughput",
    "format_report",
    "environment_block",
]