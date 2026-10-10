#!/usr/bin/env python
"""Run the ORBIT retrieval benchmark.

Indexes ``benchmarks/dataset`` into a throwaway index and measures retrieval
quality for each channel (lexical, vector, hybrid) plus indexing throughput.

Everything reported here is measured on the run that produced it. There are no
hard-coded expectations and no numbers copied from another system.

Usage::

    python benchmarks/run_benchmark.py --k 1,3,5,10
    python benchmarks/run_benchmark.py --modes lexical,hybrid --json out.json
    python benchmarks/run_benchmark.py --embedding-provider hf
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from orbit import OrbitConfig, OrbitEngine, __version__  # noqa: E402
from orbit.evaluation.benchmark import format_report, run_benchmark  # noqa: E402
from orbit.evaluation.datasets import load_cases  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORBIT retrieval benchmark")
    p.add_argument(
        "--dataset",
        default=str(REPO_ROOT / "benchmarks" / "dataset"),
        help="directory containing the corpus and questions.json",
    )
    p.add_argument("--k", default="1,3,5,10", help="comma-separated cutoffs")
    p.add_argument(
        "--modes", default="lexical,vector,hybrid", help="comma-separated retrieval modes"
    )
    p.add_argument(
        "--embedding-provider",
        default="hashing",
        choices=["hashing", "hf", "none"],
        help="embedding provider used for the run",
    )
    p.add_argument("--db", default="", help="index path (default: a temp directory)")
    p.add_argument("--json", dest="json_out", default="", help="write results to this file")
    p.add_argument(
        "--keep-db", action="store_true", help="do not delete the temporary index afterwards"
    )
    return p.parse_args(argv)


def environment_block(cfg: OrbitConfig) -> dict[str, object]:
    return {
        "orbit_version": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or platform.machine(),
        "sqlite": sqlite3.sqlite_version,
        "embedding_provider": cfg.embedding.provider,
        "embedding_dimension": cfg.embedding.dimension,
        "embedding_model": cfg.embedding.model if cfg.embedding.provider != "hashing" else "orbit-hashing-v1",
        "chunking": {
            "max_tokens": cfg.chunking.max_tokens,
            "overlap_tokens": cfg.chunking.overlap_tokens,
        },
        "retrieval": {
            "fusion": cfg.retrieval.fusion,
            "lexical_weight": cfg.retrieval.lexical_weight,
            "vector_weight": cfg.retrieval.vector_weight,
            "rrf_k": cfg.retrieval.rrf_k,
            "min_vector_score": cfg.retrieval.min_vector_score,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


LIMITATIONS = [
    "The corpus is small (~17 short documents). These numbers characterise this "
    "corpus, not performance at scale.",
    "The default 'hashing' provider is feature hashing, NOT a semantic language "
    "model. It matches shared tokens and character n-grams; it does not match "
    "synonyms or paraphrase. Install the 'embeddings' extra and use --embedding-"
    "provider hf for a real semantic comparison.",
    "Vector search is an exhaustive O(N*d) scan. There is no approximate "
    "nearest-neighbour index, so latency grows linearly with corpus size.",
    "Latency figures come from a single warm-cache pass on one machine and are "
    "not a benchmark of throughput under concurrency.",
    "Ground truth is authored by hand as (file, line-range) pairs; recall@k is "
    "judged on line-range overlap, not on a human relevance rating.",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    k_values = tuple(int(x) for x in args.k.split(",") if x.strip())
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    tmp_dir = None
    if args.db:
        db_path = Path(args.db)
    else:
        tmp_dir = tempfile.mkdtemp(prefix="orbit-bench-")
        db_path = Path(tmp_dir) / "orbit.db"

    base_config = OrbitConfig(
        db_path=str(db_path),
        embedding=replace(OrbitConfig().embedding, provider=args.embedding_provider),
    )

    try:
        engine = OrbitEngine(base_config)
        try:
            env = environment_block(base_config)
            print("=" * 78)
            print("ORBIT RETRIEVAL BENCHMARK")
            print("=" * 78)
            for key, value in env.items():
                print(f"{key:>22}: {value}")
            print()

            index_started = time.perf_counter()
            report = engine.index([str(dataset)])
            index_seconds = time.perf_counter() - index_started

            stats = engine.stats()
            print(
                f"corpus: {stats.documents_current} documents, "
                f"{stats.chunks} chunks, {stats.embeddings} embeddings "
                f"in {index_seconds * 1000:.1f} ms"
            )
            throughput_docs = (
                stats.documents_current / index_seconds if index_seconds > 0 else 0.0
            )
            throughput_chunks = stats.chunks / index_seconds if index_seconds > 0 else 0.0
            print(f"indexing throughput: {throughput_docs:.1f} docs/s, {throughput_chunks:.1f} chunks/s")
            print()

            cases = load_cases(dataset / "questions.json")
            print(f"queries: {len(cases)}")
            print()

            results = run_benchmark(engine, cases, k_values=k_values, modes=modes)
        finally:
            engine.close()
    finally:
        if tmp_dir and not args.keep_db:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(format_report(results))
    print("-" * 78)
    print("ADDITIONAL NOTES FOR THIS RUN")
    print(f"  corpus: {stats.documents_current} documents, {stats.chunks} chunks")
    print(f"  warm index build: {index_seconds * 1000:.1f} ms "
          f"({throughput_docs:.1f} docs/s, {throughput_chunks:.1f} chunks/s)")
    print("  KNOWN LIMITATIONS")
    for item in LIMITATIONS:
        print(f"    - {item}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "runner_environment": env,
                    "corpus": {
                        "documents": stats.documents_current,
                        "chunks": stats.chunks,
                        "embeddings": stats.embeddings,
                    },
                    "indexing": {
                        "seconds": round(index_seconds, 4),
                        "documents_per_second": round(throughput_docs, 2),
                        "chunks_per_second": round(throughput_chunks, 2),
                        "report": report.as_dict(),
                    },
                    "benchmark": results,
                    "limitations": LIMITATIONS,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())