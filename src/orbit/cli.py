"""Command-line interface for Orbit.

Built on :mod:`argparse` from the standard library so that the CLI has no
dependency beyond what the engine already needs. The same operations are
available programmatically through :class:`orbit.engine.OrbitEngine`; this module
is a thin, scriptable shell over it.

Contract
--------
* human-readable output goes to stdout, machine-readable JSON to stdout as well
  when ``--json`` is passed;
* errors go to stderr as ``{"error": ..., "message": ..., "detail": ...}`` so a
  caller can branch on ``error`` without parsing prose;
* exit ``0`` on success (including "no results"), ``1`` on an
  :class:`~orbit.errors.OrbitError`, ``2`` on a usage error.

``--db`` and ``--config`` are accepted both before and after the subcommand
(``orbit --db x init`` and ``orbit init --db x`` are equivalent). When both are
given, the flag placed after the subcommand wins. stdout/stderr are forced to
UTF-8 so non-ASCII passage text survives a Windows console whose active code
page is cp1252.

Answers produced by ``orbit answer`` carry the same caveat as every Orbit
generation path: prompt instructions reduce hallucination risk but do not
guarantee factual accuracy, and citations record provenance rather than truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any, TextIO

from .config import OrbitConfig, config_to_dict, load_config
from .engine import OrbitEngine
from .errors import OrbitError
from .evidence import find_conflicts, freshness_summary
from .generation import NullGenerator
from .generation import answer as generate_answer
from .models import (
    ConflictCandidate,
    Evidence,
    SearchHit,
    SearchResult,
    ValidationResult,
)

__all__ = ["main", "build_parser", "DEFAULT_DB"]

DEFAULT_DB = ".orbit/orbit.db"

_EPILOG = """\
Orbit is offline by default: lexical retrieval plus a deterministic built-in
embedder. Answer generation is optional and disabled unless
generation.enabled is true and generation.base_url is set.
"""


# --- helpers


def _force_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so indexed non-ASCII text is not mangled.

    On Windows the console code page defaults to cp1252, which turns an em dash
    in a citation detail into a replacement character. Reconfiguring the Python
    stream (not the console) is safe on every platform and is a no-op where the
    stream cannot be reconfigured, e.g. when stdout is redirected to something
    exotic.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _out(text: str = "", stream: TextIO | None = None) -> None:
    print(text, file=stream or sys.stdout)


def _evidence_dict(ev: Evidence) -> dict[str, Any]:
    return {
        "chunk_id": ev.chunk_id,
        "doc_id": ev.doc_id,
        "source_path": ev.source_path,
        "start_line": ev.start_line,
        "end_line": ev.end_line,
        "content_hash": ev.content_hash,
        "retrieval_method": str(getattr(ev.retrieval_method, "value", ev.retrieval_method)),
        "score": ev.score,
        "rank": ev.rank,
        # Both channel scores are kept so a fused ranking stays explainable
        # after the fact; see docs/architecture.md.
        "lexical_score": ev.lexical_score,
        "vector_score": ev.vector_score,
        "normalized_score": ev.normalized_score,
        "revision": ev.revision,
        "validation": str(getattr(ev.validation, "value", ev.validation)),
        "validation_detail": ev.validation_detail,
    }


def _hit_dict(hit: SearchHit, *, include_text: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "rank": hit.evidence.rank,
        "citation": hit.citation,
        "score": hit.evidence.score,
        "snippet": hit.snippet,
        "evidence": _evidence_dict(hit.evidence),
    }
    if hit.metadata:
        data["metadata"] = dict(hit.metadata)
    if include_text:
        data["text"] = hit.text
    return data


def _result_dict(result: SearchResult, *, include_text: bool = True) -> dict[str, Any]:
    return {
        "query": result.query,
        "retrieval_method": str(getattr(result.retrieval_method, "value", result.retrieval_method)),
        "total_candidates": result.total_candidates,
        "duration_ms": round(result.duration_ms, 3),
        "channels_used": list(result.channels_used),
        "degraded_reason": result.degraded_reason,
        "hits": [_hit_dict(h, include_text=include_text) for h in result.hits],
    }


def _conflict_dict(c: ConflictCandidate) -> dict[str, Any]:
    return {
        "chunk_id_a": c.chunk_id_a,
        "chunk_id_b": c.chunk_id_b,
        "path_a": c.path_a,
        "path_b": c.path_b,
        "line_range_a": c.line_range_a,
        "line_range_b": c.line_range_b,
        "rule": c.rule,
        "score": c.score,
        "detail": c.detail,
        "kind": str(getattr(c.kind, "value", c.kind)),
    }


def _validation_dict(chunk_id: str, result: ValidationResult) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "state": str(getattr(result.state, "value", result.state)),
        "is_valid": result.is_valid,
        "detail": result.detail,
        "current_hash": result.current_hash,
        "current_revision": result.current_revision,
    }


def _parse_modes(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ("lexical", "vector")
    modes = tuple(m.strip().lower() for m in raw.split(",") if m.strip())
    if not modes:
        raise OrbitError("modes must name at least one channel")
    valid = {"lexical", "vector", "hybrid"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise OrbitError(
            f"unknown retrieval mode(s): {', '.join(bad)}",
            detail="valid modes: lexical, vector, hybrid",
        )
    return modes


def _resolve_config(args: argparse.Namespace) -> OrbitConfig:
    """Merge the pre- and post-subcommand forms of ``--db`` / ``--config``.

    A value supplied after the subcommand wins over one supplied before it;
    a value supplied only before it is used as the base.
    """
    config_path = getattr(args, "config", None) or getattr(args, "g_config", None)
    cfg = load_config(config_path)
    db = getattr(args, "db", None) or getattr(args, "g_db", None)
    return cfg.with_db_path(db) if db else cfg


def _engine(args: argparse.Namespace) -> OrbitEngine:
    return OrbitEngine(_resolve_config(args))


# --- commands


def cmd_init(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        stats = engine.init()
    _out(_dumps(stats.as_dict()))
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Serve the MCP tools over stdio using the resolved db/config."""
    try:
        from .mcp_server import build_server
    except ImportError as exc:
        raise OrbitError(
            "the 'mcp' extra is required for the MCP server: pip install 'orbit[mcp]'",
            detail=str(exc),
        ) from exc
    engine = _engine(args)
    server = build_server(engine)
    server.run("stdio")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the full-screen terminal UI over the resolved db/config.

    The UI is an optional layer that needs the ``tui`` extra (textual). The
    import stays in here so the rest of the CLI doesn't depend on it, and a
    missing dependency comes back as a normal OrbitError with install guidance
    instead of a traceback.
    """
    from .tui import launch

    config = _resolve_config(args)
    launch(
        config,
        default_index_path=getattr(args, "index_path", "") or "",
        icon_set=getattr(args, "icons", None),
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        report = engine.index(args.paths or None, prune=not args.no_prune)
    if args.json:
        _out(_dumps(report.as_dict()))
        return 0
    _out(f"scanned={report.scanned} added={report.added} updated={report.updated} "
         f"unchanged={report.unchanged} deleted={report.deleted} renamed={report.renamed} "
         f"skipped={report.skipped} failed={report.failed}")
    _out(f"chunks_written={report.chunks_written} embeddings_written={report.embeddings_written} "
         f"duration_ms={report.duration_ms:.1f}")
    for path, reason, detail in report.failures:
        _out(f"  FAILED  {path}: {reason} {detail}".rstrip())
    for path, reason, detail in report.skips:
        _out(f"  skipped {path}: {reason} {detail}".rstrip())
    return 0 if report.ok else 1


def cmd_search(args: argparse.Namespace) -> int:
    modes = _parse_modes(args.modes)
    with _engine(args) as engine:
        result = engine.search(
            args.query,
            limit=args.limit,
            language=args.language,
            path_glob=args.path,
            modes=modes,
        )
    if args.json:
        _out(_dumps(_result_dict(result, include_text=args.show_text)))
        return 0
    if not result.hits:
        _out(f"no results for {result.query!r} (channels: {', '.join(result.channels_used) or 'none'})")
        return 0
    _out(
        f"{len(result.hits)} hit(s) in {result.duration_ms:.1f}ms "
        f"via {'+'.join(result.channels_used) or 'none'}"
    )
    if result.degraded_reason:
        _out(f"  degraded: {result.degraded_reason}")
    for hit in result.hits:
        ev = hit.evidence
        method = getattr(ev.retrieval_method, "value", ev.retrieval_method)
        state = getattr(ev.validation, "value", ev.validation)
        _out(f"{ev.rank}. {hit.citation}  score={ev.score:.4f}  {method}  {state}")
        _out(f"   {hit.snippet or hit.text[:160]}")
        if args.show_text:
            _out("   ---")
            for line in hit.text.splitlines():
                _out(f"   {line}")
            _out("   ---")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        hit = engine.get_chunk(args.chunk_id)
        if hit is None:
            _out(f"no chunk with id {args.chunk_id!r}")
            return 0
        validation = engine.validate_citation(args.chunk_id)
    ev = hit.evidence
    if args.json:
        _out(_dumps({
            "citation": hit.citation,
            "text": hit.text,
            "evidence": _evidence_dict(ev),
            "validation": _validation_dict(args.chunk_id, validation),
        }))
        return 0
    _out(f"{hit.citation}")
    _out(f"  chunk_id={ev.chunk_id} doc_id={ev.doc_id} hash={ev.content_hash[:16]}")
    _out(f"  revision={ev.revision} language={hit.metadata.get('language')} "
         f"ordinal={hit.metadata.get('ordinal')}")
    state = getattr(validation.state, "value", validation.state)
    _out(f"  validation={state}{' — ' + validation.detail if validation.detail else ''}")
    _out("")
    _out(hit.text)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        stats = engine.stats()
        payload: dict[str, Any] = stats.as_dict()
        if args.freshness:
            payload["freshness"] = freshness_summary(engine.db)
    _out(_dumps(payload))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        result = engine.validate_citation(args.chunk_id)
        if engine.get_chunk(args.chunk_id) is None:
            payload = _validation_dict(args.chunk_id, result)
            payload["found"] = False
            _out(_dumps(payload))
            return 0
    payload = _validation_dict(args.chunk_id, result)
    payload["found"] = True
    _out(_dumps(payload))
    return 0


#: Chunk rows scanned by ``find_conflicts`` before candidates are truncated to
#: ``--limit``. ``find_conflicts(limit=...)`` bounds the *scan*, not the number
#: of candidates returned, so passing the user's ``--limit`` straight through
#: would silently examine only that many chunks and under-report conflicts.
CONFLICT_SCAN_LIMIT = 5000


def cmd_conflicts(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        found = find_conflicts(engine.db, limit=CONFLICT_SCAN_LIMIT)
    candidates = found[: args.limit] if args.limit and args.limit > 0 else found
    if args.json:
        # Always emit JSON under --json, including `[]` when nothing matched, so
        # a script never has to parse prose to learn the list is empty.
        _out(_dumps([_conflict_dict(c) for c in candidates]))
        return 0
    if not candidates:
        _out("no conflict candidates found")
        return 0
    _out(f"{len(candidates)} conflict candidate(s); these are research leads, not findings.")
    for c in candidates:
        kind = getattr(c.kind, "value", c.kind)
        _out(f"  [{kind}] {c.rule} score={c.score:.4f}")
        _out(f"    A {c.path_a}:{c.line_range_a}  ({c.chunk_id_a})")
        _out(f"    B {c.path_b}:{c.line_range_b}  ({c.chunk_id_b})")
        if c.detail:
            _out(f"    {c.detail}")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    from .context import assemble_context, render_context

    with _engine(args) as engine:
        result = engine.search(args.query, limit=args.limit)
        assembled = assemble_context(result, engine.config.context)
    if args.json:
        _out(_dumps({
            "query": args.query,
            "blocks": [
                {
                    "index": i,
                    "source_path": b.source_path,
                    "start_line": b.start_line,
                    "end_line": b.end_line,
                    "content_hash": b.content_hash,
                    "chunk_id": b.chunk_id,
                    "token_estimate": b.token_estimate,
                    "score": b.score,
                    "text": b.text,
                }
                for i, b in enumerate(assembled.blocks, start=1)
            ],
            "total_token_estimate": assembled.total_token_estimate,
            "token_budget": assembled.token_budget,
            "truncated": assembled.truncated,
            "dropped_duplicates": assembled.dropped_duplicates,
            "dropped_irrelevant": assembled.dropped_irrelevant,
        }))
        return 0
    if not assembled.blocks:
        _out(f"no context assembled for {args.query!r}: nothing retrieved or everything was dropped as duplicate/irrelevant")
        return 0
    _out(render_context(assembled))
    _out("")
    _out(
        f"--- {len(assembled.blocks)} block(s), ~{assembled.total_token_estimate} "
        f"of {assembled.token_budget} tokens, truncated={assembled.truncated}, "
        f"dropped_duplicates={assembled.dropped_duplicates}, "
        f"dropped_irrelevant={assembled.dropped_irrelevant}"
    )
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    from .generation import build_generator

    with _engine(args) as engine:
        if args.no_llm:
            generator: Any = NullGenerator(
                engine.config.generation, reason="--no-llm was passed"
            )
        else:
            generator = build_generator(engine.config.generation)
        ok, note = generator.health()
        if not ok:
            print(f"generator unavailable: {note}", file=sys.stderr)
            return 1
        answer = generate_answer(
            engine, args.question, limit=args.limit, generator=generator
        )
        generator_name = generator.name

    if args.json:
        _out(_dumps({
            "question": args.question,
            "answer": answer.text,
            "model": answer.model,
            "generator": generator_name,
            "insufficient_evidence": answer.insufficient_evidence,
            "usage": dict(answer.usage),
            "citations": [_evidence_dict(c) for c in answer.citations],
            "disclaimer": (
                "Prompt instructions reduce hallucination risk but do not guarantee "
                "factual accuracy. Citations record provenance of the supporting "
                "material, not the truth of the generated claim."
            ),
        }))
        return 0
    _out(answer.text)
    _out("")
    _out(f"--- generator={generator_name} model={answer.model} "
         f"insufficient_evidence={answer.insufficient_evidence}")
    if answer.citations:
        _out("--- citations (retrieved records, re-verifiable):")
        for c in answer.citations:
            _out(f"  [{c.rank}] {c.source_path}:{c.line_range}  "
                 f"chunk_id={c.chunk_id}  hash={c.content_hash[:12]}")
    else:
        _out("--- no passages were supplied, so there is nothing to cite.")
    return 0


def cmd_config_schema(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", None) or getattr(args, "g_config", None)
    if args.defaults or not config_path:
        _out(_dumps(config_to_dict(OrbitConfig())))
        return 0
    _out(_dumps(config_to_dict(load_config(config_path))))
    return 0


# --- parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Attach the per-subcommand form of ``--db`` / ``--config``.

    These use the plain ``db``/``config`` dests so they take precedence over the
    equivalent flags given before the subcommand (which land in ``g_db`` /
    ``g_config``).
    """
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=f"index database path (default: config db_path, else {DEFAULT_DB})",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="JSON config file loaded with orbit.config.load_config",
    )


def _add_global(parser: argparse.ArgumentParser) -> None:
    """Attach the pre-subcommand form of ``--db`` / ``--config``."""
    parser.add_argument(
        "--db",
        dest="g_db",
        default=None,
        metavar="PATH",
        help=f"index database path (default: config db_path, else {DEFAULT_DB})",
    )
    parser.add_argument(
        "--config",
        dest="g_config",
        default=None,
        metavar="PATH",
        help="JSON config file loaded with orbit.config.load_config",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbit",
        description="Orbit: evidence-driven, local-first retrieval over your own files.",
        epilog=_EPILOG,
    )
    _add_global(parser)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_init = sub.add_parser("init", help="create the index schema and print IndexStats")
    _add_common(p_init)
    p_init.set_defaults(func=cmd_init)

    p_index = sub.add_parser("index", help="index files or directories into the store")
    p_index.add_argument("paths", nargs="*", help="files or directories (default: config.paths)")
    p_index.add_argument("--no-prune", action="store_true", help="keep index entries for deleted files")
    p_index.add_argument("--json", action="store_true", help="print the IndexReport as JSON")
    _add_common(p_index)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="hybrid search with citations")
    p_search.add_argument("query", help="free-text query")
    p_search.add_argument("--limit", type=int, default=None, help="max hits (default: config.retrieval.limit)")
    p_search.add_argument("--json", action="store_true", help="print results as JSON")
    p_search.add_argument("--language", default=None, help="restrict to a detected language, e.g. python")
    p_search.add_argument("--path", default=None, metavar="GLOB", help="restrict to paths matching GLOB")
    p_search.add_argument("--modes", default=None, metavar="LIST", help="comma-separated channels: lexical,vector")
    p_search.add_argument("--show-text", action="store_true", help="print full passage text for each hit")
    _add_common(p_search)
    p_search.set_defaults(func=cmd_search)

    p_get = sub.add_parser("get", help="print one chunk with its citation and validation state")
    p_get.add_argument("chunk_id")
    p_get.add_argument("--json", action="store_true", help="print the chunk as JSON")
    _add_common(p_get)
    p_get.set_defaults(func=cmd_get)

    p_status = sub.add_parser("status", help="print IndexStats as JSON")
    p_status.add_argument("--freshness", action="store_true", help="also validate every chunk and report freshness counts")
    _add_common(p_status)
    p_status.set_defaults(func=cmd_status)

    p_validate = sub.add_parser("validate", help="re-verify a citation against the live source")
    p_validate.add_argument("chunk_id")
    _add_common(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_conflicts = sub.add_parser("conflicts", help="list flagged conflict candidates")
    p_conflicts.add_argument("--limit", type=int, default=50, help="max candidates to report")
    p_conflicts.add_argument("--json", action="store_true", help="print candidates as JSON")
    _add_common(p_conflicts)
    p_conflicts.set_defaults(func=cmd_conflicts)

    p_context = sub.add_parser("context", help="assemble and render a bounded context window")
    p_context.add_argument("query")
    p_context.add_argument("--limit", type=int, default=None, help="candidate hits before assembly")
    p_context.add_argument("--json", action="store_true", help="print blocks as JSON")
    _add_common(p_context)
    p_context.set_defaults(func=cmd_context)

    p_answer = sub.add_parser("answer", help="assemble context and generate an answer")
    p_answer.add_argument("question")
    p_answer.add_argument("--limit", type=int, default=None, help="candidate hits before assembly")
    p_answer.add_argument("--no-llm", action="store_true", help="use the null generator; no model is contacted")
    p_answer.add_argument("--json", action="store_true", help="print the answer as JSON")
    _add_common(p_answer)
    p_answer.set_defaults(func=cmd_answer)

    p_schema = sub.add_parser("config-schema", help="print the configuration surface as JSON")
    p_schema.add_argument("--defaults", action="store_true", help="always print built-in defaults")
    _add_common(p_schema)
    p_schema.set_defaults(func=cmd_config_schema)

    p_mcp = sub.add_parser("mcp", help="serve the MCP tools over stdio")
    _add_common(p_mcp)
    p_mcp.set_defaults(func=cmd_mcp)

    p_ui = sub.add_parser("ui", help="launch the full-screen terminal UI (needs the 'tui' extra)")
    p_ui.add_argument(
        "--index-path",
        default="",
        metavar="PATH",
        help="pre-fill the Index Repository field with this path",
    )
    p_ui.add_argument(
        "--icons",
        default=None,
        choices=["unicode", "nerd", "ascii"],
        help="icon glyph set (default: unicode; or set ORBIT_ICONS)",
    )
    _add_common(p_ui)
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``orbit`` console script.

    Usable as ``python -m orbit.cli ...``, as the ``orbit`` console script, and
    as ``from orbit.cli import main; main([...])``. Returns the process exit
    code rather than calling :func:`sys.exit`, so it is testable in-process.
    """
    _force_utf8_streams()
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse already printed usage/help to stderr
        return int(exc.code) if exc.code is not None else 2

    if getattr(args, "func", None) is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        return int(args.func(args))
    except OrbitError as exc:
        print(_dumps(exc.as_dict()), file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())