"""Orbit's full-screen terminal UI.

The TUI is an optional layer over the existing engine. It needs `textual` (and
`rich`, which textual pulls in anyway), and since both are only used by the
interactive UI they live behind an optional extra (`orbit-rag[tui]`) and are
imported lazily. The non-interactive CLI, the Python API, and the MCP server
don't depend on anything in this package.

Entry points:
    * :func:`launch` — resolve textual, build the app, and run it.
    * :func:`main` — argparse front end for ``orbit ui`` / ``python -m orbit.tui``.
"""

from __future__ import annotations

__all__ = ["launch", "main", "TEXTUAL_INSTALL_HINT"]

TEXTUAL_INSTALL_HINT = (
    "the Orbit TUI needs the 'tui' extra: pip install 'orbit-rag[tui]' "
    "(or: pip install textual)"
)


def _require_textual() -> None:
    """Raise an OrbitError with install guidance when textual isn't present."""
    try:
        import textual  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via CLI path
        from ..errors import OrbitError

        raise OrbitError(TEXTUAL_INSTALL_HINT, detail=str(exc)) from exc


def launch(config, *, default_index_path: str = "", icon_set: str | None = None) -> None:
    """Build and run the Orbit TUI against ``config`` (an :class:`OrbitConfig`).

    ``icon_set`` picks the glyph set (``unicode`` / ``nerd`` / ``ascii``). With
    ``None``, the choice falls through to the ``ORBIT_ICONS`` environment
    variable and then the ``unicode`` default (see
    :func:`orbit.tui.icons.resolve_icon_set_name`).
    """
    _require_textual()
    import os

    from .app import run
    from .icons import resolve_icon_set_name

    resolved = resolve_icon_set_name(icon_set, os.environ.get("ORBIT_ICONS"))
    run(config, default_index_path=default_index_path, icon_set=resolved)


def main(argv=None) -> int:
    """``orbit ui`` / ``python -m orbit.tui`` entry point."""
    import argparse
    import sys

    from ..config import load_config
    from ..errors import OrbitError

    parser = argparse.ArgumentParser(
        prog="orbit ui",
        description="Launch Orbit's full-screen terminal UI.",
    )
    parser.add_argument("--db", default=None, metavar="PATH", help="index database path")
    parser.add_argument("--config", default=None, metavar="PATH", help="JSON config file")
    parser.add_argument(
        "--index-path",
        default="",
        metavar="PATH",
        help="pre-fill the Index Repository field with this path",
    )
    parser.add_argument(
        "--icons",
        default=None,
        choices=["unicode", "nerd", "ascii"],
        help="icon glyph set (default: unicode; or set ORBIT_ICONS)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = load_config(args.config)
        if args.db:
            config = config.with_db_path(args.db)
    except OrbitError as exc:
        import json

        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        return 1

    try:
        launch(config, default_index_path=args.index_path, icon_set=args.icons)
    except OrbitError as exc:
        import json

        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
