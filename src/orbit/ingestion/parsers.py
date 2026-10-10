"""Per-language parsers.

We chunk raw file text, never a re-serialised AST -- line provenance has to
survive. Parsers only *read* the text to pull out a title, symbols and an
outline; the only thing that rewrites anything is the optional enrichment path
used to build a separate embedding view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ParsedMeta", "parse_metadata", "enrich_text", "detect_language"]


@dataclass(frozen=True)
class ParsedMeta:
    """Lightweight, language-aware metadata for a document."""

    language: str
    title: str = ""
    symbols: tuple[str, ...] = ()
    outline: tuple[str, ...] = ()
    extra: dict | None = None


_PY_DEF = re.compile(r"^(?P<indent>\s*)(?:async\s+)?(?:def|class)\s+([A-Za-z_][\w]*)", re.M)
_JAVA = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)", re.M)
_JS_TS = re.compile(
    r"(?:^|\s)(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum|const)\s+([A-Za-z_$][\w$]*)",
    re.M,
)
_C_CPP = re.compile(
    r"^[A-Za-z_][\w \t\*&:<>,]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:const\s*)?\{",
    re.M,
)
_GO = re.compile(r"^(?:func|type|const|var)\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", re.M)
_RUST = re.compile(r"^\s*(?:pub\s+)?(?:fn|struct|enum|trait|impl|mod|const|static)\s+([A-Za-z_]\w*)", re.M)
_SQL = re.compile(r"\b(?:CREATE\s+(?:TABLE|VIEW|INDEX|FUNCTION)|ALTER\s+TABLE)\s+([A-Za-z_][\w.]*)", re.I)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


def detect_language(path: str, fallback: str = "text") -> str:
    from ..config import LANGUAGE_BY_EXTENSION

    name = path.rsplit("/", 1)[-1].lower()
    if name in ("dockerfile", "containerfile"):
        return "dockerfile"
    if name == "makefile":
        return "make"
    _, _, ext = name.rpartition(".")
    return LANGUAGE_BY_EXTENSION.get("." + ext.lower(), fallback) if ext else fallback


def _heading_title(text: str) -> str:
    m = _MD_HEADING.search(text)
    if m:
        return m.group(2).strip()
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return ""


def _outline(text: str) -> tuple[str, ...]:
    return tuple(m.group(2).strip() for m in _MD_HEADING.finditer(text))


def _symbols_for(language: str, text: str) -> tuple[str, ...]:
    rx = {
        "python": _PY_DEF,
        "java": _JAVA,
        "javascript": _JS_TS,
        "typescript": _JS_TS,
        "kotlin": _JAVA,
        "c": _C_CPP,
        "cpp": _C_CPP,
        "csharp": _C_CPP,
        "go": _GO,
        "rust": _RUST,
        "sql": _SQL,
    }.get(language)
    if rx is None:
        return ()
    found: list[str] = []
    seen: set[str] = set()
    for m in rx.finditer(text):
        name = m.group(m.lastindex or 1) if m.groups() else None
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            found.append(name)
        if len(found) >= 200:
            break
    return tuple(found)


def parse_metadata(text: str, language: str) -> ParsedMeta:
    """Extract title, symbols and outline without altering the text."""
    language = language or "text"
    title = _heading_title(text) if language in ("markdown", "restructuredtext") else _first_sentence(text)
    symbols = _symbols_for(language, text)
    outline = _outline(text) if language == "markdown" else ()
    return ParsedMeta(language=language, title=title, symbols=symbols, outline=outline)


def _first_sentence(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//", "/*", "*", "--")):
            return stripped[:120]
    return ""


def enrich_text(text: str, language: str) -> str:
    """Return an auxiliary text view for embedding, when a parser is available.

    Falls back to the raw text. For markdown we strip syntax; for HTML we strip
    tags if BeautifulSoup is installed. This is best-effort and never raises.
    """
    try:
        if language == "markdown":
            try:
                import markdown as md  # type: ignore

                return md.markdown(text, extensions=["fenced_code", "tables"])
            except Exception:
                return _strip_markdown(text)
        if language in ("html",):
            try:
                from bs4 import BeautifulSoup  # type: ignore

                return BeautifulSoup(text, "html.parser").get_text("\n")
            except Exception:
                return re.sub(r"<[^>]+>", " ", text)
    except Exception:
        return text
    return text


def _strip_markdown(text: str) -> str:
    out = re.sub(r"```.*?```", " ", text, flags=re.S)
    out = re.sub(r"`([^`]*)`", r"\1", out)
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", out)
    out = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"^#{1,6}\s+", "", out, flags=re.M)
    out = re.sub(r"[*_>]+", "", out)
    return out