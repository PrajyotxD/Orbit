"""Semantic icon sets for the Orbit TUI.

The UI doesn't hardcode glyphs. Every symbol is looked up by a *semantic name*
(``search``, ``valid``, ``warn`` …) from one of three interchangeable sets:

* ``unicode`` (the default) — single terminal-cell, non-emoji symbols. These are
  safe in any modern terminal and won't render as a double-width colour emoji.
  Every glyph is checked for ``rich.cells.cell_len == 1`` (see
  :func:`verify_single_width` and the test-suite).
* ``nerd`` — `Nerd Font <https://www.nerdfonts.com/>`_ glyphs (Private Use
  Area). They look best but need a patched font installed in the terminal.
* ``ascii`` — ASCII only, for terminals that can't render anything else.

Importing this module does not pull in ``textual`` or ``rich``, so it stays
cheap for the base install. The width check imports ``rich.cells`` lazily, only
when it's actually called.

Selection precedence (resolved by :func:`resolve_icon_set_name`):
``--icons`` flag  >  ``ORBIT_ICONS`` environment variable  >  ``unicode``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = [
    "IconSet",
    "ICON_NAMES",
    "ICON_SETS",
    "DEFAULT_ICON_SET",
    "get_icon_set",
    "resolve_icon_set_name",
    "verify_single_width",
]

#: The semantic icon names every set must define, in a stable order.
ICON_NAMES: tuple[str, ...] = (
    "search",
    "index",
    "evidence",
    "validate",
    "conflict",
    "mcp",
    "settings",
    "exit",
    "file",
    "related",
    "valid",
    "changed",
    "missing",
    "ok",
    "warn",
    "bullet",
    "arrow",
    "logo",
)


@dataclass(frozen=True)
class IconSet:
    """A complete, immutable mapping of semantic name -> glyph."""

    name: str
    search: str
    index: str
    evidence: str
    validate: str
    conflict: str
    mcp: str
    settings: str
    exit: str
    file: str
    related: str
    valid: str
    changed: str
    missing: str
    ok: str
    warn: str
    bullet: str
    arrow: str
    logo: str

    def glyph(self, name: str) -> str:
        """Return the glyph for a semantic ``name`` (``bullet`` if unknown)."""
        return getattr(self, name, self.bullet)


# --------------------------------------------------------------------------
# unicode (DEFAULT): single-width, non-emoji symbols.
# --------------------------------------------------------------------------
_UNICODE = IconSet(
    name="unicode",
    search="\u2315",     # ⌕ telephone recorder / magnifier
    index="\u2261",      # ≡ identical to (stacked lines)
    evidence="\u2761",   # ❡ curved paragraph sign
    validate="\u2713",   # ✓ check mark
    conflict="\u26a0",   # ⚠ warning sign
    mcp="\u21c4",        # ⇄ rightwards over leftwards arrow
    settings="\u2699",   # ⚙ gear
    exit="\u23fb",       # ⏻ power symbol
    file="\u25a3",       # ▣ white square containing small black square
    related="\u21aa",    # ↪ rightwards arrow with hook
    valid="\u2713",      # ✓ check mark
    changed="\u270e",    # ✎ lower right pencil
    missing="\u2717",    # ✗ ballot x
    ok="\u2713",         # ✓ check mark
    warn="\u26a0",       # ⚠ warning sign
    bullet="\u2022",     # • bullet
    arrow="\u2192",      # → rightwards arrow
    logo="\u25c9",       # ◉ fisheye
)

# --------------------------------------------------------------------------
# nerd: Nerd Font glyphs (Private Use Area). Need a patched terminal font.
# --------------------------------------------------------------------------
_NERD = IconSet(
    name="nerd",
    search="\uf002",     #  magnifier (nf-fa-search)
    index="\uf01c",      #  inbox (nf-fa-inbox)
    evidence="\uf15c",   #  file-text (nf-fa-file_text)
    validate="\uf132",   #  shield (nf-fa-shield)
    conflict="\uf071",   #  warning triangle (nf-fa-warning)
    mcp="\uf1e6",        #  plug (nf-fa-plug)
    settings="\uf013",   #  gear (nf-fa-cog)
    exit="\uf011",       #  power off (nf-fa-power_off)
    file="\uf15b",       #  file (nf-fa-file)
    related="\uf0c1",    #  link (nf-fa-link)
    valid="\uf00c",      #  check (nf-fa-check)
    changed="\uf040",    #  pencil (nf-fa-pencil)
    missing="\uf00d",    #  times (nf-fa-times)
    ok="\uf00c",         #  check
    warn="\uf071",       #  warning triangle
    bullet="\uf111",     #  circle (nf-fa-circle)
    arrow="\uf061",      #  arrow-right (nf-fa-arrow_right)
    logo="\uf0ac",       #  globe (nf-fa-globe)
)

# --------------------------------------------------------------------------
# ascii: ASCII only, for terminals that can't render the other sets. The words
# carry the meaning, so plain characters are enough.
# --------------------------------------------------------------------------
_ASCII = IconSet(
    name="ascii",
    search="/",
    index="+",
    evidence="=",
    validate="v",
    conflict="!",
    mcp="@",
    settings="*",
    exit="x",
    file="-",
    related=">",
    valid="+",
    changed="~",
    missing="x",
    ok="+",
    warn="!",
    bullet="*",
    arrow=">",
    logo="o",
)

ICON_SETS: dict[str, IconSet] = {
    "unicode": _UNICODE,
    "nerd": _NERD,
    "ascii": _ASCII,
}

#: Resolved when neither a flag nor the environment selects a set.
DEFAULT_ICON_SET = "unicode"


def get_icon_set(name: str | None) -> IconSet:
    """Return the :class:`IconSet` for ``name`` (falls back to the default)."""
    if name is None:
        return ICON_SETS[DEFAULT_ICON_SET]
    return ICON_SETS.get(name.strip().lower(), ICON_SETS[DEFAULT_ICON_SET])


def resolve_icon_set_name(flag: str | None = None, env: str | None = None) -> str:
    """Resolve the active set name with precedence ``flag > env > default``.

    A value that does not name a known set is ignored, so an invalid flag falls
    through to the environment and then to :data:`DEFAULT_ICON_SET`.
    """
    for candidate in (flag, env):
        if candidate:
            key = candidate.strip().lower()
            if key in ICON_SETS:
                return key
    return DEFAULT_ICON_SET


def verify_single_width(icon_set: IconSet) -> list[tuple[str, str, int]]:
    """Return ``(name, glyph, cell_len)`` for any glyph wider than one cell.

    An empty list means every glyph is exactly one terminal cell. The test-suite
    uses this to keep the ``unicode`` set from ever going double-width.
    """
    from rich.cells import cell_len  # lazy: keeps the base import free of rich

    offenders: list[tuple[str, str, int]] = []
    for f in fields(icon_set):
        if f.name == "name":
            continue
        glyph = getattr(icon_set, f.name)
        width = cell_len(glyph)
        if width != 1:
            offenders.append((f.name, glyph, width))
    return offenders
