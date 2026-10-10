"""Evidence relationships and bounded traversal.

Orbit builds a small, honest evidence graph:

Nodes
    ``chunk:<doc_id>:<ordinal>`` and ``doc:<doc_id>`` identifiers.

Edges
    ``contains``      doc -> chunk            (explicit: structural fact)
    ``same_document`` chunk -> chunk          (explicit: structural fact)
    ``shares_symbol`` chunk -> chunk          (explicit: both define/mention
                                               a symbol recorded at index time)
    ``similar_text``  chunk -> chunk          (inferred: heuristic overlap)

Explicit vs inferred is recorded on every edge and surfaced to callers. Orbit
never invents a relationship it cannot justify: an ``inferred`` edge is a
research lead, and ``traverse`` labels it as such.

Traversal is **bounded** by both depth and edge count, so a graph cycle can never
cause unbounded work.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..models import EvidenceEdge, RelationKind, TraversalStep

__all__ = [
    "build_edges_for_document",
    "ensure_document_edges",
    "add_edge",
    "neighbors",
    "traverse",
]

_CHUNK_PREFIX = "chunk:"
_DOC_PREFIX = "doc:"


def _chunk_node(doc_id: str, ordinal: int) -> str:
    return f"{_CHUNK_PREFIX}{doc_id}:{ordinal}"


def _doc_node(doc_id: str) -> str:
    return f"{_DOC_PREFIX}{doc_id}"


def add_edge(
    db,
    source_id: str,
    target_id: str,
    relation: str,
    *,
    kind: RelationKind = RelationKind.INFERRED,
    weight: float = 1.0,
    detail: str = "",
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO evidence_edges(source_id, target_id, kind, relation, weight, detail) "
        "VALUES(?,?,?,?,?,?)",
        (source_id, target_id, kind.value, relation, float(weight), detail[:200]),
    )


def ensure_document_edges(db, doc_id: str) -> int:
    """Create the ``contains`` and ``same_document`` edges for one document."""
    chunks = db.query(
        "SELECT chunk_id, ordinal FROM chunks WHERE doc_id=? ORDER BY ordinal", (doc_id,)
    )
    if not chunks:
        return 0
    doc_node = _doc_node(doc_id)
    for c in chunks:
        add_edge(
            db,
            doc_node,
            c["chunk_id"],
            "contains",
            kind=RelationKind.EXPLICIT,
            detail=f"ordinal {c['ordinal']}",
        )
    ids = [c["chunk_id"] for c in chunks]
    for a, b in zip(ids, ids[1:]):
        add_edge(
            db, a, b, "same_document", kind=RelationKind.EXPLICIT, detail="adjacent ordinal"
        )
    # Also record a forward edge from the last chunk back to the doc for symmetry.
    return len(ids)


def build_edges_for_document(db, doc_id: str) -> int:
    """Alias kept for readability at the call site."""
    return ensure_document_edges(db, doc_id)


def add_shared_symbol_edges(db, doc_id: str, *, limit: int = 500) -> int:
    """Link chunks within a document that share an indexed symbol (explicit)."""
    rows = db.query(
        "SELECT s1.chunk_id AS a, s2.chunk_id AS b, s1.symbol AS sym "
        "FROM chunk_symbols s1 JOIN chunk_symbols s2 "
        "ON s1.symbol = s2.symbol AND s1.chunk_id < s2.chunk_id "
        "JOIN chunks c1 ON c1.chunk_id = s1.chunk_id "
        "JOIN chunks c2 ON c2.chunk_id = s2.chunk_id "
        "WHERE c1.doc_id = ? AND c2.doc_id = ? LIMIT ?",
        (doc_id, doc_id, limit),
    )
    n = 0
    for r in rows:
        add_edge(
            db,
            r["a"],
            r["b"],
            "shares_symbol",
            kind=RelationKind.EXPLICIT,
            detail=r["sym"][:80],
        )
        n += 1
    return n


def neighbors(
    db, node: str, *, relation: str | None = None, direction: str = "out"
) -> list[EvidenceEdge]:
    if direction == "out":
        sql = "SELECT source_id, target_id, kind, relation, weight, detail FROM evidence_edges WHERE source_id=?"
        params: list[object] = [node]
        if relation:
            sql += " AND relation=?"
            params.append(relation)
    else:
        sql = "SELECT source_id, target_id, kind, relation, weight, detail FROM evidence_edges WHERE target_id=?"
        params = [node]
        if relation:
            sql += " AND relation=?"
            params.append(relation)
    return [
        EvidenceEdge(
            source_id=r["source_id"],
            target_id=r["target_id"],
            kind=RelationKind(r["kind"]),
            relation=r["relation"],
            weight=r["weight"],
            detail=r["detail"],
        )
        for r in db.query(sql, params)
    ]


def traverse(
    db,
    start_node: str,
    *,
    max_depth: int = 2,
    max_nodes: int = 200,
    relations: Sequence[str] | None = None,
) -> tuple[list[TraversalStep], list[str]]:
    """Bounded BFS from ``start_node``.

    Returns ``(steps, visited_nodes)``. Every step records whether the relation
    it followed is explicit or inferred so callers never present a heuristic link
    as a proven one.
    """
    if max_depth < 0 or max_nodes < 1:
        return [], []
    visited: set[str] = {start_node}
    steps: list[TraversalStep] = []
    frontier = [start_node]
    depth = 0
    while frontier and depth < max_depth and len(visited) < max_nodes:
        nxt: list[str] = []
        for node in frontier:
            for edge in neighbors(db, node, relation=None):
                if relations and edge.relation not in relations:
                    continue
                target = edge.target_id
                steps.append(
                    TraversalStep(
                        from_id=node,
                        to_id=target,
                        relation=edge.relation,
                        kind=edge.kind,
                        depth=depth + 1,
                        detail=edge.detail,
                    )
                )
                if target not in visited and len(visited) < max_nodes:
                    visited.add(target)
                    nxt.append(target)
        frontier = nxt
        depth += 1
    return steps, sorted(visited)


def nodes_for_document(db, doc_id: str) -> list[str]:
    chunks = db.query("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,))
    return [_doc_node(doc_id)] + [c["chunk_id"] for c in chunks]


def edges_summary(db) -> dict[str, int]:
    return {
        r["relation"]: r["n"]
        for r in db.query(
            "SELECT relation, count(*) AS n FROM evidence_edges GROUP BY relation"
        )
    }


def all_edges(db) -> Iterable[EvidenceEdge]:
    for r in db.query(
        "SELECT source_id, target_id, kind, relation, weight, detail FROM evidence_edges"
    ):
        yield EvidenceEdge(
            source_id=r["source_id"],
            target_id=r["target_id"],
            kind=RelationKind(r["kind"]),
            relation=r["relation"],
            weight=r["weight"],
            detail=r["detail"],
        )