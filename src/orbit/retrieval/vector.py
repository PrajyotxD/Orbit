"""Exact vector retrieval.

Exhaustive cosine scan over every stored embedding. No ANN index, no HNSW, no
quantisation -- results stay deterministic and easy to assert on, which matters
more than speed at this scale. Cost is O(N*d) per query;
:func:`plan_vector_search` reports it so callers can tell when that stops being
fine.

Vectors are stored L2-normalised, so cosine collapses to a dot product.
"""

from __future__ import annotations

import array
from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import RetrievalError

__all__ = ["VectorHit", "cosine_scores", "plan_vector_search", "load_matrix", "search_vector"]


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float  # cosine similarity in [-1, 1]
    rank: int


@dataclass(frozen=True)
class VectorSearchPlan:
    """Honest cost estimate for an exhaustive scan."""

    n_vectors: int
    dimension: int
    bytes_scanned: int
    is_ann: bool = False


def plan_vector_search(n_vectors: int, dimension: int) -> VectorSearchPlan:
    """Describe the cost of an exact scan. ``is_ann`` is always False today."""
    return VectorSearchPlan(
        n_vectors=n_vectors,
        dimension=dimension,
        bytes_scanned=n_vectors * dimension * 4,
        is_ann=False,
    )


def load_matrix(rows: Sequence[tuple[str, int, bytes]]) -> tuple[list[str], "object"]:
    """Build ``(chunk_ids, matrix)`` from ``(chunk_id, dim, blob)`` rows.

    ``matrix`` is a ``(n, d)`` float32 array. Rows are assumed L2-normalised.
    """
    import numpy as np

    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dims = {dim for _, dim, _ in rows}
    if len(dims) != 1:
        raise RetrievalError(f"mixed embedding dimensions in index: {sorted(dims)}")
    dim = dims.pop()
    buf = np.empty((len(rows), dim), dtype=np.float32)
    for i, (_, _, blob) in enumerate(rows):
        if len(blob) != dim * 4:
            raise RetrievalError(
                f"corrupt embedding blob at row {i}: expected {dim * 4} bytes, got {len(blob)}"
            )
        buf[i] = np.frombuffer(blob, dtype="<f4")
    return [chunk_id for chunk_id, _, _ in rows], buf


def cosine_scores(query_vec: Sequence[float], matrix) -> "object":
    """Cosine similarity of ``query_vec`` against every row of ``matrix``.

    Vectors are L2-normalised, so this is a dot product. Falls back to an
    explicit normalisation if a query vector is not unit length.
    """
    import numpy as np

    q = np.asarray(query_vec, dtype=np.float32)
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if q.shape[0] != matrix.shape[1]:
        raise RetrievalError(
            f"query vector has dimension {q.shape[0]}, index has {matrix.shape[1]}"
        )
    norm = float(np.linalg.norm(q))
    if norm > 0 and abs(norm - 1.0) > 1e-3:
        q = q / norm
    return matrix @ q


def search_vector(
    db,
    query_vec: Sequence[float],
    *,
    limit: int = 50,
    min_score: float = 0.0,
    doc_ids: Sequence[str] | None = None,
) -> list[VectorHit]:
    """Exhaustive cosine search. Returns hits sorted by descending score."""
    import numpy as np

    if doc_ids is not None and len(doc_ids) == 0:
        return []

    sql = "SELECT e.chunk_id AS chunk_id, e.dim AS dim, e.vector AS vector FROM embeddings e"
    params: list[object] = []
    if doc_ids is not None:
        placeholders = ",".join("?" for _ in doc_ids)
        sql += f" JOIN chunks c ON c.chunk_id = e.chunk_id WHERE c.doc_id IN ({placeholders})"
        params.extend(doc_ids)

    rows = db.query(sql, params)
    if not rows:
        return []
    chunk_ids, matrix = load_matrix([(r["chunk_id"], r["dim"], r["vector"]) for r in rows])
    scores = cosine_scores(query_vec, matrix)

    # Top-k without materialising a full sort when the corpus is large.
    k = min(limit, len(scores))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    out: list[VectorHit] = []
    for rank, i in enumerate(idx, start=1):
        s = float(scores[i])
        # Scores at or below the floor are non-matches. The default floor is
        # deliberately above zero because unrelated English passages still share
        # character n-grams, producing a small positive cosine that is noise.
        if s <= min_score:
            continue
        out.append(VectorHit(chunk_id=chunk_ids[i], score=s, rank=rank))
    return out


def pack_vector(vec: Sequence[float]) -> tuple[int, bytes]:
    """Serialise a float sequence to ``(dim, little-endian float32 blob)``."""
    arr = array.array("f", [float(x) for x in vec])
    if arr.itemsize != 4:  # pragma: no cover - platform guard
        raise RetrievalError("platform 'f' array is not 4 bytes; float32 packing unsupported")
    return len(arr), arr.tobytes()


def unpack_vector(dim: int, blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    if len(arr) != dim:
        raise RetrievalError(f"vector blob length {len(arr)} != declared dim {dim}")
    return list(arr)