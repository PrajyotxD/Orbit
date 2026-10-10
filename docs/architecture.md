# Architecture

This document describes how ORBIT is put together and, more importantly, *why*
it is put together that way.

---

## Design commitments

Four decisions shape everything else.

1. **Provenance is structural, not advisory.** A chunk does not merely carry a
   filename that a caller might forget to check. It carries the path, the 1-based
   line range, the character offsets and a SHA-256 of its own text. Validation
   re-derives that hash from the live file. There is no code path that produces a
   citation which has not been checkable.

2. **Offline is the default, not a fallback.** The default configuration opens no
   sockets, downloads no model, and requires no account. Remote providers exist
   but are only constructed when an explicit endpoint is configured.

3. **A citation is a pointer, never a proof.** ORBIT reports where material came
   from. It does not claim the statement built on top of that material is true,
   and no layer of the system implies otherwise.

4. **Degradation is visible.** If the vector channel cannot run, the result says
   so in `degraded_reason` and omits it from `channels_used`. A degraded search is
   a first-class result, not an error and not a silent downgrade.

---

## Layers

```
                    CLI  ·  MCP server  ·  Python API
                                  │
                          OrbitEngine (engine.py)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   IndexManager              Retriever               Evidence
   (indexing/)              (retrieval/)             (evidence/)
        │                         │                         │
        └──────────────┬──────────┴──────────┬──────────────┘
                       │                     │
              chunking.py / ingestion   embedding/
                       │                     │
                       └──────────┬──────────┘
                                  │
                             storage/ (SQLite)
```

Every module depends only on the layer below it. `models.py` is the shared
contract and imports nothing from the project.

---

## Ingestion

`ingestion/loader.py` walks roots deterministically (sorted), pruning generated
and vendored directories by default (`.git`, `node_modules`, `dist`, `.venv`,
`__pycache__`, …) and binary extensions. Candidates pass a path-safety check,
a size check, a secret-name check and a content sniff before decoding.

Failure is a first-class outcome. Every rejected file becomes a `LoadFailure`
with a typed reason. The distinction that matters:

- **skipped** — expected rejections (secret-like name, empty file, excluded path)
- **failed** — genuine problems (unreadable, decode error, too large)

Conflating these would make a security exclusion look like a bug.

Documents are decoded, line endings normalised to `\n`, and hashed. The hash is
over the normalised text, so CRLF and LF copies of the same file index
identically.

## Chunking

`chunking.py` is line-aware: it never splits a line, so `start_line` and
`end_line` always point at real lines that can be re-read from disk. Chunk
boundaries prefer natural breaks (blank lines, Markdown headings, definition
keywords such as `def`/`class`/`fn`/`func`) and otherwise cut on the budget.

The chunker is **deterministic**: boundaries depend only on the text and the
config, never on wall-clock time, randomness or iteration order. Re-indexing an
unchanged file produces byte-identical chunks. This is verified by test.

A chunk's `content_hash` is `sha256(f"{start_line}:{text}")`. Hashing the line
number alongside the text means identical prose at two different locations gets
two different identities — otherwise a moved or duplicated file would collide.

## Storage

SQLite with WAL journaling, foreign keys on, and a busy timeout so concurrent CLI
invocations wait rather than fail with "database is locked".

Tables: `documents`, `chunks`, `chunks_fts` (FTS5 external content over
`chunks.text`), `embeddings`, `chunk_symbols`, `evidence_edges`, `conflicts`,
`orbit_meta`.

FTS5 is used as an **external content** table, so chunk text is stored once and
mirrored into the index rather than duplicated. The two are kept in sync inside
the same transaction as the chunk write. `OrbitDB.rebuild_fts()` exists as a
repair path.

Embeddings are little-endian `float32` blobs, L2-normalised at write time, which
reduces cosine similarity to a dot product.

Migrations are versioned in `orbit_meta.schema_version` and applied in order.
Every statement is idempotent, so an interrupted migration is safe to re-run. A
database written by a newer ORBIT is rejected rather than downgraded.

## Indexing

`IndexManager.sync()` reconciles the corpus against `documents`.

Each document is written inside **its own transaction**. Embeddings are computed
*before* the transaction opens, so a slow or failing provider never holds a write
lock. If anything raises, that document is rolled back completely — no partial
row, no orphan chunk, no dangling FTS entry — and the sweep continues.

Change detection keys on content hash **and** a chunking signature covering
every setting that affects how a file is split (max tokens, overlap, character
limits, boundary rules, and the embedding provider). Without the signature,
changing the chunk size left the old chunks in place while reporting the corpus
as unchanged.

Pruning is deliberately conservative. A document is only pruned if it lived under
a root that was successfully opened in this sweep. A root that fails to open — a
typo, an unmounted volume, the wrong working directory — must never be read as
"the corpus is now empty". This was a real data-loss bug; see the test
`test_a_missing_root_is_reported_and_does_not_nuke_the_index`.

Deletes are soft: the `documents` row is retained as a tombstone so a rename
within the same sweep is recognised as a move rather than a delete plus an add.

## Retrieval

Three independent pieces:

- `lexical.py` sanitises the query into a safe FTS5 expression and reads SQLite's
  built-in `bm25()`, negated so that higher is better.
- `vector.py` loads the candidate embeddings and computes an **exhaustive**
  cosine scan. There is no approximate index, and none is claimed.
- `fusion.py` combines them.

Fusion defaults to Reciprocal Rank Fusion, `score(d) = Σ_c w_c / (k + rank_c(d))`
with `k=60`, following Cormack, Clarke & Büttcher (SIGIR 2009). RRF is used
because it is rank-based and therefore needs no cross-channel score calibration —
which matters because BM25 magnitudes and cosine magnitudes are not comparable. A
min-max weighted fusion is available for cases where an operator wants to express
a real channel preference.

`retriever.py` orchestrates. It reports `channels_used` only for channels that
actually contributed candidates, and populates `degraded_reason` when one could
not. It applies metadata filters, de-duplicates by content hash, and assembles a
`SearchHit` for every survivor with a fully-populated `Evidence` record.

## Evidence

`evidence/validation.py` answers one narrow question: **does the cited line range
still say what the index says it says?** It re-reads the file, extracts the cited
lines, and compares hashes.

`evidence/graph.py` maintains a small, honest graph. `contains` and
`same_document` edges are *explicit* — they are structural facts Orbit can prove.
Anything inferred is labelled `RelationKind.INFERRED` and carries the rule that
produced it. Traversal is bounded by both depth and node count, so a cycle cannot
cause unbounded work.

`evidence/freshness.py` implements two conservative conflict rules:

- `numeric_conflict` — the same attribute name carries materially different
  numbers in two passages (`chunk_tokens = 400` vs `= 900`).
- `negation_conflict` — two near-duplicate passages where exactly one negates a
  shared proposition.

Neither rule proves a contradiction. Similar text is not disagreement. Every
finding is labelled `inferred` and names its rule so it can be judged.

## Context assembly

`context.py` selects blocks for a downstream consumer under three constraints:
**no duplicates** (by content hash, then by Jaccard similarity), **no padding**
(blocks far below the top score are dropped rather than filling a budget), and
**deterministic ordering**. Each block keeps its full provenance.

Token counts are `ceil(chars / 4)` unless a real tokenizer is supplied. ORBIT does
not pretend a character ratio is a model tokenizer.

## Generation (optional)

Disabled by default. `NullGenerator` performs no generation and returns the
retrieved evidence verbatim with `insufficient_evidence=True` — this is what
proves ORBIT works with zero LLM configuration.

`GeneratedAnswer.citations` contains only `Evidence` objects for blocks that were
actually supplied to the generator. A citation is never synthesised.

The system prompt asks the model to distinguish evidence from inference and to
report insufficient evidence. That reduces hallucination risk. It does not
eliminate it, and no claim here should be read as a guarantee.

---

## What is original here, and what is not

Honest attribution matters more than a marketing line.

**Established and independently implemented:** BM25 (Robertson/Sparck-Jones
weighting, via SQLite's built-in implementation), Reciprocal Rank Fusion
(Cormack, Clarke & Büttcher, SIGIR 2009), the hashing trick (Weinberger et al.,
UAI 2009), feature hashing for text, cosine similarity, and mean-pooling for
Transformer embeddings.

**ORBIT's own engineering contribution** is the surrounding system:

- A provenance model where every retrieved passage is structurally checkable
  against the file it came from, including content-hash verification that detects
  silent source drift.
- A chunker whose boundaries are deterministic and line-exact, so provenance
  survives chunking rather than being approximated afterwards.
- Incremental reconciliation that is conservative about deletion — a failed root
  cannot delete a corpus — and that invalidates chunks when the configuration
  that produced them changes.
- Orchestration that keeps both raw channel scores alongside the fused score, so
  a ranking is explainable after the fact.
- Explicit labelling of inferred versus provable relationships in the evidence
  graph.

ORBIT does not claim a novel ranking algorithm, a novel retrieval paradigm, or
any unprecedented capability.