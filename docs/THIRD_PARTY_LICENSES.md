# Third-party dependencies and licences

ORBIT is licensed under **Apache-2.0** (see `LICENSE`). ORBIT does not
relicense third-party code. Every dependency below retains its own licence.

This file is a compliance record, not a legal opinion. It was compiled by
inspecting installed distribution metadata on the development machine
(`importlib.metadata`), not from vendor claims.

---

## Runtime dependencies (required)

| Package | Version tested | Licence | Why ORBIT needs it |
|---|---|---|---|
| **numpy** | 1.26.4 | BSD-3-Clause | Exact cosine similarity over float32 embedding matrices. Pure-Python dot products are unusable at realistic corpus sizes. |

That is the complete runtime dependency list. ORBIT's default configuration needs
nothing else — SQLite and FTS5 come from the standard library, the HTTP client is
only imported if a remote provider is explicitly configured, and the embedding
provider is built in.

## Optional dependencies

| Extra | Package | Licence | Purpose |
|---|---|---|---|
| `embeddings` | `transformers` 4.46.2 | Apache-2.0 | Real neural embeddings |
| `embeddings` | `torch` 2.5.1+cpu | BSD-3-Clause | Transformer inference (CPU build is enough) |
| `mcp` | `mcp` 2.3.0 | MIT | MCP server |
| `llm` | `httpx` 0.28.1 | BSD-3-Clause | HTTP client for OpenAI-compatible generation/embedding endpoints |
| `parsing` | `PyYAML` 6.0.3 | MIT | YAML parsing (optional; plain-text fallback) |
| `parsing` | `Markdown` 3.10 | BSD-3-Clause | Markdown-to-text for embedding enrichment (optional) |
| `parsing` | `beautifulsoup4` 4.12.3 | MIT | HTML text extraction (optional) |

## Development dependencies

| Package | Version tested | Licence |
|---|---|---|
| `pytest` | 8.3.4 | MIT |
| `pytest-cov` | — | MIT |
| `ruff` | — | MIT |
| `mypy` | — | MIT |

None of the development dependencies are required to *use* ORBIT.

---

## Bundled native libraries (transitive, binary wheels only)

This section matters for redistribution, so it is recorded precisely rather than
summarised away.

The **Windows** `numpy` wheel dynamically links OpenBLAS, which in turn bundles:

| Component | Licence | Compatibility with Apache-2.0 redistribution |
|---|---|---|
| OpenBLAS | BSD-3-Clause | Compatible |
| LAPACK (inside OpenBLAS) | BSD-3-Clause-Attribution | Compatible |
| GCC runtime (`libgfortran`) | GPL-3.0 **with GCC Runtime Library Exception v3.1** | Compatible — the Runtime Library Exception explicitly grants permission to propagate the resulting binary under other terms, provided it was produced by an Eligible Compilation Process |
| `libdivide` | Zlib | Compatible |
| `dragon4` | MIT | Compatible |
| `tempita` | MIT | Compatible |
| `libquadmath` | LGPL-2.1-or-later | Dynamically linked, used unmodified; not distributed by ORBIT |

The GCC Runtime Library Exception is the standard mechanism by which numpy wheels
are redistributable under permissive licences. ORBIT does not modify any of these
components.

The pure-Python `numpy` source distribution carries no such native bundles, and
the macOS and manylinux wheels use a different BLAS arrangement. Re-run
`python -m pip show -f numpy` after changing platform and record the result here.

---

## Licence compatibility conclusion

ORBIT is Apache-2.0. Every dependency above is under the Apache License, the BSD
licences (including 3-clause and the attribution variant), MIT, Zlib, or LGPL with
dynamic linking. The one component carrying a copyleft licence (`libgfortran`)
does so under the GCC Runtime Library Exception, which grants the redistribution
rights ORBIT needs.

On this evidence, distributing ORBIT under Apache-2.0 is compatible with the
dependency set. This has not been reviewed by a lawyer; if that matters to you,
get one.

---

## Algorithmic attribution

ORBIT implements established algorithms independently. No source code was copied
from any of these works.

| Algorithm | Original source | How ORBIT uses it |
|---|---|---|
| Okapi BM25 | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) | Ranking via SQLite's built-in `bm25()`; ORBIT calls it rather than reimplementing it |
| Reciprocal Rank Fusion | Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009) | Default hybrid fusion strategy, `k=60` |
| Hashing trick | Weinberger, Dasgupta, Langford, Smola, Attenberg, *Feature Hashing for Large Scale Multitask Learning* (ICML 2009) | Signed feature hashing into a fixed-width vector |
| Feature hashing for text | Weinberger et al.; also G spelling / hashing in NLP | Token unigram/bigram and character 4-gram hashing in the default embedder |
| Cosine similarity | Standard linear algebra | Exact vector ranking |
| Transformer mean pooling | Standard sentence-embedding practice | Used by the optional `hf` embedding provider |
| FTS5 / BM25 | SQLite project (public domain) | Lexical index and ranking |

ORBIT's own engineering contribution — the provenance model, the deterministic
line-exact chunker, conservative incremental reconciliation, the auditable
two-channel fusion, and the explicit-vs-inferred evidence graph — is described in
[`architecture.md`](architecture.md).

---

## Reviewing this file

When you add or remove a dependency:

1. Update `pyproject.toml`.
2. Re-check the licence with `python -m pip show <package>`.
3. Update the relevant table here.
4. If the dependency bundles native code, record its licence and compatibility.

Do not add a dependency whose licence is incompatible with Apache-2.0 without
raising it first.