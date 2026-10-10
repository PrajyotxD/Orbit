# Limits

Orbit has limits, and pretending otherwise would make it less useful.

## Vector search is an exact scan

There is no approximate nearest-neighbour index. Every vector query compares the
query vector against every stored embedding, so vector latency grows linearly
with the number of chunks. That is fine for tens of thousands of chunks and not
fine for millions.

## The default embedder is not semantic

The built-in hashing embedder is a deterministic projection of bag-of-words
features. It handles exact terminology well and paraphrase poorly.

## Single writer

SQLite allows one writer at a time, so indexing is a single-process operation
and cannot be parallelised across processes without coordination.

## No answer generation by default

Answer generation is opt-in and disabled in the default configuration. Orbit is a
retrieval and evidence engine; the model in the loop is the client's choice.