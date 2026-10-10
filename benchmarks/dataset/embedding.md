# Embeddings

Orbit supports several embedding providers and degrades to lexical-only when none
of them can be constructed.

## The hashing embedder

The built-in provider is a deterministic feature-hashing embedder. It is
completely offline, needs no model download, and produces the same vector for the
same text on every machine and every run.

It is explicitly not semantic. It is a projection of bag-of-words features: two
sentences that share no vocabulary receive near-zero similarity even when they
mean the same thing, and a paraphrase will be missed. This is an honest default
for offline operation, not a claim of semantic quality.

## The HuggingFace provider

The `hf` provider runs a sentence-transformers style model and mean-pools the
token vectors into a single fixed-size vector. It needs the optional embeddings
extra installed, plus either a local model directory or network access on first
use.

## Degradation

If no provider can be constructed, indexing continues with lexical retrieval only
and the result records a reason instead of silently returning nothing.