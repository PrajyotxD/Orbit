# Retrieval channels

Orbit runs two retrieval channels and can run either one alone.

## Lexical

The lexical channel builds a safe FTS5 MATCH expression from free text by
quoting each term and joining the terms with OR, so an arbitrary user string can
never raise an FTS5 syntax error.

## Vector

The vector channel embeds the query with the configured provider and scores every
stored embedding. It is an exhaustive scan, not an index lookup.

## Candidates before fusion

Each channel is asked for a multiple of the requested limit before the two lists
are fused, so fusion has material to work with even when one channel is weak.