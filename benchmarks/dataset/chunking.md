# Chunking

Orbit's chunker is deterministic and line-aware. Two properties matter for
evidence: every chunk points back at real lines in a real file, and reindexing an
unchanged file produces byte-identical chunks.

## Line awareness

The chunker never splits a line. `start_line` and `end_line` are 1-based and
inclusive, so a record with start line 42 and end line 58 means lines 42 through
58 of the source file, counted the way an editor counts them.

## Chunk identity

A chunk identifier is derived from the document, not from the text. It is the
document identifier joined to the chunk ordinal with a colon. Because the
document identifier is a hash of the path, two different files that contain
identical text still get different chunk identifiers.

## Content hash and line position

The per-chunk content hash is computed from the start line together with the
chunk text. Identical text appearing at two different line numbers therefore
produces two different content hashes.

This is deliberate. A passage is identified by where it is, not only by what it
happens to say.

## Sizes

The default budget is 400 estimated tokens per chunk with 60 estimated tokens of
overlap. Token counts are character-ratio estimates, not model tokenizer output.