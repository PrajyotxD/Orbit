# Index structure

## Documents

One row per indexed file: identifier, relative path, absolute path, content hash,
revision counter, size in bytes, modification time, language and status.

## Chunks

One row per chunk: identifier, owning document, ordinal, text, content hash,
start and end line, byte offsets and a token estimate.

## Embeddings

One row per chunk embedding: the chunk identifier, the provider name and a
fixed-width float32 blob. Vector search is a full scan over these blobs.

## Metadata

The schema version and index version live in a metadata table, so a later
release can detect an older database and migrate it instead of guessing.