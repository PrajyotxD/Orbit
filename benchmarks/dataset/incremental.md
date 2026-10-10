# Incremental indexing

Reindexing is a reconcile, not a rebuild.

## Identity by content hash

Each document's content is hashed. A file whose hash is unchanged from the last
run is skipped: it is not re-read, not re-chunked and not re-embedded.

## Renames

A file whose content hash matches a document already indexed under a different
path is treated as a rename. The existing record is retired and written at the
new path, keeping its revision counter.

## Deletions

Files that were indexed previously and are no longer present are marked deleted
and drop out of search results, while their chunk rows are retained so the
history stays auditable.