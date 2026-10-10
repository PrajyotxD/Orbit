# Measuring retrieval

A retrieval benchmark needs a labelled question set and a fixed corpus.

## Metrics

Recall at k is the fraction of relevant items that appear in the top k results.
Precision at k is the fraction of the top k that are relevant. Mean reciprocal
rank rewards putting a relevant item first. nDCG at k rewards putting highly
relevant items at the top using graded relevance rather than a binary label.

## Location-level versus chunk-level

A ground-truth label that names a file and a line range stays valid when the
chunker changes. A label that names a chunk identifier is only valid while the
chunking configuration is unchanged. Reporting both numbers is the honest thing
to do.

## Pitfall

A benchmark that reports a score for a channel which did not actually run is
worse than no benchmark at all, because it hides a broken configuration behind a
plausible number.