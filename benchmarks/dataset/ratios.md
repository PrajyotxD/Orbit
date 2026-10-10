# Scores and ratios

Orbit reports two different kinds of number for every hit, and confusing them is
the most common source of "why did my ranking look wrong" questions.

## Raw channel scores

The lexical channel returns a negated BM25 value straight from SQLite FTS5. The
vector channel returns a cosine similarity. Neither number means anything on its
own, and neither number means the same thing on a different query.

## Normalized scores

For the fused ranking, Orbit computes a display score in the closed interval
zero to one with the helper `min_max_normalize`. The rule is: subtract the lowest
score in that channel's own candidate list, divide by the range, and if every
score in the list is identical the helper returns one point zero for all of them
instead of dividing by zero.

Normalization is performed per channel, per query.

## Why raw BM25 is not comparable across queries

A raw BM25 score contains an inverse document frequency term computed from the
corpus statistics of the search itself. Two queries against the same corpus can
therefore return scores an order of magnitude apart even when the underlying
match quality is identical.

This is why Orbit never compares a BM25 number from one query against a BM25
number from another query, and why the fused ranking consumes ranks rather than
raw score magnitudes.

## Reporting discipline

A retrieval benchmark must state whether a score is raw or normalized. Mixing the
two makes two runs of the same system incomparable to each other, and to any
other system.