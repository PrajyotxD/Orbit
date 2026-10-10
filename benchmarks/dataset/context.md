# Context assembly

Assembling context turns a ranked result set into a budgeted block of text.

## Token budget

A configurable token budget, 4000 by default, caps the assembled output.
Passages are added in rank order until the budget is exhausted or the maximum
block count is reached, whichever comes first.

## Deduplication

Chunks with an identical content hash are dropped, and chunks whose similarity
exceeds a threshold of 0.92 are treated as duplicates as well.

## Rendering

The rendered context numbers each block and prints its source path and line
range, so a downstream answer can refer to blocks by number and be traced back
to a file.