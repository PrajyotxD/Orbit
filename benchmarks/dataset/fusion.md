# Rank fusion

Orbit combines its lexical and vector candidate lists with Reciprocal Rank
Fusion, the algorithm introduced by Cormack, Clarke and Buettcher at SIGIR 2009.

## The formula

For a document d, the fused score is the sum over channels of the channel weight
divided by the constant `rrf_k` plus that channel's rank for d, where rank starts
at one for the top result.

## Why ranks and not scores

RRF is rank-based. It never inspects the magnitude of a BM25 score or a cosine
similarity, so the two channels do not need to be calibrated onto a shared scale
first. That is the entire reason to prefer it here.

## The damping constant

The constant `rrf_k` defaults to 60, following the original paper. It damps the
influence of the very top ranks so that a single channel's number one result
cannot dominate the fused ranking on its own.

## The alternative

A min-max weighted fusion is also implemented. It normalises each channel's
scores first and then takes a weighted sum, which lets an operator express a
genuine preference for one channel but is more sensitive to outliers.