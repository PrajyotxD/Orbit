# Conflict candidates

Orbit can flag two passages that look like they disagree. A flag is a research
lead, not a finding.

## Status

Every conflict candidate is inferred, never explicit. The rule that raised the
candidate is named in the record, so a human or an agent can judge whether it
actually matters.

## Scoring

Candidates are ranked by a heuristic score and stored together with both chunk
identifiers, both file paths and both line ranges, so the pair can be read
directly without a second lookup.