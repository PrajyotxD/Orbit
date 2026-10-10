# Evidence validation

A citation in Orbit is a pointer, not proof. A valid citation means the passage
can be re-read from the source file at the stated lines. It does not mean the
passage answers the question, and it does not mean the claim in the passage is
true.

## What is checked

Validation re-reads the indexed document record and compares the stored content
hash against the current state of the index. Nothing is re-interpreted, and
nothing is re-ranked.

## Validation states

`VALID` - the document is current and its content hash still matches.

`CONTENT_CHANGED` - the path exists but the bytes on disk differ from the bytes
that were indexed.

`PATH_MISSING` - the recorded source path no longer exists.

`CHUNK_OUT_OF_RANGE` - the file still exists but no longer has any lines in the
recorded range, because it shrank.

`DOCUMENT_DELETED` - the document was removed and its record was retired.

`NOT_VALIDATED` - validation was not requested for this result.