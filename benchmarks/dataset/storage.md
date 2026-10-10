# Storage

Orbit keeps everything in a single SQLite file. There is no server process and
no separate vector database to operate.

## Journal mode

The database runs in WAL write-ahead-log mode so that readers do not block the
single writer, and so a crash part way through a write leaves the last committed
state intact.

## Full text index

The full-text index is an FTS5 virtual table declared as external content over
the chunks table. The chunk text is stored exactly once; the FTS table holds
only the inverted index, which keeps the database file small.

## Ranking function

Lexical ranking calls SQLite's built-in `bm25()` function directly rather than
reimplementing it, and negates the returned value so that higher is better.

## Transactions

One document is written inside one transaction. The chunk rows, the FTS rows and
the embedding rows for that document all commit together or not at all.