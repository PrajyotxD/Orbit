# Command line

Orbit ships a single `orbit` executable.

## Commands

`orbit init` creates the database schema.

`orbit index [PATH ...]` reconciles files into the index.

`orbit search QUERY` runs a search and prints ranked citations.

`orbit validate CHUNK_ID` re-checks one citation against the source file.

`orbit stats` prints document, chunk and embedding counts.

`orbit serve` starts the MCP server over stdio.

## Output

Search output is one citation per line in the form `path:start-end`, followed by
the score and the retrieval method that produced it.