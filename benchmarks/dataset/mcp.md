# MCP server

Orbit exposes retrieval to agents through a Model Context Protocol server that
speaks over stdio.

## Tools

`orbit_search` runs a hybrid query and returns ranked passages with evidence.

`orbit_get_chunk` fetches one passage by identifier so a client can read it in
full after seeing a snippet.

`orbit_validate_citation` re-checks a citation against the live source.

## Evidence, always

Every tool result carries the source path, the line range and the content hash
for each passage it returns. A client that quotes Orbit must carry those fields
through unchanged; the protocol gives the server no way to invent them.