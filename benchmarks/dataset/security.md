# Security posture

## Indexed documents are untrusted

Anything Orbit reads is treated as data, never as instructions. No indexed
document can change how Orbit behaves, and nothing found on disk is executed.

## No execution

Orbit does not run, import, build or evaluate code that it finds in an indexed
file. It reads text and nothing else.

## Secret-like files are skipped

Files whose names look like credentials (`.env`, `id_rsa`, `*.pem`, `*.key`,
`credentials`) are skipped during ingestion rather than indexed. Optional
content scanning can additionally reject files whose opening bytes match a
recognised private-key or AWS access-key pattern.

## Path containment

Discovery refuses to leave the roots it was given, including through symbolic
links, so an index cannot be pointed at arbitrary parts of the filesystem.