# Security Policy

## Supported versions

The `0.1.x` line is the actively developed line. There are no older supported
branches yet.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's "Report a vulnerability" on the Security tab, or contact the
maintainers directly. Include a reproduction, the affected version, and the
impact you observed. Please give maintainers a reasonable window to ship a fix
before disclosing publicly.

---

## Threat model

ORBIT ingests documents that the operator does not necessarily control. The
central assumption of this project is:

> **Indexed content is untrusted data.**

Everything below follows from that assumption.

### What ORBIT does to defend against it

**Indexed content is never executed.**
Ingestion decodes bytes to text. Parsers extract metadata with regular
expressions. No indexed file is imported, `eval`'d, `exec`'d, or compiled. A
`.py` file in your corpus is text to ORBIT, not code to run. The test suite
monkeypatches `os.system`, `os.remove`, `subprocess.*`, and the builtins
`eval`/`exec`/`compile` to raise, then indexes a document containing a prompt
injection payload and runs a full index → search → assemble cycle, asserting
that none of them is ever called.

**Prompt injection is treated as content, not instruction.**
A document containing "IGNORE ALL PREVIOUS INSTRUCTIONS" is indexed and returned
verbatim, because that is the correct behaviour for a search engine. It is
rendered under an explicit `[n] Source: path:lines` header so a downstream agent
can distinguish quoted material from its own instructions. ORBIT does not attempt
to "defend" against injection by mangling document text — it defends by making
provenance visible and by never elevating document content into a control channel.

This is a mitigation, not a guarantee. A sufficiently determined document can
still mislead a language model that reads it. The defence is that ORBIT shows
you *where* the text came from, so a human or an agent can inspect it.

**No accidental egress.**
Remote embedding and generation providers are only constructed when an explicit
`base_url` is configured. With the default configuration, ORBIT opens no network
connections. API keys are read from the process environment at call time via a
configured variable name; they are never written to the config file, the
database, or the logs.

**Secrets are excluded by default.**
Files matching common secret names — `.env*`, `id_rsa`, `id_ed25519`, `*.pem`,
`*.key`, `*.p12`, `*.pfx`, `*.jks`, `*.keystore`, `credentials`, `.netrc`,
`_netrc`, `.npmrc`, `.pypirc`, `.htpasswd`, `service-account.json` — are not
indexed. An opt-in content scan (`ingestion.scan_content_for_secrets`) additionally
matches well-known private-key headers and AWS access-key IDs in the first 8 KiB
of each file.

**Path traversal is rejected.**
A file that resolves outside the indexed root is refused, and this holds for
symlinked files and symlinked directories. `ingestion.follow_symlinks` defaults to
false.

**Search input cannot inject into the query engine.**
User text is never passed raw to the FTS5 `MATCH` operator. `build_match_query`
tokenises the query and rebuilds it as quoted terms, so arbitrary input cannot
produce an FTS5 syntax error or an unintended boolean expression. The test suite
fuzzes this with SQL fragments, unbalanced quotes, `NEAR(`, wildcards, NUL bytes
and 4000-character queries.

### What ORBIT does not defend against, and why

**Secret detection is best-effort, not a guarantee.**
Filename and header matching cannot find a credential stored under an innocuous
name inside an ordinary file. If your corpus might contain secrets, exclude them
at the source. Do not rely on ORBIT as a secret scanner.

**Prompt injection against a downstream model is not solved.**
ORBIT makes injected text attributable. It cannot stop a language model from
following it. If your agent acts on retrieved content, validate the cited source
before acting on it.

**No authentication or multi-tenancy.**
The database is a local file. Anyone who can read it can read the indexed
content. Do not expose it over a network. There is no per-user isolation.

**No encryption at rest.**
Chunks and embeddings are stored in plaintext SQLite. Use full-disk encryption
if the corpus is sensitive.

**Denial of service by corpus size.**
Indexing is CPU- and memory-bounded only by `ingestion.max_file_bytes` (2 MB
default) and the vector scan's `O(N·d)` cost. A hostile corpus can make queries
slow. There is no resource quota.

---

## Reporting a finding

When you report a vulnerability, please include:

- the ORBIT version and Python version,
- the configuration in use (embedding provider and mode in particular),
- a minimal reproduction,
- what an attacker gains.

If you believe you have found a way to make ORBIT execute indexed content, or to
leak a secret it was told to exclude, or to write outside the indexed root,
please report it as high severity.