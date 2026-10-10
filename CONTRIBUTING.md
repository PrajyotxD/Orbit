# Contributing to ORBIT

Thanks for considering a contribution. ORBIT is young, and the bar for a change
is specific rather than high.

## The bar

A contribution is ready when:

- **It works on a clean checkout**, with no configuration beyond the defaults.
- **It has a test that fails without it.** Not a test that passes either way.
- **It is honest about its limits.** If it is a heuristic, say so. If it does not
  scale, say so.
- **It does not weaken a guarantee.** See below.

## Guarantees you must not break

These are the properties users are entitled to rely on. A change that breaks one
is a bug, not a trade-off:

1. **A citation is checkable.** Every returned passage keeps a source path, line
   range and content hash that resolves against the live file.
2. **Indexed content is never executed.** No `eval`, no `import` of indexed files,
   no shelling out to document content.
3. **Offline by default.** No network call unless an endpoint was explicitly
   configured.
4. **Degradation is visible.** If a channel did not run, say so. Never report a
   degraded result as a full one.
5. **Inferred is labelled inferred.** Heuristic relationships and conflict
   candidates must carry their kind and the rule that produced them.

## Development setup

```console
git clone <this-repo> orbit
cd orbit
python -m pip install -e ".[dev]"
python -m pytest
```

Optional extras when you need them: `.[embeddings]`, `.[mcp]`, `.[llm]`,
`.[parsing]`.

## Running the tests

```console
python -m pytest                 # everything
python -m pytest tests/test_indexing.py -q
python -m pytest -rxX            # show expected-failure reasons
```

The suite is offline by design. A fixture traps `socket.connect`, so a test that
tries to reach the network fails loudly. Do not "fix" that fixture.

### About `xfail`

Tests documenting unimplemented or known-wrong behaviour use
`@pytest.mark.xfail(strict=True)`. **Strict** means a test that starts passing
turns the build red. That is deliberate: when you fix a bug, remove its `xfail`
marker in the same commit. A lingering `XPASS(strict)` is a maintenance signal,
not noise.

Never add `xfail` to paper over a new failure. Fix the cause or leave the test
red and report it.

## Style

- Type annotations on public functions. No `Any` without a reason.
- Docstrings explain *why*, not *what the next line does*.
- No comments restating the code.
- Prefer deleting code to adding comments to it.
- Keep the dependency count near zero. Every new dependency needs a licence
  review (see `docs/THIRD_PARTY_LICENSES.md`) and a justification.

## Adding a file type

1. Add the extension to `LANGUAGE_BY_EXTENSION` in `config.py`.
2. If it needs parsing beyond text decoding, add a parser in
   `ingestion/parsers.py`. Parsers must **never** change the text that gets
   chunked — provenance is tied to the raw bytes, so a re-serialised AST would
   break citation validation.
3. Add an ingestion test and a retrieval test.
4. Confirm the file is not classified as binary by `_looks_binary`.

## Adding an embedding provider

Implement the `EmbeddingProvider` protocol: `embed(texts)`, `embed_query(text)`,
`health()`, plus `name`, `model` and `dimension`. Then:

- Fail with `EmbeddingUnavailableError` when unavailable, so the engine degrades
  to lexical-only instead of crashing.
- Never contact a remote host during construction.
- Document the provider's real limitations in its module docstring. The hashing
  provider's docstring is the model here: it is explicit that it is not semantic.

## Adding an MCP tool

Add a plain callable to `build_tools(engine)` in `mcp_server.py`. It must not
import the MCP SDK, must validate its inputs, and must return real provenance
(`source_path`, line range, `content_hash`) rather than placeholder strings.
Every returned dict must include the untrusted-content note.

## Reporting bugs

Open an issue with:

- what you did,
- what you expected,
- what happened,
- a minimal reproduction,
- ORBIT version, Python version, and which embedding provider you were using.

If you can attach a failing test, that is the most useful possible report.

## Pull requests

- One logical change per PR.
- `python -m pytest` green, with no new xfails.
- Update the README or `docs/` if you changed behaviour a user would notice.
- Describe honestly what you did **not** do. That is more useful than a list of
  what you did.

## Licence

By contributing you agree that your work is licensed under Apache-2.0, the same
as the project.