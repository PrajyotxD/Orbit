# Example Knowledge Base

A tiny corpus for trying ORBIT end to end. Five documents, deliberately
cross-referencing each other so you can see how evidence relationships behave.

## Try it

```console
# from the repository root
orbit --db .orbit/demo.db index examples/corpus

orbit --db .orbit/demo.db search "how long are sessions valid"
orbit --db .orbit/demo.db search "rollback trigger"
orbit --db .orbit/demo.db search "what port does the gateway bind"
orbit --db .orbit/demo.db context "deployment procedure" --limit 4
orbit --db .orbit/demo.db status --freshness
orbit --db .orbit/demo.db conflicts
```

Then make it interesting — edit a file and re-index:

```console
# change the session timeout in operations.md, then:
orbit --db .orbit/demo.db index examples/corpus
orbit --db .orbit/demo.db search "session validity"
# note the incremented revision and the changed content hash
```

And break a citation on purpose:

```console
# delete operations.md, then validate a chunk id you captured earlier:
orbit --db .orbit/demo.db validate <chunk_id>
# -> "document_deleted" / "path_missing"
```

## What is in here

| File | Topic |
|---|---|
| `operations.md` | Session lifetimes, retries, maintenance windows |
| `deployment.md` | Rollout steps, health gates, rollback triggers |
| `gateway.md` | Ports, TLS, rate limits, upstream timeouts |
| `schema.sql` | Tables, retention, the two settings that conflict |
| `oncall.md` | Who to wake, and the first three checks |

`schema.sql` and `operations.md` deliberately disagree about
`session_ttl_seconds`. Run `orbit conflicts` to see ORBIT flag it as a
*candidate* — a heuristic lead for a human to check, labelled `inferred`, not a
finding.