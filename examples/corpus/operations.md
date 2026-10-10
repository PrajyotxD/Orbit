# Operations Guide

## Sessions

A session becomes valid the moment its access token is issued. The default
session lifetime is declared as `session_ttl_seconds = 3600`.

The gateway refuses a request whose session has expired before the upstream is
contacted, which keeps stale credentials from reaching the backing store.

## Retries

Upstream calls are retried at most three times. Retries use exponential backoff
with a base delay of 250 milliseconds and a hard ceiling of 5 seconds.

Retries are safe because every mutating endpoint carries an idempotency key.

## Maintenance windows

Read-only traffic is served during a maintenance window. Write traffic is
rejected with HTTP 503 and a `Retry-After` header set to the next window.

## Session storage

Sessions live in the primary region and are replicated to the secondary within
900 seconds. See `schema.sql` for the retention policy that governs expiry.