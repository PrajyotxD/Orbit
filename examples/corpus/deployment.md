# Deployment Guide

## Rollout order

1. Drain the primary node and confirm in-flight requests have completed.
2. Apply the migration, then deploy the application build.
3. Wait for the health gate to report healthy on three consecutive probes.
4. Return the node to the load balancer.

## Health gate

A node reports healthy only when the readiness probe succeeds twice, spaced by
the probe interval, and the error budget burn rate stays below 2x for the
trailing five minutes.

## Rollback trigger

Rollback is automatic when the error budget burn rate exceeds 10x for ten
consecutive minutes, or when the health gate fails twice in a row.

## Rollback procedure

Redeploy the previous build, then re-apply the reverse migration if one was
applied. Rollback is a forward operation: no manual data repair is performed,
because every schema change in this system is backwards compatible for one
release.

## Feature flags

Flags are evaluated at the edge and cached for 30 seconds. Changing a flag does
not require a deploy.