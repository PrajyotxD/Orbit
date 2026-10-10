# On-call Notes

## First three checks

Run these before anything else, in this order:

1. `orbit status --freshness` on the relevant index.
2. The error budget burn rate for the affected service.
3. The health gate history for the last 30 minutes.

## Escalation

Page the primary on-call first. Escalate to the secondary after 15 minutes
without acknowledgement. Escalate to the engineering manager after 30 minutes.

## Common causes

The three causes that account for most pages, in order of frequency:

- **A configuration change**, usually a flag flipped without a soak period.
  Check the flag history before anything else.
- **A schema migration applied out of order.** Rollback is a forward operation;
  verify migration order before attempting one.
- **Upstream saturation.** The gateway pool is capped at 64 idle connections per
  upstream; saturation shows up as upstream timeouts after 30 seconds.

## Communication

Post the first status update within 10 minutes of acknowledging a page, even if
the only news is that you are still investigating.