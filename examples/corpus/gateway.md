# Gateway Configuration

## Ports

The gateway binds to port 8080 on every interface. Health checks are served on
port 8081 and are not exposed outside the cluster.

TLS terminates at the gateway. Certificates rotate every 30 days.

## Rate limiting

Rate limiting is applied per caller identity, not per source address. The default
burst allowance is 200 requests, refilling at 50 requests per second.

A caller that exceeds its allowance receives HTTP 429 with a `Retry-After`
header.

## Upstream timeouts

Upstream requests time out after 30 seconds. Streamed responses have a longer
idle timeout of 120 seconds, measured between chunks rather than from the start
of the response.

## Connection pooling

The pool holds at most 64 idle connections per upstream and reuses them for 300
seconds. A connection idle beyond that is closed and re-established.