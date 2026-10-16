# Internal API Rate Limits

Internal services call each other through the API gateway, which enforces per-client rate limits.

## Default limits

The default limit is 600 requests per minute per client, with bursts of up to 100 requests per second. Market data endpoints allow 1,200 requests per minute.

## When limited

A limited request receives HTTP 429 with a `Retry-After` header. Clients must back off exponentially starting at the `Retry-After` value and must not retry more than 5 times.

## Raising a limit

To raise a limit, open a ticket with the platform team including expected peak traffic and the business reason. Increases above 3x the default need a load test.
