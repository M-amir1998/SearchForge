---
id: KB-API-001
title: "RetryPolicy Configuration"
category: api
team: sdk
published: 2025-02-01
year: 2025
language: en
url: https://kb.lumen.example/kb-api-001
---
# RetryPolicy Configuration

## Fields
`RetryPolicy.max_attempts` sets the maximum number of attempts including the first call (default 3). `RetryPolicy.backoff_multiplier` controls exponential backoff (default 2.0) and `RetryPolicy.initial_delay_ms` the first delay (default 200 ms).

## Jitter
Full jitter is enabled by default to avoid synchronized retries from many clients.

## Which errors are retried
Only idempotent requests and errors marked retryable (HTTP 429, 503 and network timeouts) are retried. Validation errors are never retried.
