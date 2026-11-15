---
id: KB-OPS-008
title: "Load Shedding and Rate Limiting"
category: operations
team: platform
published: 2025-04-10
year: 2025
language: en
url: https://kb.lumen.example/kb-ops-008
---
# Load Shedding and Rate Limiting

## Rate limits
Each API client has a token-bucket rate limit. When the bucket is empty the gateway returns HTTP 429 with error code ERR-4012 and a Retry-After header.

## Load shedding
When a backend is overloaded, the gateway sheds low-priority traffic first so that interactive requests keep working. Batch jobs are paused before interactive traffic is rejected.

## Client guidance
Clients must back off exponentially and respect Retry-After instead of retrying immediately.
