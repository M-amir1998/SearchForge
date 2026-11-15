---
id: KB-API-007
title: "Idempotency Keys"
category: api
team: api-platform
published: 2024-11-25
year: 2024
language: en
url: https://kb.lumen.example/kb-api-007
---
# Idempotency Keys

## Purpose
Send an `Idempotency-Key` header on POST requests so that retrying a request after a network failure does not create a second payment, job or resource.

## Behaviour
The first response for a key is stored for 24 hours. Repeating the request with the same key returns the stored response. Reusing a key with a different request body fails with ERR-4090.
