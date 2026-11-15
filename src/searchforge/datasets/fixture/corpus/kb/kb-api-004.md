---
id: KB-API-004
title: "Webhooks"
category: api
team: api-platform
published: 2025-03-22
year: 2025
language: en
url: https://kb.lumen.example/kb-api-004
---
# Webhooks

## Delivery
Lumen sends webhook events as HTTP POST requests with a JSON body. Failed deliveries are retried with exponential backoff for up to 24 hours.

## Signature verification
Every request carries an `X-Lumen-Signature` header containing an HMAC-SHA256 of the raw body computed with your webhook secret. Receivers must verify the signature with a constant-time comparison and reject requests older than five minutes to prevent replay attacks.

## Errors
An invalid signature is reported in delivery logs as ERR-4031.
