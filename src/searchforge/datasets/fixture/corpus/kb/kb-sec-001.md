---
id: KB-SEC-001
title: "API Key Rotation Policy"
category: security
team: security
published: 2025-02-10
year: 2025
language: en
url: https://kb.lumen.example/kb-sec-001
---
# API Key Rotation Policy

## Scope
This policy applies to every API key issued by the Lumen platform, including service-to-service keys and customer integration keys.

## Rotation schedule
API keys must be rotated at least every 90 days. Keys that grant write access to production data must be rotated every 30 days. The platform emails key owners 14 days before a key expires.

## How to rotate
Run `lumen keys rotate --key-id <id>` to create a successor key. Both keys stay valid for a 48 hour overlap window so clients can switch without downtime. After the overlap the old key is revoked automatically.

## Emergency revocation
If a key is leaked, revoke it immediately with `lumen keys revoke --key-id <id> --now` and open a security incident. Do not wait for the scheduled rotation.
