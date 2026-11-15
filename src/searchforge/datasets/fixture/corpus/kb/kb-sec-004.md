---
id: KB-SEC-004
title: "Secrets Management"
category: security
team: security
published: 2025-03-05
year: 2025
language: en
url: https://kb.lumen.example/kb-sec-004
---
# Secrets Management

## Rules
Secrets such as passwords, API keys, tokens and private keys must never be committed to a Git repository, pasted into tickets or written to logs.

## Where secrets live
All runtime secrets are stored in Vault. Services read secrets at start-up through the Vault agent sidecar; secrets are never baked into container images.

## Prevention
Every repository runs the detect-secrets pre-commit hook and a CI secret scan. A detected secret blocks the merge.

## If a secret leaks
Treat it as compromised: revoke or rotate it immediately, purge it from history if possible, and report a security incident.
