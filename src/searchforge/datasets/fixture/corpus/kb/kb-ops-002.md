---
id: KB-OPS-002
title: "Rollback Procedure"
category: operations
team: platform
published: 2025-01-15
year: 2025
language: en
url: https://kb.lumen.example/kb-ops-002
---
# Rollback Procedure

## When to roll back
Roll back when a release increases the error rate, breaks a critical user journey or triggers an SLO burn-rate alert. Rolling back first and debugging later is preferred over fixing forward under pressure.

## How to roll back
Run `lumen deploy rollback --to <release>` with the last known good release. The command reverts traffic to the previous version within two minutes and pins it so automated rollouts do not re-deploy the bad build.

## After a rollback
Open an incident ticket, attach the failing release id, and add a regression test before re-deploying.
