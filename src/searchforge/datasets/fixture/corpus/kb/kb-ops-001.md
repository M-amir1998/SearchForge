---
id: KB-OPS-001
title: "Deployment Process"
category: operations
team: platform
published: 2025-01-15
year: 2025
language: en
url: https://kb.lumen.example/kb-ops-001
---
# Deployment Process

## Progressive delivery
Every production release goes through a canary stage. Traffic shifts in steps: 1 percent, 10 percent, 50 percent and finally 100 percent. Each step waits at least 15 minutes while automated analysis compares error rate and p95 latency against the stable version.

## Freeze windows
Deployments are not allowed during announced freeze windows, such as the end-of-year freeze, unless the change fixes an active incident.

## Ownership
The engineer who merges the change owns the rollout and watches dashboards until the release reaches 100 percent.
