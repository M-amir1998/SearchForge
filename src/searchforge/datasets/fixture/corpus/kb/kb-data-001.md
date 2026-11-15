---
id: KB-DATA-001
title: "Data Pipeline Freshness SLAs"
category: data
team: data-platform
published: 2024-10-01
year: 2024
language: en
url: https://kb.lumen.example/kb-data-001
---
# Data Pipeline Freshness SLAs

## Freshness
Tier-1 tables must be updated within 2 hours of the source event. Tier-2 tables have a 24 hour freshness target.

## Late data
Events arriving more than 3 days late are written to a quarantine table and backfilled weekly.
