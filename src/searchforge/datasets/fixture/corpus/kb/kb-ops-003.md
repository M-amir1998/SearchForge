---
id: KB-OPS-003
title: "On-Call Handbook"
category: operations
team: sre
published: 2024-12-01
year: 2024
language: en
url: https://kb.lumen.example/kb-ops-003
---
# On-Call Handbook

## Rotation
Each team runs a weekly on-call rotation with a primary and a secondary engineer. Handoff happens every Monday at 10:00 with a short review of open alerts.

## Paging and escalation
Pages are routed by the paging system. The primary must acknowledge a page within 5 minutes. If the primary does not acknowledge, the page escalates to the secondary after 10 minutes and to the engineering manager after 20 minutes.

## Night pages
Only alerts with customer impact may page at night. Everything else is a ticket for business hours.
