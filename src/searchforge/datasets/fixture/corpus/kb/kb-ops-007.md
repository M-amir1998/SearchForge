---
id: KB-OPS-007
title: "Monitoring and Alerting"
category: operations
team: sre
published: 2024-08-30
year: 2024
language: en
url: https://kb.lumen.example/kb-ops-007
---
# Monitoring and Alerting

## Metrics
Services expose Prometheus metrics. Dashboards show request rate, error rate and latency percentiles for every endpoint.

## SLO alerts
Alerts are based on SLO burn rate rather than raw thresholds. A fast-burn alert pages when 2 percent of the monthly error budget is consumed in one hour; a slow-burn alert opens a ticket.

## Alert hygiene
Every alert needs a runbook link. Alerts that fire without action for a month are deleted.
