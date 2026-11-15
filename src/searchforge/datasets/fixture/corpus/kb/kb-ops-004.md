---
id: KB-OPS-004
title: "Capacity Planning"
category: operations
team: sre
published: 2024-10-20
year: 2024
language: en
url: https://kb.lumen.example/kb-ops-004
---
# Capacity Planning

## Headroom
Every service must keep at least 30 percent CPU and memory headroom at peak traffic so it can absorb a zone failure.

## Forecasting
Capacity is forecast quarterly from traffic growth, launch plans and the peak of the previous quarter. Requests for new hardware need four weeks of lead time.
