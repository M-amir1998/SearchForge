---
id: KB-OPS-002-OLD
title: "Rollback Procedure (legacy)"
category: operations
team: platform
published: 2022-06-01
year: 2022
language: en
url: https://kb.lumen.example/kb-ops-002-old
---
# Rollback Procedure (legacy)

This page describes the legacy rollback process and is superseded by the current rollback procedure.

## How to roll back
Re-run the previous Jenkins deployment job with the old build number. Traffic is switched manually on the load balancer. This process takes about thirty minutes and is no longer supported.
