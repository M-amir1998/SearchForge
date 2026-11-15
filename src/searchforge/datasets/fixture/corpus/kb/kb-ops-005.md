---
id: KB-OPS-005
title: "Database Backups and Restore"
category: operations
team: data-platform
published: 2025-02-28
year: 2025
language: en
url: https://kb.lumen.example/kb-ops-005
---
# Database Backups and Restore

## Backups
Primary databases take nightly snapshots and continuously archive write-ahead logs, which allows point-in-time recovery for the last 7 days. Snapshots are retained for 35 days.

## Restore drills
A restore drill runs every month: a random snapshot is restored into an isolated environment and verified with consistency checks. Drill results are recorded in the reliability dashboard.

## Restoring production
Production restores require approval from the database owner and the incident commander.
