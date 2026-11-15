---
id: KB-DATA-002
title: "Schema Evolution"
category: data
team: data-platform
published: 2024-06-18
year: 2024
language: en
url: https://kb.lumen.example/kb-data-002
---
# Schema Evolution

## Compatibility
Schema changes must be backward compatible: add optional fields with defaults, never rename or remove fields in place, and never change a field type.

## Process
Register the new schema version in the schema registry. Producers upgrade only after all consumers can read the new version. Parquet tables are rewritten during the next compaction.
