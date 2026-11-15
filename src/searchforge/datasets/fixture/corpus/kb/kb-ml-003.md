---
id: KB-ML-003
title: "Embedding Model Upgrades"
category: ml-infra
team: search
published: 2025-06-05
year: 2025
language: en
url: https://kb.lumen.example/kb-ml-003
---
# Embedding Model Upgrades

## Re-embedding
Vectors from different embedding models are not comparable. Upgrading the embedding model requires re-embedding the whole corpus into a new index version.

## Rollout
Build the new index alongside the old one, evaluate retrieval quality on the relevance test set, then switch the serving alias. Keep the old version until the new one has served traffic for a week.
