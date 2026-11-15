---
id: KB-ML-006
title: "Evaluating Retrieval Quality"
category: ml-infra
team: search
published: 2025-06-12
year: 2025
language: en
url: https://kb.lumen.example/kb-ml-006
---
# Evaluating Retrieval Quality

## Relevance judgments
Search quality is measured against human relevance judgments on a fixed query set, not against average similarity scores.

## Metrics
We report nDCG@10, Recall@20 and MRR. A change ships only if it does not regress nDCG on critical queries.

## Regression testing
Every ranking change is compared query by query against the baseline before release.
