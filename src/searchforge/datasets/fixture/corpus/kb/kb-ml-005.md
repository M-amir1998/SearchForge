---
id: KB-ML-005
title: "Feature Store"
category: ml-infra
team: data-platform
published: 2024-09-05
year: 2024
language: en
url: https://kb.lumen.example/kb-ml-005
---
# Feature Store

## Online and offline stores
Features are served from a low-latency online store and materialized to an offline store for training. Both are produced by the same feature definitions to avoid training-serving skew.

## Point-in-time joins
Training datasets use point-in-time correct joins so that a training example never sees feature values from the future.
