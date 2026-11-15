---
id: KB-ML-001
title: "Continuous Batching for LLM Inference"
category: ml-infra
team: ml-platform
published: 2025-04-18
year: 2025
language: en
url: https://kb.lumen.example/kb-ml-001
---
# Continuous Batching for LLM Inference

## Idea
Static batching waits for a full batch and for the longest sequence to finish. Continuous batching schedules at the iteration level: finished sequences leave the batch and new requests join at every decoding step, which keeps the accelerator busy.

## Effects
Continuous batching increases throughput substantially under mixed-length traffic. It also makes memory usage depend on the number of active sequences, so the KV cache budget must be configured.

## Configuration
`max_num_seqs` limits concurrent sequences and `kv_cache_fraction` limits memory reserved for the KV cache.
