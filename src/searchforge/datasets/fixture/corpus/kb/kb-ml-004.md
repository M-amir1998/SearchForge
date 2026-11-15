---
id: KB-ML-004
title: "GPU Out-of-Memory Troubleshooting"
category: ml-infra
team: ml-platform
published: 2025-03-30
year: 2025
language: en
url: https://kb.lumen.example/kb-ml-004
---
# GPU Out-of-Memory Troubleshooting

## Symptoms
Inference workers crash with CUDA out of memory errors, usually during traffic spikes or for very long prompts.

## Fixes
Reduce `max_num_seqs`, lower the maximum context length, or reduce `kv_cache_fraction`. Quantized weights free memory for the KV cache.

## Prevention
Load test with the longest supported prompts before raising concurrency limits.
