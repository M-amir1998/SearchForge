---
id: KB-ML-002
title: "Model Quantization Guide"
category: ml-infra
team: ml-platform
published: 2024-12-10
year: 2024
language: en
url: https://kb.lumen.example/kb-ml-002
---
# Model Quantization Guide

## Why quantize
Quantization stores weights with fewer bits, for example int8 or int4 instead of float16. The model becomes smaller and faster, which makes CPU and edge inference practical.

## Accuracy trade-offs
int8 weight quantization usually loses little accuracy. int4 needs calibration data and must be evaluated on the target task before release.

## Process
Quantize a copy of the model, run the evaluation suite, and compare quality and latency against the full-precision baseline.
