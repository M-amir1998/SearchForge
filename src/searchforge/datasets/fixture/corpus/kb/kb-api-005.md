---
id: KB-API-005
title: "Batch Inference API"
category: api
team: ml-platform
published: 2025-05-02
year: 2025
language: en
url: https://kb.lumen.example/kb-api-005
---
# Batch Inference API

## Creating a job
Submit a job with `POST /v2/batch/jobs` and a JSONL input file where each line is one request. The response contains a job id.

## Job states
Jobs move through QUEUED, RUNNING, SUCCEEDED, FAILED or CANCELLED. A job that does not finish within 24 hours fails with ERR-5040.

## Results
Results are written to a JSONL output file in the same order as the input lines.
