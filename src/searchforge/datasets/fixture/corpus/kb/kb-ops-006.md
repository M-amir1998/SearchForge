---
id: KB-OPS-006
title: "Kubernetes Cluster Autoscaling"
category: operations
team: platform
published: 2025-03-18
year: 2025
language: en
url: https://kb.lumen.example/kb-ops-006
---
# Kubernetes Cluster Autoscaling

## Node autoscaling
The k8s cluster autoscaler adds nodes when pods stay in the Pending state because no node has enough CPU or memory, and removes nodes that stay underutilized for 10 minutes.

## Pod autoscaling
Horizontal pod autoscaling scales replicas on CPU utilization or custom metrics such as queue length.

## Troubleshooting
If pods remain Pending, check node group maximum sizes, resource requests that exceed any instance type, and taints that prevent scheduling.
