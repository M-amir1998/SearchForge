---
id: KB-API-006
title: "Streaming Responses"
category: api
team: ml-platform
published: 2025-05-20
year: 2025
language: en
url: https://kb.lumen.example/kb-api-006
---
# Streaming Responses

## Server-sent events
Set `stream=true` to receive tokens as server-sent events while the model is still generating.

## Heartbeats
The server sends a heartbeat comment every 15 seconds so proxies do not close idle connections.

## Cancellation
Closing the connection cancels generation and stops billing for the remaining tokens.
