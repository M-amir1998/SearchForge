---
id: KB-API-003
title: "Authentication"
category: api
team: api-platform
published: 2025-01-08
year: 2025
language: en
url: https://kb.lumen.example/kb-api-003
---
# Authentication

## Bearer tokens
Requests authenticate with an API key or an OAuth access token sent as `Authorization: Bearer <token>`.

## OAuth client credentials
Service integrations use the OAuth 2.0 client credentials flow. Access tokens are valid for one hour.

## Request ids
Send an `X-Request-Id` header to correlate client logs with server traces; the server generates one if it is missing.
