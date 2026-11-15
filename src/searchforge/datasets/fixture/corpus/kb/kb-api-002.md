---
id: KB-API-002
title: "Pagination"
category: api
team: api-platform
published: 2024-07-15
year: 2024
language: en
url: https://kb.lumen.example/kb-api-002
---
# Pagination

## Cursor-based pagination
List endpoints return a `next_page_token`. Pass it as `page_token` to fetch the next page. An empty token means the last page was reached.

## Page size
`page_size` defaults to 50 and is capped at 500. Larger values are silently reduced to 500.

## Consistency
Tokens expire after 24 hours. Results are ordered by creation time, so items created during iteration may appear on a later page.
