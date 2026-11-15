---
id: KB-SEC-002
title: "TLS Certificate Renewal"
category: security
team: infrastructure
published: 2024-11-03
year: 2024
language: en
url: https://kb.lumen.example/kb-sec-002
---
# TLS Certificate Renewal

## Overview
All public endpoints terminate TLS at the edge load balancer. Certificates are issued by our ACME provider and managed by cert-manager.

## Automatic renewal
cert-manager renews certificates 30 days before expiry. A Prometheus alert `CertificateExpiringSoon` fires when any certificate is within 21 days of expiry, which means automatic renewal has failed.

## Manual renewal
If automatic renewal fails, check the ACME challenge logs, fix the DNS record and trigger renewal with `kubectl cert-manager renew <certificate>`. Verify the new expiry date with `openssl s_client`.

## Protocol versions
Only TLS 1.2 and TLS 1.3 are accepted. TLS 1.0 and 1.1 were disabled in 2023.
