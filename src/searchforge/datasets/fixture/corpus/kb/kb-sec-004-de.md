---
id: KB-SEC-004-DE
title: "Geheimnisverwaltung"
category: security
team: security
published: 2025-03-05
year: 2025
language: de
url: https://kb.lumen.example/kb-sec-004-de
---
# Geheimnisverwaltung

## Regeln
Geheimnisse wie Passwörter, API-Schlüssel, Tokens und private Schlüssel dürfen niemals in ein Git-Repository eingecheckt, in Tickets kopiert oder in Logs geschrieben werden.

## Speicherort
Alle Laufzeitgeheimnisse werden in Vault gespeichert. Dienste lesen Geheimnisse beim Start über den Vault-Agent-Sidecar.

## Vorgehen bei einem Leck
Ein geleaktes Geheimnis gilt als kompromittiert: sofort widerrufen oder rotieren und einen Sicherheitsvorfall melden.
