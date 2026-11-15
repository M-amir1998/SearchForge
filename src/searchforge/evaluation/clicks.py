"""Optional anonymized click-log support — with explicit position-bias handling.

**Clicks are not relevance labels.** Users click what they *see*, and they see
top positions far more often (position/exposure bias); attractive titles get
clicks regardless of usefulness (presentation bias); unclicked results were often
never examined. This module therefore only produces *weak, debiased label
candidates* flagged ``label_source="click_weak"`` — they are never merged into
qrels automatically and should feed human review (see review.py).

Debiasing: a simple inverse-propensity estimate where the examination propensity
of position ``p`` is approximated by ``CTR(p) / CTR(1)`` over the whole log. This
is a crude estimator (it conflates relevance with position when rankings are not
randomized) and is documented as such.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from searchforge.errors import EvaluationFailure


@dataclass(frozen=True, slots=True)
class ClickEvent:
    query_key: str
    document_id: str
    position: int
    clicked: bool


@dataclass(frozen=True, slots=True)
class WeakLabel:
    query_key: str
    document_id: str
    impressions: int
    clicks: int
    ips_clicks: float
    weak_relevance: float
    label_source: str = "click_weak"


def load_click_log(path: Path) -> list[ClickEvent]:
    events: list[ClickEvent] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "query" in row:
                raise EvaluationFailure(f"{path.name}:{n}: raw query text is not accepted; provide query_key (hash)")
            pos = int(row["position"])
            if pos < 1:
                raise EvaluationFailure(f"{path.name}:{n}: position must be >= 1")
            events.append(ClickEvent(str(row["query_key"]), str(row["document_id"]), pos, bool(row["clicked"])))
    return events


def position_ctr(events: list[ClickEvent]) -> dict[int, float]:
    imp: dict[int, int] = defaultdict(int)
    clk: dict[int, int] = defaultdict(int)
    for e in events:
        imp[e.position] += 1
        clk[e.position] += int(e.clicked)
    return {p: clk[p] / imp[p] for p in sorted(imp)}


def propensities(events: list[ClickEvent], floor: float = 0.05) -> dict[int, float]:
    ctr = position_ctr(events)
    top = ctr.get(1, 0.0)
    if top <= 0:
        return dict.fromkeys(ctr, 1.0)
    return {p: max(floor, min(1.0, c / top)) for p, c in ctr.items()}


def weak_labels(events: list[ClickEvent], min_impressions: int = 5) -> list[WeakLabel]:
    prop = propensities(events)
    imp: dict[tuple[str, str], int] = defaultdict(int)
    clk: dict[tuple[str, str], int] = defaultdict(int)
    ips: dict[tuple[str, str], float] = defaultdict(float)
    for e in events:
        key = (e.query_key, e.document_id)
        imp[key] += 1
        if e.clicked:
            clk[key] += 1
            ips[key] += 1.0 / prop.get(e.position, 1.0)
    out = []
    for key, n in imp.items():
        if n < min_impressions:
            continue
        rate = min(1.0, ips[key] / n)
        out.append(
            WeakLabel(
                query_key=key[0],
                document_id=key[1],
                impressions=n,
                clicks=clk[key],
                ips_clicks=round(ips[key], 4),
                weak_relevance=round(rate, 4),
            )
        )
    return sorted(out, key=lambda w: (w.query_key, -w.weak_relevance))
