"""Dependency-free SVG scatter plot (quality vs latency), deterministic output."""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

PALETTE = ("#2a6fdb", "#e0663d", "#2f9e62", "#9b59b6", "#c49a00", "#1aa3a3", "#d6336c", "#5c6b7a", "#8a5a2b")


@dataclass(frozen=True, slots=True)
class Point:
    label: str
    x: float
    y: float


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def scatter_svg(
    points: Sequence[Point],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 720,
    height: int = 460,
    y_min: float | None = 0.0,
    y_max: float | None = 1.0,
) -> str:
    ml, mr, mt, mb = 70, 200, 50, 60
    pw, ph = width - ml - mr, height - mt - mb
    xs = [p.x for p in points] or [0.0, 1.0]
    ys = [p.y for p in points] or [0.0, 1.0]
    x_lo, x_hi = 0.0, max(xs) * 1.15 if max(xs) > 0 else 1.0
    y_lo = min(ys) if y_min is None else y_min
    y_hi = max(ys) if y_max is None else y_max

    def sx(v: float) -> float:
        return ml + (v - x_lo) / (x_hi - x_lo) * pw

    def sy(v: float) -> float:
        return mt + ph - (v - y_lo) / ((y_hi - y_lo) or 1.0) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{ml}" y="28" font-size="15" font-weight="bold" fill="#1f2933">{html.escape(title)}</text>',
    ]
    for t in _ticks(x_lo, x_hi):
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="#e4e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle" fill="#52606d">{t:.1f}</text>')
    for t in _ticks(y_lo, y_hi):
        y = sy(t)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#e4e7eb"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#52606d">{t:.2f}</text>')
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#9aa5b1"/>')
    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{height - 15}" text-anchor="middle" fill="#323f4b">'
        f"{html.escape(x_label)}</text>"
    )
    parts.append(
        f'<text transform="translate(18 {mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
        f'fill="#323f4b">{html.escape(y_label)}</text>'
    )
    for i, p in enumerate(points):
        color = PALETTE[i % len(PALETTE)]
        parts.append(
            f'<circle cx="{sx(p.x):.1f}" cy="{sy(p.y):.1f}" r="6" fill="{color}" fill-opacity="0.85">'
            f"<title>{html.escape(p.label)}: x={p.x:.2f}, y={p.y:.3f}</title></circle>"
        )
        ly = mt + 10 + i * 20
        parts.append(f'<circle cx="{ml + pw + 20}" cy="{ly}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{ml + pw + 30}" y="{ly + 4}" fill="#1f2933">{html.escape(p.label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
