"""Render the README "Results" section from generated result files (never hand-typed numbers).

Usage: uv run python scripts/update_readme_results.py
Reads results/local-demo/{metrics.json, adaptive_metrics.json, fusion_tuning.json}
and results/beir-scifact/metrics.json, and rewrites the block between the
RESULTS:BEGIN / RESULTS:END markers in README.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"


def load(path: str) -> dict[str, Any] | None:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(v: Any) -> str:
    return "-" if v is None else f"{v:.3f}"


def size(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


def main_table(systems: list[dict[str, Any]], groups: tuple[str, ...] = ("core",)) -> list[str]:
    out = [
        "| System | nDCG@10 | Recall@20 | MRR | p50 | p95 Latency | QPS | Index Size |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in systems:
        if s["group"] in groups:
            out.append(
                f"| {s['system']} | {f3(s['ndcg@10'])} | {f3(s['recall@20'])} | {f3(s['mrr'])} | "
                f"{s['p50_ms']:.2f} ms | {s['p95_ms']:.2f} ms | {s['qps']:.0f} | {size(s['index_bytes'])} |"
            )
    return out


def render() -> str:
    lines: list[str] = []
    demo = load("results/local-demo/metrics.json")
    if demo:
        env = demo["environment"]
        lines += [
            "### LOCAL DEMONSTRATION RESULTS",
            "",
            "> Synthetic fixture collection (48 documents, 20 held-out **test** queries), CPU only, measured by",
            "> `searchforge demo`. These numbers demonstrate the methodology; they are not industry benchmarks,",
            "> and differences of a few hundredths are within noise at this size.",
            f"> Environment: Python {env['python']}, {env['platform']}, {env['logical_cpus']} logical CPUs, "
            f"{env['ram_gb']} GB RAM, no GPU.",
            "",
            *main_table(demo["systems"]),
            "",
            "Quality by query type (nDCG@10):",
            "",
            "| Configuration | Keyword | Semantic | Identifier | Ambiguous | Filter |",
            "|---|---|---|---|---|---|",
        ]
        for s in demo["systems"]:
            if s["group"] == "core":
                bt = demo["by_query_type"][s["system"]]
                lines.append(
                    f"| {s['system']} | "
                    + " | ".join(
                        f3(bt.get(t, {}).get("ndcg@10"))
                        for t in ("keyword", "semantic", "identifier", "ambiguous", "filter")
                    )
                    + " |"
                )
        depth = [s for s in demo["systems"] if s["group"] == "depth"]
        chunking = [s for s in demo["systems"] if s["group"] == "chunking"]
        lines += [
            "",
            "Candidate depth (hybrid + lexical-feature reranker) and chunking experiments:",
            "",
            "| System | Chunks | nDCG@10 | Recall@20 | p95 |",
            "|---|---|---|---|---|",
        ]
        for s in depth + chunking:
            lines.append(
                f"| {s['system']} | {s['chunks']} | {f3(s['ndcg@10'])} | {f3(s['recall@20'])} | {s['p95_ms']:.2f} ms |"
            )
        tuning = load("results/local-demo/fusion_tuning.json")
        if tuning:
            lines += [
                "",
                f"Fusion tuning ({tuning['strategy']}, weights chosen on validation, reported once on test): "
                f"best weights {tuning['best_weights']}, test nDCG@10 {tuning['test']:.3f} vs "
                f"unweighted RRF {tuning['baseline_test']:.3f}.",
            ]
        adaptive = load("results/local-demo/adaptive_metrics.json")
        if adaptive:
            routed = adaptive["routing"]
            total = sum(routed.values())
            comp = adaptive["comparison"]
            lines += [
                "",
                "Adaptive planning vs always hybrid + rerank (all 56 fixture queries, `searchforge demo adaptive`):",
                "",
                "| System | nDCG@10 | Recall@20 | MRR | p50 | p95 | QPS |",
                "|---|---|---|---|---|---|---|",
            ]
            for s in adaptive["systems"]:
                lines.append(
                    f"| {s['system']} | {f3(s['ndcg@10'])} | {f3(s['recall@20'])} | {f3(s['mrr'])} | "
                    f"{s['p50_ms']:.2f} ms | {s['p95_ms']:.2f} ms | {s['qps']:.0f} |"
                )
            lines += [
                "",
                f"The planner skipped the reranker for {total - routed.get('hybrid_rerank', 0)}/{total} queries "
                f"and dense retrieval for {routed.get('lexical', 0)}/{total}; vs always-hybrid-rerank: "
                f"{comp['improved']} improved, {comp['regressed']} regressed, {comp['unchanged']} unchanged "
                f"(ΔnDCG@10 {comp['mean_deltas']['ndcg@10']:+.3f}).",
            ]
        lines += [
            "",
            "Full report: [results/local-demo/report.md](results/local-demo/report.md) · "
            "[adaptive](results/local-demo/adaptive_report.md) · "
            "![quality vs latency](results/local-demo/plots/quality_latency.svg)",
            "",
        ]
    sci = load("results/beir-scifact/metrics.json")
    if sci:
        lines += [
            "### First public-dataset study: BEIR SciFact (CPU)",
            "",
            "> 5,183 abstracts, 300 test claims; dense = TF-IDF + LSA (256 d), not a neural model.",
            "> Produced by `searchforge benchmark configs/benchmarks/scifact.yaml`; the dataset is downloaded locally",
            "> and never committed.",
            "",
            *main_table(sci["systems"]),
            "",
        ]
        best = max(sci["systems"], key=lambda s: s["ndcg@10"] or 0)
        comps = sci["comparisons"]
        lines += [
            f"Measured outcome: **{best['system']}** has the best nDCG@10 here. Against the BM25 baseline, "
            + "; ".join(
                f"{name}: {c['improved']} improved / {c['regressed']} regressed queries "
                f"(ΔnDCG@10 {c['mean_deltas']['ndcg@10']:+.3f}, Δp95 {c['latency']['p95_delta_ms']:+.1f} ms)"
                for name, c in comps.items()
            )
            + ". With a weak classical dense retriever, adding hybrid fusion and reranking costs latency "
            "without improving relevance on this dataset — exactly the trade-off SearchForge is built to "
            "measure before adopting a more complex stack. The neural-embedding study is the next step.",
            "",
            "Full report: [results/beir-scifact/report.md](results/beir-scifact/report.md)",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    block = f"{BEGIN}\n{render()}\n{END}"
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    readme.write_text(new, encoding="utf-8", newline="\n")
    print("README results section updated")


if __name__ == "__main__":
    main()
