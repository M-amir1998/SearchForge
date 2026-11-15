"""CPU end-to-end flows: regression gate failure on a bad candidate, CLI, full demo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from searchforge.cli.main import app
from searchforge.config.loader import load_config
from searchforge.config.models import GateConfig
from searchforge.datasets import fixture_config_path
from searchforge.evaluation.datasets import load_qrels, load_queries
from searchforge.evaluation.regression import compare_runs, evaluate_gates
from searchforge.evaluation.runner import evaluate
from searchforge.indexes.builder import IndexBuilder
from searchforge.indexes.handle import open_index
from searchforge.service.search_service import SearchService

pytestmark = pytest.mark.e2e
runner = CliRunner()


def test_bm25_baseline_vs_intentionally_bad_candidate_fails_gate(tmp_path: Path) -> None:
    """A dense retriever on 8-dimensional hash embeddings is near-random: the gate must catch it."""
    ws = {"workspace": str(tmp_path / "ws")}
    base_cfg = load_config(fixture_config_path(), overrides=ws)
    bad_cfg = load_config(
        fixture_config_path(),
        overrides={**ws, "index": {"name": "bad-hash8", "dense": {"provider": "hash", "dimension": 8}}},
    )
    ev = base_cfg.config.evaluation
    assert ev is not None
    queries, qrels = load_queries(base_cfg.resolve(ev.queries)), load_qrels(base_cfg.resolve(ev.qrels))
    runs = []
    for loaded, profile in ((base_cfg, "lexical"), (bad_cfg, "dense")):
        IndexBuilder(loaded).build()
        handle = open_index(loaded.workspace, loaded.config.index.name, loaded.config)
        service = SearchService(handle, loaded.config)
        runs.append(evaluate(service, queries, qrels, profile=profile, name=profile))
        service.close()
        handle.close()
    comparison = compare_runs(runs[0], runs[1])
    assert comparison.summary["regressed"] > comparison.summary["improved"]
    assert comparison.summary["mean_deltas"]["ndcg@10"] < -0.1
    assert comparison.critical_regressions  # critical identifier lookups break
    report = evaluate_gates(comparison, runs[1], GateConfig())
    assert not report.passed
    failed = {c.name for c in report.checks if not c.passed}
    assert {"ndcg_at_10_min_delta", "max_query_regression_rate", "no_critical_regressions"} <= failed
    same = evaluate_gates(compare_runs(runs[0], runs[0]), runs[0], GateConfig())
    assert same.passed


def test_cli_index_search_compare_and_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("include: pkg:fixture/fixture.yaml\nworkspace: ws\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "include: pkg:fixture/fixture.yaml\nworkspace: ws\nindex:\n  name: bad\n  dense:\n"
        "    provider: hash\n    dimension: 8\n",
        encoding="utf-8",
    )
    for args in (
        ["doctor"],
        ["corpus", "inspect", str(cfg)],
        ["index", "build", str(cfg)],
        ["index", "build", str(bad)],
        ["index", "list", "-c", str(cfg)],
        ["index", "validate", "fixture", "-c", str(cfg)],
        ["index", "inspect", "fixture", "-c", str(cfg)],
        ["search", "ERR-4012", "-c", str(cfg), "-p", "adaptive", "--json"],
        ["search", "rollback", "-c", str(cfg), "-f", "category=operations", "--explain"],
        ["corpus", "prepare", str(cfg)],
        ["storage", "status", "-c", str(cfg)],
        ["storage", "clean", "-c", str(cfg)],
        ["cache", "stats", "-c", str(cfg)],
        ["evaluate", str(cfg), "-p", "lexical", "--out", str(tmp_path / "results"), "--repeats", "1"],
        ["hard-negatives", "build", str(cfg), "--out", str(tmp_path / "hn.jsonl")],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)
    run_dir = next((tmp_path / "results").iterdir())
    for name in (
        "config.yaml",
        "manifest.json",
        "metrics.json",
        "per_query.parquet",
        "regressions.parquet",
        "failures.parquet",
        "performance.json",
    ):
        assert (run_dir / name).exists(), name
    assert runner.invoke(app, ["report", str(run_dir)]).exit_code == 0 and (run_dir / "report.md").exists()
    ok = runner.invoke(app, ["compare", "lexical", "adaptive", "-c", str(cfg), "--split", "test"])
    assert ok.exit_code == 0, ok.output
    gate = runner.invoke(
        app,
        [
            "compare",
            "lexical",
            "dense",
            "-c",
            str(cfg),
            "--candidate-config",
            str(bad),
            "--out",
            str(tmp_path / "cmp.json"),
        ],
    )
    assert gate.exit_code == 1 and "FAIL" in gate.output
    assert json.loads((tmp_path / "cmp.json").read_text())["gates"]["passed"] is False
    assert runner.invoke(app, ["search", "x", "-c", str(cfg), "--filters", '{"nope": 1}']).exit_code == 2
    assert runner.invoke(app, ["index", "inspect", "missing", "-c", str(cfg)]).exit_code == 2


def test_full_cpu_demo(tmp_path: Path) -> None:
    out, ws = tmp_path / "demo", tmp_path / "ws"
    result = runner.invoke(app, ["demo", "--out", str(out), "--workspace", str(ws)])
    assert result.exit_code == 0, result.output
    assert "LOCAL DEMONSTRATION RESULTS" in result.output
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "| System | nDCG@10 | Recall@20 | MRR | p95 Latency | Index Size |" in report
    assert "| Configuration | Keyword | Semantic | Identifier | Ambiguous | Filter |" in report
    for name in (
        "plots/quality_latency.svg",
        "metrics.json",
        "gates.json",
        "reliability_export.json",
        "hard_negatives.jsonl",
        "examples.json",
    ):
        assert (out / name).exists(), name
    assert json.loads((out / "examples.json").read_text())["citation_validity"]["valid_rate"] == 1.0
    adaptive = runner.invoke(app, ["demo", "--out", str(out), "--workspace", str(ws), "adaptive"])
    assert adaptive.exit_code == 0, adaptive.output
    assert "identifier" in adaptive.output and (out / "adaptive_report.md").exists()
