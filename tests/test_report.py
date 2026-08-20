from __future__ import annotations

import json

from ftlab.artifacts import create_run_contract, update_manifest, write_status
from ftlab.report import build_report


def test_report_builds_aggregate_and_plot(tmp_path) -> None:
    evaluation = tmp_path / "runs" / "run-1"
    evaluation.mkdir(parents=True)
    (evaluation / "evaluation.json").write_text(
        json.dumps({"count": 1, "task_success": 1.0}), encoding="utf-8"
    )
    result = build_report(tmp_path)
    assert result["rows"] == 1
    assert (tmp_path / "reports" / "aggregate.json").exists()
    assert (tmp_path / "reports" / "aggregate.csv").exists()
    assert (tmp_path / "reports" / "task-success.png").stat().st_size > 0
    robustness = json.loads((tmp_path / "reports" / "robustness.json").read_text())
    assert robustness["suite"] == "synthetic_no_call"


def test_report_emits_all_stable_figures_and_skips_legacy(tmp_path) -> None:
    run = create_run_contract(tmp_path, "controlled", kind="evaluation", config={})
    run.path("predictions.jsonl").write_text('{"prediction":"x"}\n')
    run.path("evaluation.json").write_text(json.dumps({"count": 1, "task_success": 0.5}))
    update_manifest(run, controlled=True)
    write_status(run, "completed")
    result = build_report(tmp_path)
    assert result["eligible_rows"] == 1
    for name in (
        "quality-vs-rss-pareto.png",
        "headline-precision-table.png",
        "rank-by-layer-heatmap.png",
        "data-scaling-curve.png",
        "distractor-robustness-curve.png",
        "model-size-quality-throughput-frontier.png",
        "error-distribution-comparison.png",
    ):
        assert (tmp_path / "reports" / name).is_file()
