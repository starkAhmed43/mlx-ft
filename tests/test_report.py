from __future__ import annotations

import json
from pathlib import Path

from ftlab.artifacts import append_system_metric, create_run_contract, update_manifest, write_status
from ftlab.data import normalize_record
from ftlab.metrics import evaluate_predictions
from ftlab.rendering import render_json_call
from ftlab.report import aggregate_evaluations, build_report, nondominated_pareto


def _record(source_id: str = "record-1") -> dict[str, object]:
    return {
        "id": source_id,
        "query": "Find weather.",
        "tools": [
            {
                "name": "weather.get",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ],
        "answers": [{"name": "weather.get", "arguments": {"city": "Delhi"}}],
    }


def _completed_run(
    root: Path,
    run_id: str,
    *,
    workflow: str = "validation",
    role: str = "headline",
    axis: str = "precision",
    prediction_rows: list[dict[str, object]] | None = None,
    evaluation: dict[str, object] | None = None,
    task_success: float = 0.5,
) -> Path:
    contract = create_run_contract(
        root,
        run_id,
        kind="evaluation",
        config={
            "model": "fixture-model",
            "workflow": workflow,
            "study_role": role,
            "axis": axis,
            "dataset": {"version": "core"},
        },
        model_hash="model-hash",
        dataset_hash="dataset-hash",
    )
    rows = prediction_rows or [{"source_id": "record-1", "prediction": "not audited"}]
    contract.path("predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    contract.path("evaluation.json").write_text(
        json.dumps(evaluation or {"count": len(rows), "task_success": task_success}),
        encoding="utf-8",
    )
    append_system_metric(contract, {"rss_gib": 1.0})
    update_manifest(
        contract,
        controlled=True,
        comparable_rendering=True,
        dataset_prompt_contract_version=1,
        prompt_hash="shared-prompt",
        decoding_hash="shared-decoding",
        git={"commit": "a" * 40},
    )
    write_status(contract, "completed")
    return contract.run_dir


def test_report_no_data_is_preliminary(tmp_path) -> None:
    result = build_report(tmp_path)
    assert result["eligible_rows"] == 0
    assert "preliminary" in (tmp_path / "reports" / "findings.md").read_text()
    assert json.loads((tmp_path / "reports" / "pareto.json").read_text()) == []


def test_report_accepts_indirect_dataset_provenance_and_axis_filters(tmp_path) -> None:
    _completed_run(tmp_path, "headline")
    _completed_run(tmp_path, "scaling", role="data_context_batch", axis="dataset_size")
    result = build_report(tmp_path)
    assert result["eligible_rows"] == 2
    assert (
        len((tmp_path / "reports" / "headline-precision-table.csv").read_text().splitlines()) == 2
    )
    scaling = (tmp_path / "reports" / "data-scaling.csv").read_text().splitlines()
    assert len(scaling) == 2
    assert "scaling" in scaling[1]


def test_report_rejects_auditable_metric_mismatch(tmp_path) -> None:
    raw = _record()
    record = normalize_record(raw, "record-1")
    good = render_json_call(record.function_name, record.arguments)
    measured = evaluate_predictions([record], [good], confidence=False).as_dict()
    measured["task_success"] = 0.0
    _completed_run(
        tmp_path,
        "mismatch",
        prediction_rows=[{"source_id": "record-1", "record": raw, "prediction": good}],
        evaluation=measured,
    )
    assert aggregate_evaluations(tmp_path / "runs") == []


def test_report_audits_worker_rows_without_queries(tmp_path) -> None:
    raw = _record()
    record = normalize_record(raw, "record-1")
    good = render_json_call(record.function_name, record.arguments)
    metrics = evaluate_predictions([record], [good], confidence=False).as_dict()
    worker_record = {key: value for key, value in raw.items() if key != "query"}
    _completed_run(
        tmp_path,
        "worker-row",
        prediction_rows=[{"source_id": "record-1", "record": worker_record, "prediction": good}],
        evaluation=metrics,
    )
    assert [row["run"] for row in aggregate_evaluations(tmp_path / "runs")] == ["worker-row"]


def test_nondominated_pareto_excludes_dominated_rows() -> None:
    rows = [
        {
            "run": "winner",
            "task_success": 1.0,
            "argument_value_f1": 1.0,
            "schema_validity": 1.0,
            "rss_gib": 1.0,
        },
        {
            "run": "dominated",
            "task_success": 0.9,
            "argument_value_f1": 1.0,
            "schema_validity": 1.0,
            "rss_gib": 2.0,
        },
        {
            "run": "tradeoff",
            "task_success": 1.0,
            "argument_value_f1": 1.0,
            "schema_validity": 1.0,
            "rss_gib": 0.5,
        },
    ]
    assert [row["run"] for row in nondominated_pareto(rows)] == ["tradeoff"]


def test_representatives_are_verified_and_do_not_leak_prompt(tmp_path) -> None:
    raw = _record()
    record = normalize_record(raw, "record-1")
    good = render_json_call(record.function_name, record.arguments)
    bad = '<tool_call>{"name":"other","arguments":{}}</tool_call>'
    base_metrics = evaluate_predictions([record], [bad], confidence=False).as_dict()
    tuned_metrics = evaluate_predictions([record], [good], confidence=False).as_dict()
    _completed_run(
        tmp_path,
        "base",
        workflow="base",
        prediction_rows=[{"source_id": "record-1", "record": raw, "prediction": bad}],
        evaluation=base_metrics,
    )
    _completed_run(
        tmp_path,
        "tuned",
        workflow="tuned",
        prediction_rows=[{"source_id": "record-1", "record": raw, "prediction": good}],
        evaluation=tuned_metrics,
    )
    build_report(tmp_path)
    examples = json.loads((tmp_path / "reports" / "representative-examples.json").read_text())
    intervals = json.loads((tmp_path / "reports" / "paired-delta-intervals.json").read_text())
    assert examples["status"] == "verified"
    assert examples["examples"] == [
        {"source_id": "record-1", "base_prediction": bad, "tuned_prediction": good}
    ]
    assert "Find weather" not in json.dumps(examples)
    assert intervals["status"] == "verified"
    assert intervals["pairs"][0]["source_ids"] == ["record-1"]
    assert set(intervals["pairs"][0]["tuned_minus_base_95ci"]) == {
        "tool_accuracy",
        "json_validity",
        "schema_validity",
        "exact_match",
        "task_success",
        "argument_key_f1",
        "argument_value_f1",
    }


def test_final_evidence_refuses_unavailable_raw_audit(tmp_path) -> None:
    _completed_run(tmp_path, "final-no-raw", workflow="final")
    result = build_report(tmp_path)
    assert result["final_evidence"]["status"] == "nonfinal"
    assert not (tmp_path / "reports" / "final").exists()


def test_final_evidence_uses_sanitized_tracked_paths(tmp_path) -> None:
    raw = _record()
    record = normalize_record(raw, "record-1")
    good = render_json_call(record.function_name, record.arguments)
    metrics = evaluate_predictions([record], [good], confidence=False).as_dict()
    _completed_run(
        tmp_path,
        "final-audited",
        workflow="final",
        prediction_rows=[{"source_id": "record-1", "record": raw, "prediction": good}],
        evaluation=metrics,
    )
    result = build_report(tmp_path)
    final = tmp_path / "reports" / "final"
    assert result["final_evidence"]["status"] == "final"
    assert (final / "tables" / "final-aggregate.csv").is_file()
    assert (final / "figures" / "quality-vs-rss-pareto.png").is_file()
    assert (final / "findings" / "findings.md").is_file()
    assert not list(final.rglob("predictions*"))


def test_report_emits_all_stable_figures(tmp_path) -> None:
    _completed_run(tmp_path, "controlled")
    build_report(tmp_path)
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
