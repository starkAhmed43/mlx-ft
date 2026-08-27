from __future__ import annotations

import json

from ftlab.bfcl import (
    _case_hash,
    _supported_schema,
    build_bfcl_eval_commands,
    evaluate_bfcl,
    export_bfcl_responses,
    prepare_bfcl,
    select_bfcl_cases,
)
from ftlab.worker import bfcl_row_inputs, run_bfcl_request


def test_bfcl_schema_filter_recurses_through_function_descriptor() -> None:
    valid = {
        "id": "nested-valid",
        "function": {
            "name": "fixture.echo",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            },
        },
    }
    invalid = {
        "id": "nested-invalid",
        "function": {
            "name": "fixture.echo",
            "parameters": {
                "type": "object",
                "properties": {"items": {"oneOf": []}},
            },
        },
    }
    assert _supported_schema(valid)
    assert not _supported_schema(invalid)


def test_bfcl_worker_converts_nested_official_row() -> None:
    query, tools = bfcl_row_inputs(
        {
            "id": "official-1",
            "question": [[{"role": "user", "content": "Call the weather tool."}]],
            "function": [
                {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
    )
    assert query == "Call the weather tool."
    assert tools[0]["function"]["name"] == "weather"
    assert tools[0]["function"]["parameters"]["type"] == "object"


def test_bfcl_worker_handles_nested_shape_and_preserves_id(tmp_path, monkeypatch) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "id": "official-1",
                "question": [[{"role": "user", "content": "Call weather."}]],
                "function": [
                    {"name": "weather", "parameters": {"type": "object", "properties": {}}}
                ],
            }
        )
        + "\n"
    )
    predictions = tmp_path / "predictions.jsonl"

    class Tokenizer:
        def encode(self, value, **kwargs):
            return list(range(len(value)))

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    monkeypatch.setattr(
        "ftlab.worker.load_model",
        lambda *args, **kwargs: (object(), Tokenizer()),
        raising=False,
    )
    monkeypatch.setattr(
        "ftlab.modeling.load_model",
        lambda *args, **kwargs: (object(), Tokenizer()),
    )
    monkeypatch.setattr("ftlab.modeling.generate_predictions", lambda *args, **kwargs: ["answer"])
    result = run_bfcl_request(
        {
            "model": "fixture",
            "model_revision": "fixture",
            "records_path": str(records),
            "predictions_path": str(predictions),
        }
    )
    assert result["count"] == 1
    assert json.loads(predictions.read_text())["id"] == "official-1"


def test_bfcl_selection_is_deterministic() -> None:
    rows = [{"id": f"s-{i}", "category": "simple_python"} for i in range(200)]
    rows += [{"id": f"i-{i}", "category": "irrelevance"} for i in range(200)]
    selected = select_bfcl_cases(rows)
    assert len(selected) == 400
    assert [row["id"] for row in selected] == [row["id"] for row in select_bfcl_cases(rows)]


def test_bfcl_commands_cover_both_categories() -> None:
    commands = build_bfcl_eval_commands()
    assert commands == [
        [
            "bfcl",
            "evaluate",
            "--model",
            "mlx",
            "--test-category",
            "simple_python",
            "--partial-eval",
        ],
        ["bfcl", "evaluate", "--model", "mlx", "--test-category", "irrelevance", "--partial-eval"],
    ]


def test_bfcl_prepare_separates_local_records_and_recursively_filters_schema(tmp_path) -> None:
    rows = []
    valid_schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    for category in ("simple_python", "irrelevance"):
        for index in range(201):
            rows.append({"id": f"{category}-{index}", "category": category, "schema": valid_schema})
    rows[0]["schema"] = {"type": "object", "properties": {"x": {"oneOf": []}}}
    source = tmp_path / "source.json"
    target = tmp_path / "manifest.json"
    source.write_text(json.dumps(rows))
    prepare_bfcl(source, target)
    metadata = json.loads(target.read_text())
    assert metadata["count"] == 400
    assert "records" not in metadata
    assert (tmp_path / "manifest.records.jsonl").is_file()


def test_bfcl_export_omits_project_run_id(tmp_path) -> None:
    target = tmp_path / "results.jsonl"
    export_bfcl_responses([{"id": "x", "model_response": "ok"}], target, run_id="private")
    row = json.loads(target.read_text())
    assert "run_id" not in row


def test_bfcl_evaluate_stages_jsonl_rows_and_commands(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "bfcl.manifest.records.jsonl").write_text(
        '{"id":"s-1"}\n{"id":"i-1"}\n', encoding="utf-8"
    )
    (data / "bfcl.manifest.json").write_text(
        json.dumps(
            {
                "records_file": "bfcl.manifest.records.jsonl",
                "rows": [
                    {"id": "s-1", "category": "simple_python", "hash": _case_hash({"id": "s-1"})},
                    {"id": "i-1", "category": "irrelevance", "hash": _case_hash({"id": "i-1"})},
                ],
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "runs" / "fixture"
    run.mkdir(parents=True)
    (run / "predictions.jsonl").write_text(
        '{"id":"s-1","prediction":"one"}\n{"id":"i-1","prediction":"two"}\n',
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return __import__("subprocess").CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("ftlab.bfcl.subprocess.run", fake_run)
    monkeypatch.setattr("ftlab.bfcl.bfcl_preflight", lambda: {"bfcl_version": "2026.3.23"})
    monkeypatch.setattr(
        "ftlab.study.load_final_lock",
        lambda root: {"candidates": ["fixture"], "content_sha256": "fixture"},
    )
    evaluate_bfcl(run, project_root=tmp_path)
    staged = tmp_path / "result" / "mlx_ftlab" / "BFCL_v4_simple_python_result.json"
    lines = staged.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"id": "s-1", "result": "one"}
    assert all("--result-dir" in command for command in calls)
    assert all(command[:4] == ["conda", "run", "-n", "mlx-ft-bfcl"] for command in calls)
