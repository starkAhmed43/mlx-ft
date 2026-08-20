from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from ftlab import cli as cli_module
from ftlab.cli import _exclude_locked_duplicate_components, _sweep_dataset_paths, app
from ftlab.config import load_config
from ftlab.data import AcceptedRecord, normalize_record
from ftlab.rendering import render_json_call


def test_cli_exposes_report_build() -> None:
    result = CliRunner().invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "build" in result.stdout


def test_context_sweep_requires_and_selects_context_view(tmp_path) -> None:
    class Dataset:
        version = "nested_5k"

    class Experiment:
        dataset = Dataset()

    base = tmp_path / "data" / "nested_5k"
    (base / "contexts" / "1024").mkdir(parents=True)
    (base / "contexts" / "1024" / "train.jsonl").write_text("train\n")
    (base / "contexts" / "1024" / "validation.jsonl").write_text("valid\n")
    train, validation = _sweep_dataset_paths(tmp_path, Experiment(), 1024)
    assert train.name == "train.jsonl"
    assert validation.parent.name == "1024"


def test_context_prepare_keeps_locked_splits_and_recovers_long_record(
    tmp_path, monkeypatch
) -> None:
    records = [
        {
            "id": f"context-{index}",
            "query": f"Question {index}",
            "tools": [{"name": "fixture.echo", "parameters": {"type": "object", "properties": {}}}],
            "answers": [{"name": "fixture.echo", "arguments": {}}],
            "rendered_length": 700 if index == 224 else 400,
        }
        for index in range(225)
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    base = CliRunner().invoke(
        app, ["data", "prepare", "--version", "smoke", "--source", str(source)]
    )
    assert base.exit_code == 0, base.output
    context = CliRunner().invoke(
        app,
        [
            "data",
            "prepare",
            "--version",
            "smoke",
            "--source",
            str(source),
            "--max-seq-length",
            "1024",
        ],
    )
    assert context.exit_code == 0, context.output
    payload = json.loads((tmp_path / "data" / "smoke" / "manifest.json").read_text())
    context_train = payload["context_views"]["1024"]["counts"]["train"]
    assert context_train == 129
    assert any(
        json.loads(line)["prompt"].endswith("Question 224\n<|assistant|>\n")
        for line in (tmp_path / "data" / "smoke" / "contexts" / "1024" / "train.jsonl")
        .read_text()
        .splitlines()
    )


def test_benchmark_probe_cli_passes_32_microbatch_request(tmp_path, monkeypatch) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "experiments" / "smoke-06b.yaml"
    experiment = load_config(config_path)
    captured = {}

    def fake_run(contract, request):
        captured.update(request)
        contract.path("system_metrics.csv").write_text(
            "memory_available_gib,swap_used_gib,disk_free_gib,pressure_state,rss_gib\n"
            "8,0,50,normal,1\n",
            encoding="utf-8",
        )
        from ftlab.runner import RunnerResult

        return RunnerResult(0, None, "completed", {}, None, 1)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "load_config", lambda _: experiment)
    monkeypatch.setattr(cli_module, "run_isolated", fake_run)
    result = CliRunner().invoke(
        app, ["benchmark", "probe", "--config", str(config_path), "--steps", "32"]
    )
    assert result.exit_code == 0, result.output
    assert captured["mode"] == "probe"
    assert captured["steps"] == 32
    assert captured["config"]["iters"] == 32


def test_cli_validates_manifest(tmp_path, monkeypatch) -> None:
    (tmp_path / "data").mkdir()
    manifest = {"splits": {"train": [{"query": "x"}]}}
    (tmp_path / "data" / "smoke.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["data", "validate", "--version", "smoke"])
    assert result.exit_code == 1
    assert "size" in result.stdout or "size" in result.stderr


def test_fixture_prepare_validate_and_tamper(tmp_path, monkeypatch) -> None:
    records = [
        {
            "id": f"fixture-{index}",
            "query": f"Question {index}",
            "tools": [{"name": "fixture.echo", "parameters": {"type": "object", "properties": {}}}],
            "answers": [{"name": "fixture.echo", "arguments": {}}],
        }
        for index in range(224)
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    shutil.copy(Path(__file__).parents[1] / "uv.lock", tmp_path / "uv.lock")
    monkeypatch.chdir(tmp_path)
    prepared = CliRunner().invoke(
        app, ["data", "prepare", "--version", "smoke", "--source", str(source)]
    )
    assert prepared.exit_code == 0, prepared.output
    validated = CliRunner().invoke(app, ["data", "validate", "--version", "smoke"])
    assert validated.exit_code == 0, validated.output
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps([render_json_call("fixture.echo", {})] * 64), encoding="utf-8"
    )
    evaluated = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--model",
            "fixture",
            "--manifest",
            "data/smoke/manifest.json",
            "--predictions",
            str(predictions),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    run_dirs = list((tmp_path / "runs").iterdir())
    assert run_dirs
    assert (run_dirs[0] / "predictions.raw.json").exists()
    assert (run_dirs[0] / "evaluation.json").exists()
    evaluation_manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert evaluation_manifest["controlled"] is False
    processed = tmp_path / "data" / "smoke" / "train.jsonl"
    processed.write_text(processed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = CliRunner().invoke(app, ["data", "validate", "--version", "smoke"])
    assert tampered.exit_code == 1
    assert "processed hash" in tampered.stderr


def test_robustness_fixture_predictions_are_uncontrolled(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "robustness.json"
    predictions.write_text(json.dumps(["no call"] * 100), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "robustness",
            "evaluate",
            "--model",
            "Qwen/Qwen3-0.6B-MLX-4bit",
            "--predictions",
            str(predictions),
        ],
    )
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["controlled"] is False


def test_train_cli_marks_mocked_job_uncontrolled(tmp_path, monkeypatch) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "experiments" / "smoke-06b.yaml"
    experiment = load_config(config_path)
    data = tmp_path / "data" / "smoke"
    data.mkdir(parents=True)
    for name in ("train.jsonl", "validation.jsonl"):
        (data / name).write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")
    (data / "manifest.json").write_text('{"splits": {}}', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock\n", encoding="utf-8")

    def fake_run(contract, request):
        adapter = contract.path("adapter") / "adapters.safetensors"
        adapter.write_bytes(b"adapter")
        from ftlab.runner import RunnerResult

        return RunnerResult(
            0,
            None,
            "completed",
            {
                "microbatches": 128,
                "optimizer_updates": 16,
                "training_metrics": [{"loss": 1.0}],
                "manifest": {"trainable_parameters": 1},
                "prompt_tokens": 1,
                "target_tokens": 1,
            },
            None,
            1,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "load_config", lambda _: experiment)
    monkeypatch.setattr(cli_module, "run_isolated", fake_run)
    result = CliRunner().invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["controlled"] is False


def test_robustness_worker_cli_marks_job_uncontrolled(tmp_path, monkeypatch) -> None:
    def fake_run(contract, request):
        contract.path("predictions.jsonl").write_text(
            "".join(
                json.dumps({"case_id": str(index), "prediction": "no call"}) + "\n"
                for index in range(100)
            ),
            encoding="utf-8",
        )
        from ftlab.runner import RunnerResult

        return RunnerResult(
            0,
            None,
            "completed",
            {"evaluation": {}, "mlx_allocator_peak_bytes": 123},
            None,
            1,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "run_isolated", fake_run)
    result = CliRunner().invoke(
        app,
        ["robustness", "evaluate", "--model", "Qwen/Qwen3-0.6B-MLX-4bit"],
    )
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["controlled"] is False


def test_full_training_excludes_transitive_locked_duplicate_component(
    sample_record: dict[str, object],
) -> None:
    def make(source_id: str, query: str, task: str) -> AcceptedRecord:
        value = dict(sample_record)
        value["id"] = source_id
        value["query"] = query
        result = normalize_record(value, source_id)
        assert isinstance(result, AcceptedRecord)
        return replace(result, normalized_task=task)

    locked = make("locked", "locked-query", "locked-task")
    query_edge = make("query-edge", "locked-query", "query-edge-task")
    task_edge = make("task-edge", "other-query", "query-edge-task")
    independent = make("independent", "independent-query", "independent-task")
    selected = _exclude_locked_duplicate_components([query_edge, task_edge, independent], [locked])
    assert [item.source_id for item in selected] == ["independent"]
