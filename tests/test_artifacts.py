from __future__ import annotations

import json

from ftlab.artifacts import (
    append_system_metric,
    append_training_metric,
    controlled_run_eligible,
    create_run_contract,
    update_manifest,
    validate_run_artifacts,
    write_status,
)


def test_controlled_eligibility_requires_matching_pinned_evidence() -> None:
    base = {
        "artifact_version": 2,
        "prompt_contract_version": 1,
        "dataset_prompt_contract_version": 1,
        "git": {"commit": "abc"},
        "hashes": {
            "model": "m",
            "tokenizer_files": "t",
            "chat_template": "c",
            "golden_tokens": "g",
        },
        "counts": {"examples": 4, "updates": 2},
        "comparable_rendering": True,
        "safety_probe": {"status": "safe", "model_hash": "m"},
    }
    assert controlled_run_eligible(
        base,
        registered_model=True,
        model_hash="m",
        rendering_hashes={"tokenizer_files": "t", "chat_template": "c", "golden_tokens": "g"},
        expected_examples=4,
        expected_updates=2,
        require_safe_probe=True,
    )
    assert not controlled_run_eligible(
        {**base, "hashes": {**base["hashes"], "chat_template": "other"}},
        registered_model=True,
        model_hash="m",
        rendering_hashes={"tokenizer_files": "t", "chat_template": "c", "golden_tokens": "g"},
        expected_examples=4,
        expected_updates=2,
        require_safe_probe=True,
    )


def test_run_contract_is_sanitized_and_validated(tmp_path) -> None:
    run = create_run_contract(
        tmp_path, "run-1", kind="training", config={"path": str(tmp_path / "secret")}
    )
    payload = json.loads((run.run_dir / "manifest.json").read_text())
    assert str(tmp_path) not in json.dumps(payload)
    assert validate_run_artifacts(run.run_dir, kind="training") == []
    (run.run_dir / "adapter" / "adapters.safetensors").write_bytes(b"adapter")
    append_training_metric(run, {"train_loss": 1.0})
    (run.run_dir / "train.log").write_text("safe log\n")
    append_system_metric(run, {"rss_gib": 1.0})
    write_status(run, "completed")
    assert validate_run_artifacts(run.run_dir, kind="training") == []


def test_completed_run_rejects_empty_required_outputs_and_required_metadata(tmp_path) -> None:
    run = create_run_contract(tmp_path, "required", kind="training", config={})
    write_status(run, "completed")
    assert validate_run_artifacts(run.run_dir, kind="training") == [
        "adapter",
        "system_metrics.csv",
        "train.log",
        "training_metrics.jsonl",
    ]
    (run.run_dir / "adapter" / "adapters.safetensors").write_bytes(b"adapter")
    (run.run_dir / "training_metrics.jsonl").write_text('{"loss": 1}\n')
    (run.run_dir / "train.log").write_text("safe log\n")
    append_system_metric(run, {"rss_gib": 1.0})
    update_manifest(
        run,
        requirements={"tokenizer_hash": True, "checkpoint_bytes": True},
    )
    assert validate_run_artifacts(run.run_dir, kind="training") == [
        "manifest.bytes.checkpoint",
        "manifest.hashes.tokenizer",
    ]
    update_manifest(run, hashes={"tokenizer": "hash"}, bytes={"checkpoint": 42})
    assert validate_run_artifacts(run.run_dir, kind="training") == []


def test_manifest_updates_are_nested(tmp_path) -> None:
    run = create_run_contract(tmp_path, "run-2", kind="evaluation", config={})
    update_manifest(run, counts={"examples": 4}, hashes={"adapter": "abc"})
    payload = json.loads((run.run_dir / "manifest.json").read_text())
    assert payload["counts"]["examples"] == 4
    assert payload["hashes"]["adapter"] == "abc"


def test_manifest_redacts_credentials_and_records_command(tmp_path) -> None:
    run = create_run_contract(
        tmp_path,
        "privacy",
        kind="evaluation",
        config={
            "username": "alice",
            "api_token": "do-not-write",
            "path": str(tmp_path / "nested"),
        },
        command=["ftlab", "evaluate", "--manifest", str(tmp_path / "manifest.json")],
    )
    payload = json.loads((run.run_dir / "manifest.json").read_text())
    serialized = json.dumps(payload)
    assert "do-not-write" not in serialized
    assert "alice" not in serialized
    assert str(tmp_path) not in serialized
    assert payload["command"][0] == "ftlab"
