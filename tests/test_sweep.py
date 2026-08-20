from __future__ import annotations

from ftlab.sweep import SweepConfig, expand_matrix, run_sweep, select_best


def test_sweep_expansion_and_resume(tmp_path) -> None:
    config = SweepConfig(axes={"training.rank": [4, 8], "lr": [1, 2]})
    runs = expand_matrix(config)
    assert len(runs) == 4
    result_path = tmp_path / "sweep.jsonl"
    first = run_sweep(config, output=result_path, execute=lambda run: {"status": "completed"})
    second = run_sweep(
        config, output=result_path, resume=True, execute=lambda run: {"status": "failed"}
    )
    assert len(first) == len(second) == 4
    assert select_best([{"status": "completed", "task_success": 1.0}]) is not None


def test_run_sweep_without_executor_is_failed_not_planned(tmp_path) -> None:
    result = run_sweep(SweepConfig(axes={"lr": [1]}), output=tmp_path / "planned.jsonl")
    assert result[0]["status"] == "failed"
    assert "planned" not in result[0]


def test_select_best_uses_nested_measured_selection() -> None:
    best = select_best(
        [
            {"status": "completed", "selection": {"task_success": 0.5, "rss_gib": 1.0}},
            {"status": "completed", "selection": {"task_success": 0.8, "rss_gib": 8.0}},
        ]
    )
    assert best is not None
    assert best["selection"]["task_success"] == 0.8


def test_full_study_has_explicit_controlled_roles() -> None:
    from collections import Counter

    runs = expand_matrix("configs/sweeps/full-study.yaml")
    assert len(runs) == 30
    assert Counter(item.values["study_role"] for item in runs) == {
        "lr_pilot": 9,
        "headline": 3,
        "capacity": 8,
        "data_context_batch": 6,
        "size_endpoint": 2,
        "winner_seed": 2,
    }
    for item in runs:
        values = item.values
        assert values["lora_scale"] == 20.0
        if values["study_role"] == "lr_pilot":
            assert values["examples"] == 2496
            assert values["updates"] == 312
            assert values["learning_rate"] in {1e-5, 5e-5, 1e-4}
        if values["study_role"] == "headline":
            assert values["dataset_version"] == "core"
            assert values["dataset_size"] == 10000
        if values["study_role"] == "capacity":
            assert tuple(values["adapter_targets"]) in {
                ("self_attn.q_proj", "self_attn.v_proj"),
                (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ),
                (
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ),
            }
    assert {
        item.values["adapter_rank"] for item in runs if item.values["study_role"] == "capacity"
    } == {
        4,
        8,
        16,
        32,
    }
    assert {
        item.values["adapter_layers"] for item in runs if item.values["study_role"] == "capacity"
    } == {
        4,
        8,
        16,
        28,
    }
    data_rows = [item.values for item in runs if item.values["study_role"] == "data_context_batch"]
    assert {item["dataset_version"] for item in data_rows if item["axis"] == "dataset_size"} == {
        "nested_1k",
        "nested_10k",
        "nested_20k",
    }
    assert {item["context"] for item in data_rows if item["axis"] == "context"} == {1024, 2048}
    batch = next(item for item in data_rows if item["axis"] == "effective_batch")
    assert (batch["batch_size"], batch["grad_accumulation_steps"], batch["effective_batch"]) == (
        2,
        4,
        8,
    )
    assert {item.values["seed"] for item in runs if item.values["study_role"] == "winner_seed"} == {
        123,
        2026,
    }
