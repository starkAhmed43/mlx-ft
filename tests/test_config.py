from __future__ import annotations

from pathlib import Path

from ftlab.config import load_config


def test_smoke_config_is_exactly_exposed() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/experiments/smoke-06b.yaml")
    assert config.model == "Qwen/Qwen3-0.6B-MLX-4bit"
    assert config.training.iters == 128
    assert config.training.grad_accumulation_steps == 8
    assert config.training.num_layers == 8
    assert config.training.lora_parameters.scale == 20.0
