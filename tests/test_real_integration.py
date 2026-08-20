from __future__ import annotations

import os

import pytest

from ftlab.config import TrainingConfig
from ftlab.data import AcceptedRecord, normalize_record
from ftlab.modeling import (
    assert_adapter_layout,
    assert_pinned_architecture,
    generate_predictions,
    load_model,
    tokenize_prompt_completion,
)
from ftlab.parser import parse_prediction
from ftlab.rendering import load_tokenizer, render_record, render_tools_prompt, rendering_metadata
from ftlab.storage import cache_root


@pytest.mark.real
@pytest.mark.skipif(
    os.environ.get("FTLAB_RUN_REAL") != "1", reason="set FTLAB_RUN_REAL=1 for the model gate"
)
def test_pinned_tokenizer_golden(sample_record: dict[str, object]) -> None:
    tokenizer = load_tokenizer(
        "Qwen/Qwen3-0.6B-MLX-4bit",
        revision="173234aa840d113125e9f2271100ddbaf16c9620",
        cache_root=cache_root(os.getcwd()),
    )
    record = normalize_record(sample_record, "golden")
    assert isinstance(record, AcceptedRecord)
    rendered = render_record(record, tokenizer)
    inference_prompt = render_tools_prompt(list(record.tools), record.query, tokenizer)
    assert rendered.prompt + rendered.completion == rendered.full
    assert inference_prompt == rendered.prompt
    metadata = rendering_metadata(record, tokenizer)
    assert metadata["prompt_hash"] == metadata["tools_query_prompt_hash"]
    assert rendered.full_tokens[: len(rendered.prompt_tokens)] == rendered.prompt_tokens
    assert "<think>" not in rendered.completion
    assert rendered.target_start > 0
    assert parse_prediction(rendered.completion, record.tools).valid


@pytest.mark.real
@pytest.mark.skipif(
    os.environ.get("FTLAB_RUN_REAL") != "1", reason="set FTLAB_RUN_REAL=1 for the model gate"
)
def test_real_training_tokens_match_rendered_record(sample_record: dict[str, object]) -> None:
    tokenizer = load_tokenizer(
        "Qwen/Qwen3-0.6B-MLX-4bit",
        revision="173234aa840d113125e9f2271100ddbaf16c9620",
        cache_root=cache_root(os.getcwd()),
    )
    config = TrainingConfig()
    record = normalize_record(sample_record, "training-golden")
    assert isinstance(record, AcceptedRecord)
    rendered = render_record(record, tokenizer)
    assert render_tools_prompt(list(record.tools), record.query, tokenizer) == rendered.prompt
    tokens, offset = tokenize_prompt_completion(
        rendered.prompt,
        rendered.completion,
        tokenizer,
        max_seq_length=config.max_seq_length,
    )
    assert tokens == list(rendered.full_tokens)
    assert offset == len(rendered.prompt_tokens)
    assert tokenizer.decode(tokens[offset:], skip_special_tokens=False).startswith("<tool_call>")

    exact_record = None
    exact_rendered = None
    for count in range(1, config.max_seq_length * 2):
        candidate = dict(sample_record)
        candidate["query"] = "x" + " x" * count
        accepted = normalize_record(candidate, f"max-{count}")
        if isinstance(accepted, AcceptedRecord):
            candidate_rendered = render_record(accepted, tokenizer)
            if len(candidate_rendered.full_tokens) == config.max_seq_length:
                exact_record = accepted
                exact_rendered = candidate_rendered
                break
    assert exact_record is not None
    assert exact_rendered is not None
    exact_tokens, exact_offset = tokenize_prompt_completion(
        exact_rendered.prompt,
        exact_rendered.completion,
        tokenizer,
        max_seq_length=config.max_seq_length,
    )
    assert len(exact_tokens) == config.max_seq_length
    assert exact_tokens == list(exact_rendered.full_tokens)
    assert exact_offset == len(exact_rendered.prompt_tokens)
    assert exact_tokens[-1] != tokenizer.eos_token_id


@pytest.mark.real
@pytest.mark.skipif(
    os.environ.get("FTLAB_RUN_REAL") != "1", reason="set FTLAB_RUN_REAL=1 for the model gate"
)
def test_pinned_model_architecture_and_generation(sample_record: dict[str, object]) -> None:
    root = os.getcwd()
    model, tokenizer = load_model(
        "Qwen/Qwen3-0.6B-MLX-4bit",
        revision="173234aa840d113125e9f2271100ddbaf16c9620",
        cache_root=cache_root(root),
    )
    assert_pinned_architecture(model)
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers

    linear_to_lora_layers(
        model,
        8,
        {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        },
    )
    manifest = assert_adapter_layout(model, expected_trainable=327680)
    assert len(manifest["modules"]) == 16
    record = normalize_record(sample_record, "golden")
    assert isinstance(record, AcceptedRecord)
    rendered = render_record(record, tokenizer)
    output = generate_predictions(model, tokenizer, [rendered.prompt], max_tokens=8)[0]
    assert isinstance(output, str)
