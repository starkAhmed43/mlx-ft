from __future__ import annotations

from pathlib import Path

from ftlab.data import AcceptedRecord, normalize_record
from ftlab.rendering import TOOL_END, TOOL_START, render_record, tokenizer_fingerprint


def test_prompt_completion_contract(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    rendered = render_record(record)
    assert rendered.prompt + rendered.completion == rendered.full
    assert rendered.completion.startswith(TOOL_START)
    assert rendered.completion.endswith(TOOL_END)
    assert "think" not in rendered.full.lower()


def test_template_preserves_model_owned_prefix_and_starts_target_at_tool_call(
    sample_record: dict[str, object],
) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            prefix = "prefix<think>\n\n</think>\n\n"
            if kwargs["add_generation_prompt"]:
                return prefix
            return prefix + render_record(record).completion + "<|im_end|>"

        def encode(self, text, **kwargs):
            return list(text.encode())

    rendered = render_record(record, Tokenizer())
    assert "<think>" in rendered.prompt
    assert rendered.completion.startswith(TOOL_START)
    assert rendered.prompt + rendered.completion == rendered.full
    assert rendered.target_start > 0
    assert rendered.completion.endswith("<|im_end|>")


def test_tokenizer_fingerprint_ignores_checkpoint_weights(tmp_path: Path) -> None:
    class Tokenizer:
        chat_template = "fixture-template-v1"

        def apply_chat_template(self, messages, **kwargs):
            prefix = "prompt-prefix"
            if kwargs["add_generation_prompt"]:
                return prefix
            assistant = messages[-1]["content"]
            return prefix + assistant

        def encode(self, text, **kwargs):
            return list(text.encode())

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("tokenizer-v1", encoding="utf-8")
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights-v1")
    tokenizer = Tokenizer()
    first = tokenizer_fingerprint(tokenizer, snapshot)
    weights.write_bytes(b"weights-v2")
    second = tokenizer_fingerprint(tokenizer, snapshot)
    assert first == second
