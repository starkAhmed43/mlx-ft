from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from ftlab.config import TrainingConfig
from ftlab.exceptions import FtlabError
from ftlab.modeling import (
    PredictionMeasurement,
    assert_adapter_layout,
    generate_predictions_measured,
    run_training,
    tokenize_prompt_completion,
)


def _model(extra: bool = False):
    layers = []
    for index in range(28):
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(q_proj=SimpleNamespace(), v_proj=SimpleNamespace())
        )
        if index >= 20:
            layer.self_attn.q_proj.lora_a = object()
            layer.self_attn.v_proj.lora_a = object()
        if extra and index == 0:
            layer.self_attn.q_proj.lora_a = object()
        layers.append(layer)

    class Model:
        def __init__(self):
            self.layers = layers

        def named_modules(self):
            for index, layer in enumerate(self.layers):
                yield f"layers.{index}.self_attn.q_proj", layer.self_attn.q_proj
                yield f"layers.{index}.self_attn.v_proj", layer.self_attn.v_proj

    return Model()


def test_adapter_layout_is_exact() -> None:
    manifest = assert_adapter_layout(_model())
    assert manifest["selected_layers"] == list(range(20, 28))
    assert len(manifest["modules"]) == 16


def test_adapter_layout_rejects_early_adapter() -> None:
    with pytest.raises(FtlabError):
        assert_adapter_layout(_model(extra=True))


def test_expected_trainable_count_is_checked() -> None:
    model = _model()
    model.trainable_parameters = {"lora": SimpleNamespace(size=327680)}
    manifest = assert_adapter_layout(model, expected_trainable=327680)
    assert manifest["trainable_parameters"] == 327680
    with pytest.raises(FtlabError):
        assert_adapter_layout(model, expected_trainable=1)


def test_token_preprocessing_has_exact_offset_and_limit() -> None:
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    tokens, offset = tokenize_prompt_completion(
        "prompt", "completion", Tokenizer(), max_seq_length=20
    )
    assert tokens == list(b"promptcompletion")
    assert offset == len(b"prompt")
    with pytest.raises(FtlabError, match="exceeding max_seq_length"):
        tokenize_prompt_completion("prompt", "completion", Tokenizer(), max_seq_length=5)


def test_measured_generation_separates_prefill_decode_and_allocator(monkeypatch) -> None:
    class Mx:
        active = [10, 13]
        peak = [11, 17]

        @classmethod
        def get_active_memory(cls):
            return cls.active.pop(0)

        @classmethod
        def get_peak_memory(cls):
            return cls.peak.pop(0)

    class Tokenizer:
        @staticmethod
        def encode(value, add_special_tokens=False):
            assert add_special_tokens is False
            return list(value)

    class Response:
        def __init__(self, text, final=False):
            self.text = text
            self.prompt_tokens = 4
            self.generation_tokens = 2
            self.prompt_tps = 40.0
            self.generation_tps = 20.0
            self.finish_reason = "stop" if final else None

    class MlxLm:
        @staticmethod
        def stream_generate(model, tokenizer, *, prompt, max_tokens):
            assert prompt == "test"
            assert max_tokens == 8
            yield Response("hi")
            yield Response("!", final=True)

    monkeypatch.setattr("ftlab.modeling._mlx_modules", lambda: (Mx, MlxLm, object()))
    measured = generate_predictions_measured(object(), Tokenizer(), ["test"], max_tokens=8)
    assert len(measured) == 1
    item = measured[0]
    assert item.prediction == "hi!"
    assert item.prompt_tokens == 4
    assert item.output_tokens == 3
    assert item.prompt_processing_tokens_per_second == 40.0
    assert item.generation_tokens_per_second == 20.0
    assert (item.allocator_before_bytes, item.allocator_peak_bytes, item.allocator_after_bytes) == (
        10,
        17,
        13,
    )


def test_evaluation_predictions_store_recomputable_gold_without_prompt(
    tmp_path, monkeypatch
) -> None:
    from ftlab.worker import run_evaluation_request

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "test": [
                        {
                            "id": "stable-id",
                            "query": "Private prompt text must not be persisted.",
                            "tools": [
                                {
                                    "name": "weather",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"],
                                    },
                                }
                            ],
                            "answer": {"name": "weather", "arguments": {"city": "Paris"}},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    class Tokenizer:
        @staticmethod
        def encode(value, add_special_tokens=False):
            return list(value)

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            prompt = "rendered private prompt"
            if kwargs["add_generation_prompt"]:
                return prompt
            return prompt + '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'

    prediction = '<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>'
    monkeypatch.setattr(
        "ftlab.modeling.load_model", lambda *args, **kwargs: (object(), Tokenizer())
    )
    monkeypatch.setattr("ftlab.modeling.generate_predictions", lambda *args, **kwargs: [prediction])
    monkeypatch.setattr(
        "ftlab.modeling.generate_predictions_measured",
        lambda *args, **kwargs: [PredictionMeasurement(prediction, 0.1, 5, 8, 50.0, 80.0, 1, 3, 2)],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    run_evaluation_request(
        {
            "model": "fixture",
            "model_revision": "fixture",
            "manifest": str(manifest),
            "predictions_path": str(predictions_path),
            "evaluation_path": str(tmp_path / "evaluation.json"),
        }
    )
    row = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert row["source_id"] == "stable-id"
    assert row["record"]["answer"] == {"name": "weather", "arguments": {"city": "Paris"}}
    assert row["record"]["tools"][0]["name"] == "weather"
    assert row["diagnostics"]["truncated"] is False
    assert set(row["hashes"]) == {"prompt", "record"}
    serialized = json.dumps(row)
    assert "Private prompt text" not in serialized
    assert "rendered private prompt" not in serialized
    assert str(tmp_path) not in serialized


def test_training_invocation_preserves_smoke_contract(tmp_path, monkeypatch) -> None:
    class FakeModel:
        def __init__(self):
            self.args = SimpleNamespace(
                hidden_size=1024,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
            )
            self.layers = [
                SimpleNamespace(
                    self_attn=SimpleNamespace(q_proj=SimpleNamespace(), v_proj=SimpleNamespace())
                )
                for _ in range(28)
            ]

        def freeze(self):
            return None

        def named_modules(self):
            for index, layer in enumerate(self.layers):
                yield f"layers.{index}.self_attn.q_proj", layer.self_attn.q_proj
                yield f"layers.{index}.self_attn.v_proj", layer.self_attn.v_proj

        def trainable_parameters(self):
            return {"lora": [SimpleNamespace(size=327680)]}

    class FakeMx:
        class random:
            @staticmethod
            def seed(value):
                assert value == 42

    class FakeOptimizers:
        @staticmethod
        def AdamW(learning_rate):
            assert learning_rate == 5.0e-5
            return object()

    class FakeTokenizer:
        eos_token_id = 99

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text.encode())

    captured = {}

    def fake_lora(model, num_layers, parameters):
        assert num_layers == 8
        assert parameters == {
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        }
        for layer in model.layers[-8:]:
            layer.self_attn.q_proj.lora_a = object()
            layer.self_attn.v_proj.lora_a = object()

    class FakeCallback:
        def on_train_loss_report(self, info):
            pass

        def on_val_loss_report(self, info):
            pass

    class FakeCacheDataset:
        def __init__(self, data):
            self._data = data
            self._proc_data = [None] * len(data)

        def __getitem__(self, index):
            if self._proc_data[index] is None:
                self._proc_data[index] = self._data.process(self._data[index])
            return self._proc_data[index]

        def __len__(self):
            return len(self._data)

        def itemlen(self, index):
            return len(self._data[index])

    def fake_train(**kwargs):
        captured.update(kwargs)
        tokens, offset = kwargs["train_dataset"][0]
        assert tokens == list(b"prefixsuffix")
        assert tokens[-1] != 99
        assert offset == len(b"prefix")
        assert kwargs["train_dataset"].itemlen(0) == len(tokens)
        if "nonfinite-adapter" in kwargs["args"].adapter_file:
            kwargs["training_callback"].on_train_loss_report({"train_loss": float("nan")})
            return
        kwargs["training_callback"].on_val_loss_report({"val_loss": 1.0})
        kwargs["training_callback"].on_train_loss_report({"iteration": 8, "train_loss": 2.0})
        Path(kwargs["args"].adapter_file).write_bytes(b"adapter")

    from pathlib import Path

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.__path__ = []
    tuner = types.ModuleType("mlx_lm.tuner")
    tuner.__path__ = []
    datasets = types.ModuleType("mlx_lm.tuner.datasets")
    datasets.CacheDataset = FakeCacheDataset
    trainer = types.ModuleType("mlx_lm.tuner.trainer")
    trainer.TrainingArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    trainer.train = fake_train
    utils = types.ModuleType("mlx_lm.tuner.utils")
    utils.linear_to_lora_layers = fake_lora
    callbacks = types.ModuleType("mlx_lm.tuner.callbacks")
    callbacks.TrainingCallback = FakeCallback
    for name, module in {
        "mlx_lm": mlx_lm,
        "mlx_lm.tuner": tuner,
        "mlx_lm.tuner.datasets": datasets,
        "mlx_lm.tuner.trainer": trainer,
        "mlx_lm.tuner.utils": utils,
        "mlx_lm.tuner.callbacks": callbacks,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("ftlab.modeling._mlx_modules", lambda: (FakeMx(), mlx_lm, FakeOptimizers()))
    monkeypatch.setattr(
        "ftlab.modeling.load_model", lambda *args, **kwargs: (FakeModel(), FakeTokenizer())
    )
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    row = {"prompt": "prefix", "completion": "suffix"}
    for path in (train_path, valid_path):
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = TrainingConfig()
    result = run_training(
        config,
        model_path="fixture",
        train_path=str(train_path),
        valid_path=str(valid_path),
        adapter_path=str(tmp_path / "adapter"),
    )
    args = captured["args"]
    assert args.iters == 128
    assert args.grad_accumulation_steps == 8
    assert args.max_seq_length == 512
    assert args.steps_per_report == 8
    assert args.steps_per_eval == 128
    assert args.steps_per_save == 129
    assert result["microbatches"] == 128
    assert result["optimizer_updates"] == 16
    assert result["adapter_bytes"] == len(b"adapter")
    assert result["adapter_hash"]
    assert result["tokenizer_hash"]
    with pytest.raises(FtlabError, match="training metric train_loss is not finite"):
        run_training(
            config,
            model_path="fixture",
            train_path=str(train_path),
            valid_path=str(valid_path),
            adapter_path=str(tmp_path / "nonfinite-adapter"),
        )
    with pytest.raises(FtlabError, match="exceeding max_seq_length"):
        run_training(
            TrainingConfig(max_seq_length=5),
            model_path="fixture",
            train_path=str(train_path),
            valid_path=str(valid_path),
            adapter_path=str(tmp_path / "short-adapter"),
        )
    with pytest.raises(FtlabError, match="overwrite"):
        run_training(
            config,
            model_path="fixture",
            train_path=str(train_path),
            valid_path=str(valid_path),
            adapter_path=str(tmp_path / "adapter"),
        )
