"""Lazy MLX-LM integration and LoRA layout assertions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from .config import MODEL_ID, MODEL_REVISION, ModelSpec, TrainingConfig, get_model_spec
from .exceptions import ExternalDependencyError, FtlabError
from .huggingface import resolve_snapshot

EXPECTED_TRAINABLE_PARAMETERS = 327680


@dataclass(frozen=True)
class PredictionMeasurement:
    """One measured inference output, excluding caller-side prompt rendering."""

    prediction: str
    latency_seconds: float
    prompt_tokens: int
    output_tokens: int
    prompt_processing_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    allocator_before_bytes: int | None
    allocator_peak_bytes: int | None
    allocator_after_bytes: int | None


def _mlx_modules() -> tuple[Any, Any, Any]:
    try:
        import mlx.core as mx
        import mlx.optimizers as optimizers
        import mlx_lm
    except ImportError as exc:
        raise ExternalDependencyError(
            "MLX and MLX-LM are required; install the apple extra on Apple Silicon"
        ) from exc
    return mx, mlx_lm, optimizers


def _allocator_memory(mx: Any) -> tuple[int | None, int | None]:
    def read(name: str) -> int | None:
        getter = getattr(mx, name, None)
        if not callable(getter):
            return None
        try:
            return int(getter())
        except TypeError, ValueError:
            return None

    return read("get_active_memory"), read("get_peak_memory")


def load_model(
    model: str = MODEL_ID,
    *,
    revision: str = MODEL_REVISION,
    adapter_path: str | None = None,
    cache_root: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load a pinned model and tokenizer only for a real integration command."""
    model_source = model
    if cache_root is not None:
        cache = Path(cache_root).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.update(
            {
                "HF_HOME": str(cache),
                "HF_HUB_CACHE": str(cache / "hub"),
                "HF_DATASETS_CACHE": str(cache / "datasets"),
                "TRANSFORMERS_CACHE": str(cache / "transformers"),
            }
        )
        model_source = str(resolve_snapshot(model, revision, cache))
    elif not Path(model).expanduser().exists():
        raise ExternalDependencyError("a project cache root is required for a remote pinned model")
    _, mlx_lm, _ = _mlx_modules()
    try:
        kwargs: dict[str, Any] = {}
        if adapter_path is not None:
            kwargs["adapter_path"] = adapter_path
        loaded = mlx_lm.load(model_source, **kwargs)
    except Exception as exc:
        raise ExternalDependencyError(f"failed to load model {model}@{revision}") from exc
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise FtlabError("MLX-LM returned an unexpected model and tokenizer value")
    return loaded[0], loaded[1]


def model_layers(model: Any) -> list[Any]:
    for value in (getattr(model, "model", None), model):
        if value is None:
            continue
        for name in ("layers", "h", "transformer_layers"):
            layers = getattr(value, name, None)
            if layers is not None:
                try:
                    return list(layers)
                except TypeError:
                    pass
        nested = getattr(value, "model", None)
        if nested is not None:
            layers = getattr(nested, "layers", None)
            if layers is not None:
                return list(layers)
    raise FtlabError("could not find transformer layers in loaded model")


def assert_pinned_architecture(model: Any, spec: ModelSpec | None = None) -> dict[str, Any]:
    """Check a loaded model against its registered architecture signature."""
    if spec is None:
        spec = get_model_spec(MODEL_ID, MODEL_REVISION)
    layers = model_layers(model)
    expected_layers = spec.layer_count
    if len(layers) != expected_layers:
        raise FtlabError(f"expected {expected_layers} model layers, found {len(layers)}")
    args = getattr(model, "args", None)
    if args is None:
        raise FtlabError("pinned model does not expose architecture arguments")
    expected = {
        key: value
        for key, value in spec.architecture_signature.items()
        if key != "num_hidden_layers"
    }
    mismatches = {
        name: (getattr(args, name, None), value)
        for name, value in expected.items()
        if getattr(args, name, None) != value
    }
    if mismatches:
        raise FtlabError(f"pinned architecture mismatch: {mismatches}")
    return {
        "model": spec.model_id,
        "revision": spec.revision,
        "precision": spec.precision,
        "architecture_signature": dict(spec.architecture_signature),
    }


def _has_adapter(module: Any) -> bool:
    return any(hasattr(module, name) for name in ("lora_a", "lora_b", "adapter", "lora"))


def assert_adapter_layout(
    model: Any,
    *,
    total_layers: int | None = 28,
    num_layers: int = 8,
    keys: Iterable[str] = ("self_attn.q_proj", "self_attn.v_proj"),
    expected_trainable: int | None = None,
    spec: ModelSpec | None = None,
) -> dict[str, Any]:
    """Assert adapters cover only the final layers and return a manifest."""
    layers = model_layers(model)
    if spec is not None:
        total_layers = spec.layer_count
    if total_layers is None:
        total_layers = len(layers)
    if len(layers) != total_layers:
        raise FtlabError(f"expected {total_layers} model layers, found {len(layers)}")
    selected = list(range(total_layers - num_layers, total_layers))
    key_list = tuple(keys)
    found: list[str] = []
    for index, layer in enumerate(layers):
        for key in key_list:
            module: Any = layer
            for part in key.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            adapted = module is not None and _has_adapter(module)
            if index in selected and not adapted:
                raise FtlabError(f"missing LoRA adapter at layer {index} {key}")
            if index not in selected and adapted:
                raise FtlabError(f"unexpected LoRA adapter at layer {index} {key}")
            if adapted:
                found.append(f"layers.{index}.{key}")
    if len(found) != len(selected) * len(key_list):
        raise FtlabError(
            f"expected {len(selected) * len(key_list)} LoRA linear modules, found {len(found)}"
        )
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        adapted_names = [name for name, module in named_modules() if _has_adapter(module)]
        if len(adapted_names) != len(found):
            raise FtlabError(
                f"expected exactly {len(found)} LoRA modules, found {len(adapted_names)}"
            )
    trainable = trainable_parameter_count(model)
    if expected_trainable is not None and trainable != expected_trainable:
        raise FtlabError(
            f"trainable parameter count {trainable} does not match manifest {expected_trainable}"
        )
    return {
        "total_layers": total_layers,
        "selected_layers": selected,
        "modules": found,
        "trainable_parameters": trainable,
        "module_names": list(key_list),
    }


def lora_parameters_for_rank(
    config: TrainingConfig, rank: int, *, sweep: bool = False
) -> dict[str, Any]:
    """Return direct MLX-LM parameters; rank sweeps always use scale 20."""
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    parameters = config.lora_parameters.model_dump()
    parameters["rank"] = rank
    if sweep:
        parameters["scale"] = 20.0
    return parameters


def loaded_snapshot_identity(
    model: Any, tokenizer: Any, spec: ModelSpec, snapshot_root: str | Path | None = None
) -> dict[str, Any]:
    """Build comparable loaded-architecture, tokenizer, and template identity fields."""
    args = getattr(model, "args", None)
    architecture = {
        key: (len(model_layers(model)) if key == "num_hidden_layers" else getattr(args, key, None))
        for key in spec.architecture_signature
    }
    template = str(getattr(tokenizer, "chat_template", ""))
    config_path = Path(snapshot_root) / "config.json" if snapshot_root else None
    config_hash = None
    if config_path is not None and config_path.is_file():
        import hashlib

        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    import hashlib

    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "precision": spec.precision,
        "architecture_signature": architecture,
        "tokenizer_revision": getattr(tokenizer, "_ftlab_revision", spec.revision),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "config_sha256": config_hash,
    }


def assert_loaded_snapshot_identity(
    model: Any, tokenizer: Any, spec: ModelSpec, snapshot_root: str | Path | None = None
) -> dict[str, Any]:
    identity = loaded_snapshot_identity(model, tokenizer, spec, snapshot_root)
    expected = spec.architecture_signature
    if identity["architecture_signature"] != expected:
        raise FtlabError(
            f"loaded architecture does not match {spec.model_id}: "
            f"{identity['architecture_signature']} != {expected}"
        )
    return identity


def trainable_parameter_count(model: Any) -> int:
    total = 0
    parameters = getattr(model, "trainable_parameters", None)
    if callable(parameters):
        parameters = parameters()
    if parameters is not None:

        def count(value: Any) -> int:
            if isinstance(value, dict):
                return sum(count(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count(item) for item in value)
            size = getattr(value, "size", None)
            if size is not None:
                return int(size)
            shape = getattr(value, "shape", None)
            if shape is not None:
                result = 1
                for dimension in shape:
                    result *= int(dimension)
                return result
            return 0

        total = count(parameters)
    return total


def tokenize_prompt_completion(
    prompt: str,
    completion: str,
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> tuple[list[int], int]:
    """Tokenize the exact training record and return tokens plus prompt offset."""
    prompt_tokens = list(tokenizer.encode(prompt, add_special_tokens=False))
    full_tokens = list(tokenizer.encode(prompt + completion, add_special_tokens=False))
    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise FtlabError("tokenizer prompt tokens do not match the full record prefix")
    if len(full_tokens) > max_seq_length:
        raise FtlabError(
            f"training example has {len(full_tokens)} tokens, exceeding "
            f"max_seq_length={max_seq_length}"
        )
    return full_tokens, len(prompt_tokens)


def run_training(
    config: TrainingConfig,
    *,
    model_path: str,
    model_revision: str = MODEL_REVISION,
    cache_root: str | Path | None = None,
    train_path: str,
    valid_path: str,
    adapter_path: str,
) -> dict[str, Any]:
    """Call MLX-LM training lazily and retain the smoke exposure contract."""
    mx, _, optimizers = _mlx_modules()
    try:
        from mlx_lm.tuner.datasets import CacheDataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise ExternalDependencyError("MLX-LM training support is not installed") from exc
    adapter = Path(adapter_path)
    if adapter.exists() and any(adapter.iterdir()):
        raise FtlabError(f"refusing to overwrite existing adapter directory: {adapter}")
    adapter.mkdir(parents=True, exist_ok=True)

    import numpy as np

    mx.random.seed(config.seed)
    np.random.seed(config.seed)
    model, tokenizer = load_model(model_path, revision=model_revision, cache_root=cache_root)
    try:
        from .rendering import tokenizer_fingerprint

        tokenizer_identity = tokenizer_fingerprint(
            tokenizer, getattr(tokenizer, "_ftlab_snapshot", None)
        )
    except Exception:
        tokenizer_identity = {}
    spec = None
    try:
        spec = get_model_spec(model_path, model_revision)
    except ValueError:
        # Fixture tests and local adapters can expose architecture without a registry ID.
        spec = None
    assert_pinned_architecture(model, spec)
    model.freeze()
    linear_to_lora_layers(
        model, config.num_layers, lora_parameters_for_rank(config, config.lora_parameters.rank)
    )
    manifest = assert_adapter_layout(
        model,
        total_layers=spec.layer_count if spec is not None else len(model_layers(model)),
        num_layers=config.num_layers,
        keys=config.lora_parameters.keys,
        expected_trainable=None,
        spec=spec,
    )

    class PromptCompletionDataset:
        def __init__(self, path: str) -> None:
            self.rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        def process(self, row: dict[str, str]) -> tuple[list[int], int]:
            return tokenize_prompt_completion(
                row["prompt"],
                row["completion"],
                tokenizer,
                max_seq_length=config.max_seq_length,
            )

        def __getitem__(self, index: int) -> dict[str, str]:
            return self.rows[index]

        def __len__(self) -> int:
            return len(self.rows)

        def itemlen(self, index: int) -> int:
            return len(self.process(self.rows[index])[0])

    class TokenLengthCacheDataset(CacheDataset):
        """Use rendered token lengths for MLX-LM batch ordering."""

        def itemlen(self, index: int) -> int:
            return self._data.itemlen(index)

    train_data = PromptCompletionDataset(train_path)
    valid_data = PromptCompletionDataset(valid_path)
    prompt_tokens = 0
    target_tokens = 0
    for row in train_data.rows:
        full, prompt_length = train_data.process(row)
        prompt_tokens += prompt_length
        target_tokens += len(full) - prompt_length
    training_args = TrainingArgs(
        batch_size=config.batch_size,
        iters=config.iters,
        val_batches=config.val_batches,
        steps_per_report=config.steps_per_report,
        steps_per_eval=config.steps_per_eval,
        steps_per_save=config.save_every,
        max_seq_length=config.max_seq_length,
        grad_checkpoint=config.grad_checkpoint,
        grad_accumulation_steps=config.grad_accumulation_steps,
        adapter_file=str(adapter / "adapters.safetensors"),
    )
    from mlx_lm.tuner.callbacks import TrainingCallback

    class MetricsCallback(TrainingCallback):
        def __init__(self) -> None:
            self.values: list[dict[str, float]] = []

        def _record(self, value: dict[str, Any]) -> None:
            clean: dict[str, float] = {}
            for key, item in value.items():
                if isinstance(item, Real):
                    numeric = float(item)
                    if not math.isfinite(numeric):
                        raise FtlabError(f"training metric {key} is not finite")
                    clean[key] = numeric
            if clean:
                self.values.append(clean)

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            self._record(train_info)

        def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
            self._record(val_info)

    callback = MetricsCallback()
    try:
        optimizer = optimizers.AdamW(learning_rate=config.learning_rate)
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=TokenLengthCacheDataset(train_data),
            val_dataset=TokenLengthCacheDataset(valid_data),
            args=training_args,
            training_callback=callback,
        )
    except ExternalDependencyError:
        raise
    except Exception as exc:
        raise FtlabError(f"MLX-LM training failed: {exc}") from exc
    adapter_config = {
        "fine_tune_type": config.fine_tune_type,
        "num_layers": config.num_layers,
        "lora_parameters": config.lora_parameters.model_dump(),
        "trainable_parameters": manifest["trainable_parameters"],
        "microbatches": config.iters,
        "optimizer_updates": config.iters // config.grad_accumulation_steps,
        "mask_prompt": config.mask_prompt,
        "optimizer": config.optimizer,
        "steps_per_report": config.steps_per_report,
        "steps_per_eval": config.steps_per_eval,
        "val_batches": config.val_batches,
        "save_every": config.save_every,
        "grad_checkpoint": config.grad_checkpoint,
        "lora_keys": list(config.lora_parameters.keys),
        "lora_rank": config.lora_parameters.rank,
        "lora_scale": config.lora_parameters.scale,
        "lora_dropout": config.lora_parameters.dropout,
    }
    (adapter / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
    )
    if (
        not (adapter / "adapters.safetensors").exists()
        or not (adapter / "adapters.safetensors").stat().st_size
    ):
        raise FtlabError("MLX-LM did not produce a non-empty final adapter")
    adapter_file = adapter / "adapters.safetensors"
    adapter_bytes = adapter_file.stat().st_size
    tokenizer_hash = hashlib.sha256(
        json.dumps(tokenizer_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "microbatches": config.iters,
        "optimizer_updates": config.iters // config.grad_accumulation_steps,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_hash": tokenizer_hash,
        "adapter": str(adapter),
        "adapter_bytes": adapter_bytes,
        "checkpoint_bytes": None,
        "adapter_hash": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
        "manifest": manifest,
        "training_metrics": callback.values,
    }


def generate_predictions(
    model: Any, tokenizer: Any, prompts: Iterable[str], *, max_tokens: int = 128
) -> list[str]:
    """Generate outputs with MLX-LM only after the model is loaded."""
    _, mlx_lm, _ = _mlx_modules()
    generate = getattr(mlx_lm, "generate", None)
    if not callable(generate):
        raise ExternalDependencyError("MLX-LM generation API is unavailable")
    outputs: list[str] = []
    for prompt in prompts:
        try:
            result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as exc:
            raise FtlabError(f"MLX-LM generation failed: {exc}") from exc
        outputs.append(result if isinstance(result, str) else str(result))
    return outputs


def generate_predictions_measured(
    model: Any, tokenizer: Any, prompts: Iterable[str], *, max_tokens: int = 128
) -> list[PredictionMeasurement]:
    """Generate with MLX-LM stream telemetry for separate prefill and decode rates."""
    mx, mlx_lm, _ = _mlx_modules()
    stream_generate = getattr(mlx_lm, "stream_generate", None)
    if not callable(stream_generate):
        raise ExternalDependencyError("MLX-LM streaming generation API is unavailable")
    outputs: list[PredictionMeasurement] = []
    for prompt in prompts:
        before_active, before_peak = _allocator_memory(mx)
        started = time.perf_counter()
        segments: list[str] = []
        last: Any = None
        try:
            for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                segments.append(str(getattr(response, "text", "")))
                last = response
        except Exception as exc:
            raise FtlabError(f"MLX-LM generation failed: {exc}") from exc
        if last is None:
            raise FtlabError("MLX-LM streaming generation returned no response")
        after_active, after_peak = _allocator_memory(mx)
        output = "".join(segments)
        prompt_tokens = int(getattr(last, "prompt_tokens", 0))
        generated = int(getattr(last, "generation_tokens", 0))
        try:
            output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
        except AttributeError, TypeError, ValueError:
            output_tokens = generated
        prompt_tps = getattr(last, "prompt_tps", None)
        generation_tps = getattr(last, "generation_tps", None)
        outputs.append(
            PredictionMeasurement(
                prediction=output,
                latency_seconds=time.perf_counter() - started,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                prompt_processing_tokens_per_second=(
                    float(prompt_tps) if isinstance(prompt_tps, Real) else None
                ),
                generation_tokens_per_second=(
                    float(generation_tps) if isinstance(generation_tps, Real) else None
                ),
                allocator_before_bytes=before_active,
                allocator_peak_bytes=after_peak if after_peak is not None else before_peak,
                allocator_after_bytes=after_active,
            )
        )
    return outputs
