"""Typed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

MODEL_ID = "Qwen/Qwen3-0.6B-MLX-4bit"
MODEL_REVISION = "173234aa840d113125e9f2271100ddbaf16c9620"
DATASET_ID = "Salesforce/xlam-function-calling-60k"
DATASET_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
PROMPT_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    """Pinned model identity and architecture expectations."""

    model_id: str
    revision: str
    precision: Literal["4bit", "8bit", "bf16"]
    upstream_identity: str
    architecture_signature: dict[str, int]
    tokenizer_identity: dict[str, str]
    template_identity: dict[str, str | None]
    layer_count: int
    module_names: tuple[str, ...] = ("self_attn.q_proj", "self_attn.v_proj")

    @property
    def expected_module_names(self) -> tuple[str, ...]:
        return self.module_names

    @property
    def name(self) -> str:
        return self.model_id

    @property
    def checkpoint_revision(self) -> str:
        return self.revision

    @property
    def expected_tokenizer_identity(self) -> dict[str, str]:
        return self.tokenizer_identity

    @property
    def expected_template_identity(self) -> dict[str, str | None]:
        return self.template_identity


def _spec(
    model_id: str,
    revision: str,
    precision: Literal["4bit", "8bit", "bf16"],
    architecture: dict[str, int],
    upstream_identity: str,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        revision=revision,
        precision=precision,
        upstream_identity=upstream_identity,
        architecture_signature=architecture,
        tokenizer_identity={"revision": revision},
        template_identity={"sha256": None},
        layer_count=architecture["num_hidden_layers"],
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen/Qwen3-0.6B-MLX-4bit": _spec(
        "Qwen/Qwen3-0.6B-MLX-4bit",
        "173234aa840d113125e9f2271100ddbaf16c9620",
        "4bit",
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "Qwen/Qwen3-0.6B",
    ),
    "Qwen/Qwen3-1.7B-MLX-bf16": _spec(
        "Qwen/Qwen3-1.7B-MLX-bf16",
        "720c04346ea2b095c801ebbd545c109230964cd4",
        "bf16",
        {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "Qwen/Qwen3-1.7B",
    ),
    "Qwen/Qwen3-1.7B-MLX-8bit": _spec(
        "Qwen/Qwen3-1.7B-MLX-8bit",
        "95400cbada1bef81d2c4fb6f9d2b10aa27f1f1a0",
        "8bit",
        {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "Qwen/Qwen3-1.7B",
    ),
    "Qwen/Qwen3-1.7B-MLX-4bit": _spec(
        "Qwen/Qwen3-1.7B-MLX-4bit",
        "21457c6f51ed54a7c16e988c0844db973815c137",
        "4bit",
        {
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "Qwen/Qwen3-1.7B",
    ),
    "Qwen/Qwen3-4B-MLX-4bit": _spec(
        "Qwen/Qwen3-4B-MLX-4bit",
        "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495",
        "4bit",
        {
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        "Qwen/Qwen3-4B",
    ),
}


def get_model_spec(model: str, revision: str | None = None) -> ModelSpec:
    """Return a pinned model spec and reject unknown checkpoint identities."""
    spec = MODEL_REGISTRY.get(model)
    if spec is None:
        raise ValueError(f"model is not in the pinned registry: {model}")
    if revision is not None and revision != spec.revision:
        raise ValueError(
            f"revision mismatch for {model}: expected {spec.revision}, found {revision}"
        )
    return spec


def model_registry() -> tuple[ModelSpec, ...]:
    return tuple(MODEL_REGISTRY.values())


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = DATASET_ID
    revision: str = DATASET_REVISION
    split: str = "train"
    version: Literal[
        "smoke",
        "day1",
        "core",
        "full",
        "schema_ood",
        "nested_1k",
        "nested_5k",
        "nested_10k",
        "nested_20k",
        "nested-1k",
        "nested-5k",
        "nested-10k",
        "nested-20k",
    ] = "smoke"
    seed: int = 42


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = MODEL_ID
    revision: str = MODEL_REVISION
    tokenizer_revision: str | None = None
    precision: Literal["4bit", "8bit", "bf16"] | None = None


class RenderingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_thinking: bool = False
    max_seq_length: int = Field(default=512, gt=0)
    mask_prompt: bool = True


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keys: list[str] = ["self_attn.q_proj", "self_attn.v_proj"]
    rank: int = Field(default=8, gt=0)
    scale: float = Field(default=20.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fine_tune_type: Literal["lora"] = "lora"
    train: bool = True
    mask_prompt: bool = True
    seed: int = 42
    num_layers: int = Field(default=8, gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accumulation_steps: int = Field(default=8, gt=0)
    iters: int = Field(default=128, gt=0)
    max_seq_length: int = Field(default=512, gt=0)
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=5e-5, gt=0)
    steps_per_report: int = Field(default=8, gt=0)
    steps_per_eval: int = Field(default=128, gt=0)
    val_batches: int = -1
    save_every: int = Field(default=129, gt=0)
    grad_checkpoint: bool = False
    lora_parameters: LoraConfig = Field(default_factory=LoraConfig)

    @field_validator("val_batches")
    @classmethod
    def validate_val_batches(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("val_batches must be -1 or a positive integer")
        return value


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: str = "test"
    max_new_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    online: bool = False
    entity: str | None = None
    project: str = "mlx-ft"


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Path(".")
    min_free_gib: float = Field(default=30.0, ge=0)


class RunProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = "pipeline validation"
    seed: int = 42
    source: str = ""


class ExperimentConfig(BaseModel):
    """Complete configuration for one reproducible run."""

    model_config = ConfigDict(extra="forbid")

    model: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model_config_data: ModelConfig = Field(default_factory=ModelConfig, alias="model_info")
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    provenance: RunProvenance = Field(default_factory=RunProvenance)
    prompt_contract_version: int = PROMPT_CONTRACT_VERSION

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value:
            raise ValueError("model cannot be empty")
        return value

    def resolved(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""
        return self.model_dump(mode="json", by_alias=True)

    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.model, self.model_revision)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must contain a mapping")

    # The smoke plan uses MLX-LM's flat training keys. Accept that format and
    # map it into the typed model while retaining strict validation afterwards.
    training_keys = {
        "fine_tune_type",
        "train",
        "mask_prompt",
        "seed",
        "num_layers",
        "batch_size",
        "grad_accumulation_steps",
        "iters",
        "max_seq_length",
        "optimizer",
        "learning_rate",
        "steps_per_report",
        "steps_per_eval",
        "val_batches",
        "save_every",
        "grad_checkpoint",
        "lora_parameters",
    }
    if any(key in raw for key in training_keys):
        raw = dict(raw)
        raw["training"] = {key: raw.pop(key) for key in list(raw) if key in training_keys}
        raw.setdefault("rendering", {})
        raw["rendering"].setdefault("max_seq_length", raw["training"].get("max_seq_length", 512))
        raw["rendering"].setdefault("mask_prompt", raw["training"].get("mask_prompt", True))
        raw.pop("mask_prompt", None)
    if "model" in raw and isinstance(raw["model"], str):
        raw.setdefault("model_info", {"name": raw["model"]})
    return ExperimentConfig.model_validate(raw)
