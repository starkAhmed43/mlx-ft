"""Reproducible run artifacts and privacy-safe run manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import PROMPT_CONTRACT_VERSION

ARTIFACT_VERSION = 2
COMMON_ARTIFACTS = ("resolved_config.yaml", "manifest.json", "status.json")
COMPLETED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "training": ("adapter", "train.log", "training_metrics.jsonl", "system_metrics.csv"),
    "evaluation": ("predictions.jsonl", "evaluation.json", "system_metrics.csv"),
    "benchmark": ("system_metrics.csv",),
    "robustness": ("predictions.jsonl", "evaluation.json", "system_metrics.csv"),
    "bfcl": ("predictions.jsonl", "evaluation.json", "system_metrics.csv"),
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_label(path: str | Path, root: str | Path) -> str:
    """Return a project-relative path, never an absolute user path."""
    value = Path(path).resolve()
    base = Path(root).resolve()
    try:
        return value.relative_to(base).as_posix()
    except ValueError:
        return f"external:{_sha256(str(value).encode())[:16]}"


def _git_identity(root: Path) -> dict[str, str | bool | None]:
    def git(*args: str) -> str | None:
        try:
            return (
                subprocess.run(
                    ["git", *args], cwd=root, check=True, capture_output=True, text=True
                ).stdout.strip()
                or None
            )
        except OSError, subprocess.CalledProcessError:
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _hardware() -> dict[str, str | int | float]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }


def _sanitize_value(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        secret_words = (
            "password",
            "secret",
            "credential",
            "api_key",
            "token",
            "username",
            "user_name",
            "home",
        )
        for key, item in value.items():
            label = str(key)
            if any(word in label.casefold() for word in secret_words):
                clean[label] = "[redacted]"
            else:
                clean[label] = _sanitize_value(item, root)
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, root) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value, root)
    return value


def sanitize_persisted_value(value: Any, root: str | Path) -> Any:
    """Remove private values before writing metadata to a run artifact."""
    return _sanitize_value(value, Path(root).resolve())


def _sanitize_string(value: str, root: Path) -> str:
    """Remove paths, credentials, UUIDs, and home fragments from free text."""
    text = value
    if os.path.isabs(text):
        return project_label(text, root)
    root_text = str(root.resolve())
    if root_text in text:
        text = text.replace(root_text, "<project>")
    text = re.sub(r"/(?:Users|home|private|var|tmp)/[^\s,'\"]+", "<path>", text)
    text = re.sub(r"(?i)(?:bearer\s+|hf_)[A-Za-z0-9._-]{8,}", "<credential>", text)
    text = re.sub(
        r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]+",
        "<credential>",
        text,
    )
    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        "<uuid>",
        text,
        flags=re.IGNORECASE,
    )
    return text


@dataclass(frozen=True)
class RunContract:
    root: Path
    run_dir: Path
    kind: Literal["training", "evaluation", "benchmark", "robustness", "bfcl"]
    manifest: dict[str, Any]

    def path(self, name: str) -> Path:
        return self.run_dir / name


def controlled_run_eligible(
    manifest: Mapping[str, Any],
    *,
    registered_model: bool,
    model_hash: str,
    rendering_hashes: Mapping[str, str],
    expected_examples: int | None = None,
    expected_updates: int | None = None,
    require_safe_probe: bool = False,
    comparable_rendering: bool | None = None,
) -> bool:
    """Return whether a run has enough pinned evidence for controlled reports."""
    if not registered_model or manifest.get("artifact_version") != ARTIFACT_VERSION:
        return False
    if manifest.get("prompt_contract_version") != PROMPT_CONTRACT_VERSION:
        return False
    if manifest.get("dataset_prompt_contract_version") != PROMPT_CONTRACT_VERSION:
        return False
    hashes = manifest.get("hashes", {})
    if not isinstance(hashes, Mapping) or hashes.get("model") != model_hash:
        return False
    if any(not value or hashes.get(key) != value for key, value in rendering_hashes.items()):
        return False
    if comparable_rendering is None:
        comparable_rendering = manifest.get("comparable_rendering") is True
    if not comparable_rendering:
        return False
    counts = manifest.get("counts", {})
    if not isinstance(counts, Mapping):
        return False
    if counts.get("examples") is None:
        return False
    if expected_examples is not None and counts.get("examples") != expected_examples:
        return False
    if expected_updates is not None and counts.get("updates") is None:
        return False
    if expected_updates is not None and counts.get("updates") != expected_updates:
        return False
    if require_safe_probe:
        safety = manifest.get("safety_probe", {})
        if not isinstance(safety, Mapping) or safety.get("status") != "safe":
            return False
        if safety.get("model_hash") != model_hash:
            return False
    return True


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def create_run_contract(
    root: str | Path,
    run_id: str,
    *,
    kind: Literal["training", "evaluation", "benchmark", "robustness", "bfcl"],
    config: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    model_hash: str | None = None,
    tokenizer_hash: str | None = None,
    dataset_hash: str | None = None,
    adapter_hash: str | None = None,
    lock_hash: str | None = None,
    seed: int | None = None,
    trainable_parameters: int | None = None,
    command: str | list[str] | None = None,
) -> RunContract:
    """Create the stable run tree and sanitized manifest."""
    base = Path(root).resolve()
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    safe_config = _sanitize_value(config, base)
    # Training backends receive this output directory. Its emptiness is only
    # acceptable while the run is still started.
    (run_dir / "adapter").mkdir()
    # Do not create empty output placeholders. A completed run must prove that
    # each artifact was actually produced.
    (run_dir / "status.json").write_text(json.dumps({"status": "started"}) + "\n")
    now = datetime.now(UTC).isoformat()
    hashes = {
        "model": model_hash or "",
        "tokenizer": tokenizer_hash or "",
        "dataset": dataset_hash or "",
        "adapter": adapter_hash or "",
        "config": _sha256(json.dumps(safe_config, sort_keys=True).encode()),
        "lock": lock_hash or "",
    }
    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "kind": kind,
        "run_id": run_id,
        "command": _sanitize_value(command or ["ftlab"], base),
        "worker_command": None,
        "started_at": now,
        "finished_at": None,
        "git": _git_identity(base),
        "hardware": _hardware(),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(aliased=True),
        },
        "hashes": hashes,
        "seed": seed,
        "trainable_parameters": trainable_parameters,
        "counts": {
            "examples": None,
            "updates": None,
            "prompt_tokens": None,
            "target_tokens": None,
        },
        "bytes": {"adapter": None, "checkpoint": None},
        "requirements": {"tokenizer_hash": False, "checkpoint_bytes": False},
        "controlled": False,
        "comparable_rendering": False,
        "status": "started",
        "safety_probe": {"status": "not_run"},
        "config": safe_config,
    }
    if manifest:
        payload["manifest"] = _sanitize_value(manifest, base)
    _write_json(run_dir / "manifest.json", payload)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(safe_config, sort_keys=True))
    return RunContract(base, run_dir, kind, payload)


def open_run_contract(
    root: str | Path,
    run_dir: str | Path,
    *,
    kind: Literal["training", "evaluation", "benchmark", "robustness", "bfcl"],
) -> RunContract:
    """Read an existing run without changing legacy artifacts.

    New jobs must call :func:`create_run_contract` before work starts.  This
    helper remains only for read-only inspection of old runs.
    """
    base = Path(root).resolve()
    path = Path(run_dir).resolve()
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be an object")
    return RunContract(base, path, kind, manifest)


def update_manifest(run: RunContract, **updates: Any) -> dict[str, Any]:
    payload = json.loads(run.path("manifest.json").read_text(encoding="utf-8"))
    for key, value in updates.items():
        value = _sanitize_value(value, run.root)
        if key == "hashes" and isinstance(value, dict):
            payload.setdefault("hashes", {}).update(value)
        elif key == "counts" and isinstance(value, dict):
            payload.setdefault("counts", {}).update(value)
        elif key in {"bytes", "requirements"} and isinstance(value, dict):
            payload.setdefault(key, {}).update(value)
        else:
            payload[key] = value
    _write_json(run.path("manifest.json"), payload)
    return payload


def write_status(run: RunContract, status: str, **extra: Any) -> None:
    if status not in {"started", "completed", "failed", "aborted", "infeasible"}:
        raise ValueError(f"unknown run status: {status}")
    payload = {
        "status": status,
        "finished_at": datetime.now(UTC).isoformat(),
        **_sanitize_value(extra, run.root),
    }
    _write_json(run.path("status.json"), payload)
    update_manifest(run, status=status, finished_at=payload["finished_at"])


def append_training_metric(run: RunContract, value: dict[str, Any]) -> None:
    with run.path("training_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_sanitize_value(value, run.root), sort_keys=True, allow_nan=False) + "\n"
        )


def append_system_metric(run: RunContract, value: dict[str, Any]) -> None:
    value = _sanitize_value(value, run.root)
    path = run.path("system_metrics.csv")
    fields = list(value)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(value)
        handle.flush()


def summarize_system_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return before, peak, and after values for runtime resource samples."""
    fields = (
        "rss_gib",
        "mlx_allocator_active_bytes",
        "mlx_allocator_peak_bytes",
        "swap_used_gib",
    )
    summary: dict[str, Any] = {"samples": len(samples)}
    for field in fields:
        values = [
            float(sample[field]) for sample in samples if isinstance(sample.get(field), int | float)
        ]
        if values:
            summary[field] = {"before": values[0], "peak": max(values), "after": values[-1]}
    pressures = [
        str(sample["pressure_state"]) for sample in samples if sample.get("pressure_state")
    ]
    if pressures:
        order = {"unknown": 0, "normal": 1, "warning": 2, "critical": 3}
        summary["pressure_state"] = {
            "before": pressures[0],
            "peak": max(pressures, key=lambda item: order.get(item, 0)),
            "after": pressures[-1],
        }
    return summary


def validate_run_artifacts(
    run_dir: str | Path,
    *,
    kind: Literal["training", "evaluation", "benchmark", "robustness", "bfcl"],
) -> list[str]:
    """Return missing artifacts, with type-specific completion requirements."""
    path = Path(run_dir)
    missing = [name for name in COMMON_ARTIFACTS if not (path / name).exists()]
    status_path = path / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = None
        if status == "completed":
            required = [*COMMON_ARTIFACTS, *COMPLETED_ARTIFACTS[kind]]
            missing.extend(
                name
                for name in required
                if not (path / name).exists()
                or (path / name).is_file()
                and (path / name).stat().st_size == 0
                or (path / name).is_dir()
                and not any((path / name).iterdir())
            )
            manifest_path = path / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except OSError, json.JSONDecodeError:
                manifest = {}
            requirements = manifest.get("requirements", {}) if isinstance(manifest, dict) else {}
            hashes = manifest.get("hashes", {}) if isinstance(manifest, dict) else {}
            sizes = manifest.get("bytes", {}) if isinstance(manifest, dict) else {}
            if isinstance(requirements, Mapping) and requirements.get("tokenizer_hash"):
                if not isinstance(hashes, Mapping) or not hashes.get("tokenizer"):
                    missing.append("manifest.hashes.tokenizer")
            if isinstance(requirements, Mapping) and requirements.get("checkpoint_bytes"):
                if not isinstance(sizes, Mapping) or not isinstance(sizes.get("checkpoint"), int):
                    missing.append("manifest.bytes.checkpoint")
    return sorted(set(missing))
