"""Typed deterministic sweep expansion and resumable selection."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "sweep"
    base_config: str | None = None
    axes: dict[str, list[Any]] = Field(default_factory=dict)
    runs: list[dict[str, Any]] = Field(default_factory=list)
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    seed: int = 42


@dataclass(frozen=True)
class SweepRun:
    index: int
    values: dict[str, Any]
    fingerprint: str


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[parts[-1]] = value


def _fingerprint(values: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_sweep(path: str | Path) -> SweepConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("sweep YAML must contain a mapping")
    return SweepConfig.model_validate(value)


def expand_matrix(config: SweepConfig | str | Path) -> list[SweepRun]:
    """Expand axes in sorted path order, independent of YAML insertion order."""
    plan = load_sweep(config) if isinstance(config, (str, Path)) else config
    if plan.runs:
        explicit_runs: list[SweepRun] = []
        for index, values in enumerate(plan.runs):
            row = {"name": plan.name, **values}
            explicit_runs.append(SweepRun(index, row, _fingerprint(row)))
        return explicit_runs
    axes = [(path, plan.axes[path]) for path in sorted(plan.axes)]
    if not axes:
        values = {"name": plan.name, **plan.profiles.get("default", {})}
        return [SweepRun(0, values, _fingerprint(values))]
    runs: list[SweepRun] = []
    for index, combination in enumerate(
        itertools.product(*(axis_values for _, axis_values in axes))
    ):
        expanded: dict[str, Any] = {"name": plan.name}
        for (path, _), value in zip(axes, combination):
            _set_path(expanded, path, value)
        expanded["seed"] = plan.seed
        runs.append(SweepRun(index, expanded, _fingerprint(expanded)))
    return runs


def expand_matrix_values(config: SweepConfig | str | Path) -> list[dict[str, Any]]:
    return [run.values for run in expand_matrix(config)]


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    """Rank validation runs with the locked study tie-break order."""
    measured = result.get("selection")
    if isinstance(measured, dict):
        result = {**result, **measured}
    return (
        float(result.get("task_success", result.get("call_success", 0.0))),
        float(result.get("argument_value_f1", 0.0)),
        float(result.get("schema_validity", 0.0)),
        -float(result.get("rss_gib", result.get("peak_rss_gib", float("inf")))),
        float(
            result.get(
                "generation_throughput_tokens_per_second",
                result.get("throughput_tokens_per_second", 0.0),
            )
        ),
        str(result.get("fingerprint", result.get("run_id", ""))),
    )


def select_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    complete = [
        item
        for item in results
        if item.get("status", "completed") == "completed"
        and isinstance(item.get("selection", item), dict)
    ]
    return max(complete, key=selection_key) if complete else None


def select_learning_rates_by_precision(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select one completed learning-rate pilot for each precision."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        selection = row.get("selection", row)
        if not isinstance(selection, dict) or selection.get("study_role") != "lr_pilot":
            continue
        precision = str(selection.get("precision", ""))
        if precision:
            grouped.setdefault(precision, []).append(row)
    return {
        precision: selected
        for precision, rows in grouped.items()
        if (selected := select_best(rows))
    }


def run_sweep(
    config: SweepConfig | str | Path,
    *,
    output: str | Path,
    resume: bool = False,
    execute: Callable[[SweepRun], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Execute runs sequentially and skip matching completed fingerprints."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict[str, Any]] = {}
    if resume and target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                previous[row.get("fingerprint", "")] = row
    results: list[dict[str, Any]] = []
    with target.open("a", encoding="utf-8") as handle:
        for run in expand_matrix(config):
            old = previous.get(run.fingerprint)
            if old and old.get("status") == "completed":
                results.append(old)
                continue
            row = {"index": run.index, "fingerprint": run.fingerprint, "values": run.values}
            try:
                if execute is None:
                    raise RuntimeError("a sweep executor is required for --run")
                row.update(execute(run))
            except Exception as exc:
                row.update({"status": "failed", "error": str(exc)})
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            results.append(row)
    return results
