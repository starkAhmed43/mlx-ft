"""Privacy-safe reports from completed artifact-contract v1 runs."""

from __future__ import annotations

import base64
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import validate_run_artifacts
from .robustness import synthetic_no_call_cases
from .storage import write_json

LEGACY_RUN_IDS = {
    "20260809T104310.301493Z-base-test",
    "20260809T104939.096339Z-base-test",
    "20260809T105219.723552Z-smoke-06b",
    "20260809T105438.796044Z-tuned-test",
}
FIGURE_NAMES = (
    "quality-vs-rss-pareto.png",
    "headline-precision-table.png",
    "rank-by-layer-heatmap.png",
    "data-scaling-curve.png",
    "distractor-robustness-curve.png",
    "model-size-quality-throughput-frontier.png",
    "error-distribution-comparison.png",
)
TABLE_NAMES = (
    "headline-precision-table.csv",
    "data-scaling.csv",
    "distractor-robustness.csv",
    "error-distribution.csv",
)
_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _eligible_run(path: Path) -> bool:
    if path.name in LEGACY_RUN_IDS:
        return False
    manifest = _read_json(path / "manifest.json")
    status = _read_json(path / "status.json")
    if manifest is None or status is None:
        return False
    if manifest.get("artifact_version") != 1 or manifest.get("prompt_contract_version") != 1:
        return False
    controlled = manifest.get("controlled")
    if controlled is not True:
        controlled = (manifest.get("controlled_run") or {}).get("status") == "controlled"
    kind = manifest.get("kind", "evaluation")
    if kind not in {"training", "evaluation", "robustness", "bfcl"}:
        return False
    return (
        controlled is True
        and status.get("status") == "completed"
        and manifest.get("status") == "completed"
        and not validate_run_artifacts(path, kind=kind)
    )


def _run_row(path: Path) -> dict[str, Any] | None:
    if not _eligible_run(path):
        return None
    evaluation = _read_json(path / "evaluation.json")
    manifest = _read_json(path / "manifest.json")
    if evaluation is None or manifest is None:
        return None
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        config = {}
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {}
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    lora = training.get("lora_parameters", {})
    if not isinstance(lora, dict):
        lora = {}
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    return {
        "run": path.name,
        **evaluation,
        "model": config.get("model", manifest.get("model", "")),
        "dataset_version": dataset.get("version", ""),
        "controlled": manifest.get("controlled", True),
        "rss_gib": manifest.get("rss_gib", evaluation.get("peak_rss_gib")),
        "throughput_tokens_per_second": evaluation.get("throughput_tokens_per_second"),
        "train_examples": counts.get("examples"),
        "rank": lora.get("rank"),
        "layers": training.get("num_layers"),
        "study_role": config.get("study_role", manifest.get("study_role", "")),
        "kind": manifest.get("kind", ""),
        "workflow": config.get("workflow", ""),
        "git_commit": (manifest.get("git") or {}).get("commit"),
        "provenance": bool((manifest.get("git") or {}).get("commit")),
        "seed": training.get("seed", manifest.get("seed")),
        "precision": config.get("precision")
        or (
            "bf16"
            if "-bf16" in str(config.get("model", ""))
            else "8bit"
            if "-8bit" in str(config.get("model", ""))
            else "4bit"
            if "-4bit" in str(config.get("model", ""))
            else ""
        ),
        "prompt_hash": manifest.get("prompt_hash", evaluation.get("prompt_hash")),
        "decoding_hash": manifest.get("decoding_hash", evaluation.get("decoding_hash")),
        "winner_source_run": manifest.get("winner_source_run"),
    }


def aggregate_evaluations(run_root: str | Path) -> list[dict[str, Any]]:
    """Read only completed v1 runs with prompt-contract v1 metadata."""
    root = Path(run_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()) if root.exists() else []:
        row = _run_row(path)
        if row is not None:
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, title: str, rows: list[dict[str, Any]], x: str, y: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        path.write_bytes(_BLANK_PNG)
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    points: list[tuple[float, float]] = [
        (float(row[x]), float(row[y]))
        for row in rows
        if isinstance(row.get(x), (int, float)) and isinstance(row.get(y), (int, float))
    ]
    if points:
        axis.scatter([item[0] for item in points], [item[1] for item in points])
        for row in rows:
            if (
                isinstance(row.get(x), (int, float))
                and isinstance(row.get(y), (int, float))
                and row.get("model")
            ):
                axis.annotate(
                    str(row["model"]).split("/")[-1],
                    (float(row[x]), float(row[y])),
                    fontsize=7,
                )
        axis.set_xlabel(x)
        axis.set_ylabel(y)
    else:
        axis.text(0.5, 0.5, "preliminary / no data", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_precision_table(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        path.write_bytes(_BLANK_PNG)
        return
    figure, axis = plt.subplots(figsize=(8, max(2.0, 0.5 * len(rows) + 1)))
    axis.axis("off")
    if rows:
        values = [
            [
                row.get("model", ""),
                row.get("precision", ""),
                row.get("task_success", ""),
                row.get("argument_value_f1", ""),
                row.get("schema_validity", ""),
            ]
            for row in rows
        ]
        axis.table(
            cellText=values,
            colLabels=["model", "precision", "task_success", "value_f1", "schema"],
            loc="center",
        )
    else:
        axis.text(0.5, 0.5, "preliminary / no data", ha="center", va="center")
    axis.set_title("Headline precision (preliminary)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_heatmap(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        path.write_bytes(_BLANK_PNG)
        return
    pairs = [
        row
        for row in rows
        if isinstance(row.get("rank"), (int, float))
        and isinstance(row.get("layers"), (int, float))
        and isinstance(row.get("task_success"), (int, float))
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    if pairs:
        ranks = sorted({int(row["rank"]) for row in pairs})
        layers = sorted({int(row["layers"]) for row in pairs})
        matrix = [
            [
                next(
                    (
                        float(row["task_success"])
                        for row in pairs
                        if int(row["rank"]) == rank and int(row["layers"]) == layer
                    ),
                    float("nan"),
                )
                for rank in ranks
            ]
            for layer in layers
        ]
        image = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_xticks(range(len(ranks)))
        axis.set_xticklabels([str(value) for value in ranks])
        axis.set_yticks(range(len(layers)))
        axis.set_yticklabels([str(value) for value in layers])
        axis.set_xlabel("adapter rank")
        axis.set_ylabel("adapter layers")
        figure.colorbar(image, ax=axis, label="task success")
    else:
        axis.text(0.5, 0.5, "preliminary / no data", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_error_figure(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        path.write_bytes(_BLANK_PNG)
        return
    figure, axis = plt.subplots(figsize=(8, 4))
    if rows:
        labels = [str(row["category"]) for row in rows]
        values = [float(row["count"]) for row in rows]
        axis.bar(range(len(labels)), values)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel("count")
    else:
        axis.text(0.5, 0.5, "preliminary / no data", ha="center", va="center")
        axis.set_axis_off()
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _audit_predictions(base: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for row in rows:
        prediction_path = base / "runs" / str(row["run"]) / "predictions.jsonl"
        count = (
            sum(
                bool(line.strip())
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
            )
            if prediction_path.exists()
            else None
        )
        checks.append(
            {
                "run": row["run"],
                "prediction_rows": count,
                "evaluation_count": row.get("count"),
                "match": count is not None and count == row.get("count"),
            }
        )
    return {"checks": checks, "all_match": all(item["match"] for item in checks)}


def _three_seed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    source_runs = {
        str(row.get("winner_source_run"))
        for row in rows
        if row.get("study_role") == "winner_seed" and row.get("winner_source_run")
    }
    for row in rows:
        if row.get("study_role") == "winner_seed" or row.get("run") in source_runs:
            key = (
                row.get("model"),
                row.get("dataset_version"),
                row.get("rank"),
                row.get("layers"),
            )
            groups[key].append(row)
    for key, values in groups.items():
        if len(values) >= 3:
            metrics = {}
            for field in ("task_success", "argument_value_f1", "schema_validity"):
                numbers = [
                    float(item[field])
                    for item in values
                    if isinstance(item.get(field), (int, float))
                ]
                if len(numbers) >= 3:
                    metrics[field] = {
                        "mean": statistics.fmean(numbers),
                        "sample_std": statistics.stdev(numbers),
                    }
            return {
                "status": "measured",
                "group": list(key),
                "count": len(values),
                "metrics": metrics,
            }
    return {"status": "preliminary", "runs": []}


def _representative_pairs(base: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("workflow") in {"base", "tuned"}:
            key = (
                row.get("model"),
                row.get("dataset_version"),
                row.get("prompt_hash"),
                row.get("decoding_hash"),
            )
            grouped[key][str(row["workflow"])] = row
    output: list[dict[str, Any]] = []
    for pair in grouped.values():
        if set(pair) != {"base", "tuned"} or not pair["base"].get("prompt_hash"):
            continue
        base_rows = [
            json.loads(line)
            for line in (base / "runs" / pair["base"]["run"] / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        tuned_rows = [
            json.loads(line)
            for line in (base / "runs" / pair["tuned"]["run"] / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        tuned_by_id = {str(item.get("source_id")): item for item in tuned_rows}
        for item in base_rows:
            source_id = str(item.get("source_id"))
            tuned = tuned_by_id.get(source_id)
            if tuned is not None:
                output.append(
                    {
                        "source_id": source_id,
                        "base_prediction": item.get("prediction", ""),
                        "tuned_prediction": tuned.get("prediction", ""),
                    }
                )
                if len(output) == 5:
                    return output
    return output


def build_report(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    run_root = base / "runs"
    discovered = (
        sum(1 for item in run_root.iterdir() if item.is_dir() and item.name not in {".ftlab.lock"})
        if run_root.exists()
        else 0
    )
    rows = aggregate_evaluations(run_root)
    output = base / "reports"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate.json", rows)
    fields = [
        "run",
        "model",
        "precision",
        "dataset_version",
        "study_role",
        "count",
        "train_examples",
        "rank",
        "layers",
        "task_success",
        "argument_value_f1",
        "schema_validity",
        "rss_gib",
        "throughput_tokens_per_second",
    ]
    _write_csv(output / "aggregate.csv", rows, fields)

    headline = [row for row in rows if row.get("study_role") == "headline"]
    scaling = [row for row in rows if row.get("train_examples")]

    robustness_path = output / "robustness.json"
    eligible_robustness = [row for row in rows if row.get("kind") == "robustness"]
    if eligible_robustness:
        robustness = {
            key: eligible_robustness[-1].get(key)
            for key in ("call_success", "call_success_by_distractors", "no_call_accuracy")
        }
        robustness["status"] = "preliminary"
        robustness["source_runs"] = [row["run"] for row in eligible_robustness]
    else:
        robustness = {
            "suite": "synthetic_no_call",
            "count": len(synthetic_no_call_cases()),
            "status": "preliminary",
            "no_call_accuracy": None,
        }
    write_json(robustness_path, robustness)

    robustness_rows = [
        {"distractors": int(key), "call_success": value}
        for key, value in (robustness.get("call_success_by_distractors", {}) or {}).items()
    ]
    error_rows: list[dict[str, Any]] = []
    for row in rows:
        for category, count in (row.get("parser_errors", {}) or {}).items():
            error_rows.append({"run": row["run"], "category": category, "count": count})

    _write_figure(
        output / "quality-vs-rss-pareto.png",
        "Quality vs RSS (preliminary)",
        rows,
        "rss_gib",
        "task_success",
    )
    _write_precision_table(output / "headline-precision-table.png", headline)
    _write_heatmap(
        output / "rank-by-layer-heatmap.png",
        "Rank by layer (preliminary)",
        rows,
    )
    _write_figure(
        output / "data-scaling-curve.png",
        "Data scaling (preliminary)",
        scaling,
        "train_examples",
        "task_success",
    )
    _write_figure(
        output / "distractor-robustness-curve.png",
        "Distractor robustness (preliminary)",
        robustness_rows,
        "distractors",
        "call_success",
    )
    _write_figure(
        output / "model-size-quality-throughput-frontier.png",
        "Model-size frontier (preliminary)",
        rows,
        "throughput_tokens_per_second",
        "task_success",
    )
    _write_error_figure(
        output / "error-distribution-comparison.png",
        "Error distribution (preliminary)",
        error_rows,
    )
    _write_figure(
        output / "task-success.png", "Task success (preliminary)", rows, "count", "task_success"
    )
    _write_csv(output / "headline-precision-table.csv", headline, fields)
    _write_csv(output / "headline-precision.csv", headline, fields)
    _write_csv(output / "data-scaling.csv", scaling, fields)
    _write_csv(
        output / "distractor-robustness.csv", robustness_rows, ["distractors", "call_success"]
    )
    _write_csv(output / "error-distribution.csv", error_rows, ["run", "category", "count"])

    write_json(
        output / "ci-summary.json",
        {row["run"]: row.get("confidence_intervals", {}) for row in rows},
    )
    write_json(output / "three-seed-summary.json", _three_seed_summary(rows))
    write_json(output / "representative-examples.json", _representative_pairs(base, rows))
    write_json(output / "raw-to-table-audit.json", _audit_predictions(base, rows))
    (output / "findings.md").write_text(
        "# Findings\n\nStatus: preliminary. No quality claim or resume bullet is emitted.\n"
        "Legacy prompt-contract-0 runs are excluded. Final measurements require completed "
        "controlled runs with commit and provenance.\n",
        encoding="utf-8",
    )
    return {
        "rows": discovered,
        "eligible_rows": len(rows),
        "aggregate": str(output / "aggregate.csv"),
        "plot": str(output / "task-success.png"),
        "robustness": str(robustness_path),
        "figures": [str(output / name) for name in FIGURE_NAMES],
        "tables": [str(output / name) for name in TABLE_NAMES],
    }
