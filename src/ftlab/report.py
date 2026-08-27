"""Privacy-safe reports from completed artifact-contract v2 runs."""

from __future__ import annotations

import base64
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import hash_file, validate_run_artifacts
from .data import normalize_record, validate_dataset_manifest_v2
from .metrics import (
    TASK_ERROR_TAXONOMY,
    _leaves,
    _same_value,
    _schema_at,
    _schema_for_tool,
    evaluate_predictions,
    paired_bootstrap_metric_delta_interval,
)
from .parser import parse_prediction
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
PROPORTION_METRICS = (
    "tool_accuracy",
    "json_validity",
    "schema_validity",
    "exact_match",
    "task_success",
)
MICRO_F1_METRICS = ("argument_key_f1", "argument_value_f1")
_BLANK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _dataset_provenance_valid(path: Path, manifest: dict[str, Any]) -> bool:
    """Accept the v2 descriptor reference used by a run, not a copied descriptor."""
    config = manifest.get("config")
    config = config if isinstance(config, dict) else {}
    dataset = config.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    version = str(dataset.get("version", config.get("dataset_version", "")))
    hashes = manifest.get("hashes")
    hashes = hashes if isinstance(hashes, dict) else {}
    if not version or not hashes.get("dataset"):
        return False

    project = path.parent.parent
    raw_manifest = project / "data" / version / "manifest.json"
    if raw_manifest.exists() and hash_file(raw_manifest) != hashes["dataset"]:
        return False
    descriptor = project / "data" / "manifests" / f"{version}.json"
    if not descriptor.exists():
        # Runs are portable. The content hash of their raw manifest is enough
        # evidence when the sanitized descriptor is not present in this checkout.
        return True
    value = _read_json(descriptor)
    if value is None:
        return False
    try:
        validate_dataset_manifest_v2(value)
    except ValueError:
        return False
    expected = hashes.get("dataset_manifest", manifest.get("dataset_manifest_hash"))
    return not expected or expected == value.get("content_sha256")


def _eligible_run(path: Path) -> bool:
    if path.name in LEGACY_RUN_IDS:
        return False
    manifest = _read_json(path / "manifest.json")
    status = _read_json(path / "status.json")
    if manifest is None or status is None:
        return False
    if manifest.get("artifact_version") != 2 or manifest.get("prompt_contract_version") != 1:
        return False
    controlled = manifest.get("controlled")
    if controlled is not True:
        controlled = (manifest.get("controlled_run") or {}).get("status") == "controlled"
    kind = manifest.get("kind", "evaluation")
    if kind not in {"training", "evaluation", "robustness", "bfcl"}:
        return False
    workflow = (manifest.get("config") or {}).get("workflow", "")
    if kind == "evaluation" and workflow not in {"validation", "final", "base", "tuned"}:
        return False
    if kind == "training" and not isinstance(manifest.get("selection"), dict):
        return False
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict) or not hashes.get("model") or not hashes.get("config"):
        return False
    if not isinstance(manifest.get("git"), dict) or not manifest["git"].get("commit"):
        return False
    return (
        controlled is True
        and status.get("status") == "completed"
        and manifest.get("status") == "completed"
        and _dataset_provenance_valid(path, manifest)
        and not validate_run_artifacts(path, kind=kind)
    )


def _system_summary(path: Path) -> dict[str, Any]:
    try:
        with (path / "system_metrics.csv").open(encoding="utf-8") as handle:
            samples = list(csv.DictReader(handle))
    except OSError:
        return {}
    output: dict[str, Any] = {}
    for field in (
        "rss_gib",
        "swap_used_gib",
        "mlx_allocator_active_bytes",
        "mlx_allocator_peak_bytes",
    ):
        values = [float(item[field]) for item in samples if item.get(field) not in {None, ""}]
        if values:
            output[field] = {"before": values[0], "peak": max(values), "after": values[-1]}
    return output


def _prediction_records(path: Path) -> tuple[list[Any], list[str], list[dict[str, Any]]] | None:
    """Read prediction rows only when they contain enough local gold data to audit."""
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError, json.JSONDecodeError:
        return None
    records = []
    predictions = []
    for index, value in enumerate(values):
        raw = value.get("record") if isinstance(value, dict) else None
        if not isinstance(raw, dict) or not isinstance(value.get("prediction"), str):
            return None
        # Worker rows deliberately omit user queries. Metrics use tool schemas
        # and gold calls, so restore only a fixed placeholder for normalization.
        normalized_raw = (
            raw if isinstance(raw.get("query"), str) else {**raw, "query": "[redacted]"}
        )
        record = normalize_record(normalized_raw, value.get("source_id", index))
        if not hasattr(record, "tools"):
            return None
        records.append(record)
        predictions.append(value["prediction"])
    return records, predictions, values


def _metrics_match(stored: dict[str, Any], computed: dict[str, Any]) -> bool:
    fields = (
        "count",
        "task_success",
        "argument_key_f1",
        "argument_value_f1",
        "schema_validity",
        "exact_match",
    )
    for field in fields:
        if field not in stored:
            return False
        if field == "count":
            if stored[field] != computed[field]:
                return False
        elif (
            not isinstance(stored[field], int | float)
            or abs(float(stored[field]) - float(computed[field])) > 1e-9
        ):
            return False
    return True


def _run_row(path: Path) -> dict[str, Any] | None:
    if not _eligible_run(path):
        return None
    evaluation = _read_json(path / "evaluation.json")
    manifest = _read_json(path / "manifest.json")
    if evaluation is None or manifest is None:
        return None
    audited = _prediction_records(path / "predictions.jsonl")
    if audited is not None:
        records, predictions, _ = audited
        computed = evaluate_predictions(records, predictions, confidence=False).as_dict()
        if not _metrics_match(evaluation, computed):
            return None
        evaluation = {**evaluation, **computed}
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
    recipe = manifest.get("recipe", evaluation.get("recipe", {}))
    if not isinstance(recipe, dict):
        recipe = {}
    memory = evaluation.get("allocator_memory_bytes", {})
    if not isinstance(memory, dict):
        memory = {}
    system_memory = _system_summary(path)
    system_rss = system_memory.get("rss_gib", {})
    system_rss = system_rss if isinstance(system_rss, dict) else {}
    return {
        "run": path.name,
        **evaluation,
        "model": config.get("model", manifest.get("model", "")),
        "dataset_version": dataset.get("version", config.get("dataset_version", "")),
        "controlled": manifest.get("controlled", True),
        "rss_gib": manifest.get(
            "rss_gib",
            evaluation.get("rss_gib", evaluation.get("peak_rss_gib", system_rss.get("peak"))),
        ),
        "prompt_processing_tokens_per_second": evaluation.get(
            "prompt_processing_tokens_per_second"
        ),
        "generation_throughput_tokens_per_second": evaluation.get(
            "generation_throughput_tokens_per_second",
            evaluation.get("throughput_tokens_per_second"),
        ),
        "throughput_tokens_per_second": evaluation.get(
            "generation_throughput_tokens_per_second",
            evaluation.get("throughput_tokens_per_second"),
        ),
        "memory_before_bytes": memory.get("before"),
        "memory_peak_bytes": memory.get("peak"),
        "memory_after_bytes": memory.get("after"),
        "system_memory": system_memory,
        "train_examples": counts.get("examples"),
        "rank": lora.get("rank"),
        "layers": training.get("num_layers"),
        "study_role": config.get(
            "study_role", manifest.get("study_role", evaluation.get("study_role", ""))
        ),
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
        "axis": config.get("axis", manifest.get("axis", evaluation.get("axis", ""))),
        "recipe": recipe,
        "task_errors": evaluation.get("task_error_categories", {}),
    }


def aggregate_evaluations(run_root: str | Path) -> list[dict[str, Any]]:
    """Read only completed v2 runs with prompt-contract v1 metadata."""
    root = Path(run_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.iterdir() if item.is_dir()) if root.exists() else []:
        row = _run_row(path)
        if row is not None:
            rows.append(row)
    return rows


def nondominated_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return actual quality-high, RSS-low nondominated report rows."""
    metrics = ("task_success", "argument_value_f1", "schema_validity")

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if not all(isinstance(left.get(key), (int, float)) for key in metrics):
            return False
        if not all(isinstance(right.get(key), (int, float)) for key in metrics):
            return False
        a_rss = float(left.get("rss_gib", float("inf")))
        b_rss = float(right.get("rss_gib", float("inf")))
        not_worse = all(float(left[key]) >= float(right[key]) for key in metrics) and a_rss <= b_rss
        strict = any(float(left[key]) > float(right[key]) for key in metrics) or a_rss < b_rss
        return not_worse and strict

    return sorted(
        [
            row
            for row in rows
            if not any(dominates(other, row) for other in rows if other is not row)
        ],
        key=lambda row: str(row["run"]),
    )


def _controlled_axis_rows(rows: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    """Keep only measurements explicitly produced for one study axis."""
    return [row for row in rows if row.get("axis") == axis]


def _headline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("study_role") == "headline"]


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
        audited = _prediction_records(prediction_path)
        checks.append(
            {
                "run": row["run"],
                "prediction_rows": count,
                "evaluation_count": row.get("count"),
                "row_count_match": count is not None and count == row.get("count"),
                "metric_audit": "verified" if audited is not None else "unavailable",
            }
        )
    return {
        "checks": checks,
        "all_row_counts_match": all(item["row_count_match"] for item in checks),
        "audit_status": "verified"
        if checks and all(item["metric_audit"] == "verified" for item in checks)
        else "audit unavailable",
    }


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


def _per_record_metrics(records: list[Any], predictions: list[str]) -> dict[str, list[Any]]:
    """Return scalar and micro-F1 inputs in the same form as published metrics."""
    result: dict[str, list[Any]] = {metric: [] for metric in PROPORTION_METRICS + MICRO_F1_METRICS}
    for record, prediction in zip(records, predictions, strict=True):
        single = evaluate_predictions([record], [prediction], confidence=False)
        for metric in PROPORTION_METRICS:
            result[metric].append(float(getattr(single, metric)))
        parsed = parse_prediction(prediction, record.tools)
        comparable = parsed.error is None or parsed.error.category == "schema_invalid"
        predicted = _leaves(parsed.arguments) if comparable else {}
        gold = _leaves(record.arguments)
        schema = _schema_for_tool(record)
        common = set(gold) & set(predicted)
        key_counts = (len(common), len(predicted), len(gold))
        value_counts = (
            sum(
                int(_same_value(gold[path], predicted[path], _schema_at(schema, path)))
                for path in common
            ),
            len(predicted),
            len(gold),
        )
        result["argument_key_f1"].append(key_counts)
        result["argument_value_f1"].append(value_counts)
    return result


def _paired_delta_intervals(base: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic paired tuned-minus-base intervals from raw audit rows."""
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
    pairs: list[dict[str, Any]] = []
    unavailable = False
    for key in sorted(grouped, key=str):
        pair = grouped[key]
        if set(pair) != {"base", "tuned"}:
            continue
        base_audit = _prediction_records(base / "runs" / pair["base"]["run"] / "predictions.jsonl")
        tuned_audit = _prediction_records(
            base / "runs" / pair["tuned"]["run"] / "predictions.jsonl"
        )
        if base_audit is None or tuned_audit is None:
            unavailable = True
            continue
        base_records, base_predictions, _ = base_audit
        tuned_records, tuned_predictions, _ = tuned_audit
        base_by_id = dict(
            zip((record.source_id for record in base_records), base_predictions, strict=True)
        )
        tuned_by_id = dict(
            zip((record.source_id for record in tuned_records), tuned_predictions, strict=True)
        )
        if set(base_by_id) != set(tuned_by_id):
            unavailable = True
            continue
        source_ids = sorted(base_by_id)
        record_by_id = {record.source_id: record for record in base_records}
        base_values = _per_record_metrics(
            [record_by_id[source_id] for source_id in source_ids],
            [base_by_id[source_id] for source_id in source_ids],
        )
        tuned_record_by_id = {record.source_id: record for record in tuned_records}
        if any(
            record_by_id[source_id] != tuned_record_by_id[source_id] for source_id in source_ids
        ):
            unavailable = True
            continue
        tuned_values = _per_record_metrics(
            [tuned_record_by_id[source_id] for source_id in source_ids],
            [tuned_by_id[source_id] for source_id in source_ids],
        )
        intervals = {
            metric: {
                "low": low,
                "high": high,
            }
            for metric in PROPORTION_METRICS + MICRO_F1_METRICS
            for low, high in [
                paired_bootstrap_metric_delta_interval(
                    base_values[metric],
                    tuned_values[metric],
                    metric=metric,
                    base_source_ids=source_ids,
                    tuned_source_ids=source_ids,
                )
            ]
        }
        pairs.append(
            {
                "base_run": pair["base"]["run"],
                "tuned_run": pair["tuned"]["run"],
                "source_ids": source_ids,
                "tuned_minus_base_95ci": intervals,
            }
        )
    return {
        "status": "audit unavailable" if unavailable and not pairs else "verified",
        "pairs": pairs,
    }


def _representative_pairs(base: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    audit_unavailable = False
    for key in sorted(grouped, key=str):
        pair = grouped[key]
        if set(pair) != {"base", "tuned"} or not pair["base"].get("prompt_hash"):
            continue
        base_audit = _prediction_records(base / "runs" / pair["base"]["run"] / "predictions.jsonl")
        tuned_audit = _prediction_records(
            base / "runs" / pair["tuned"]["run"] / "predictions.jsonl"
        )
        if base_audit is None or tuned_audit is None:
            audit_unavailable = True
            continue
        base_records, base_predictions, base_rows = base_audit
        tuned_records, tuned_predictions, tuned_rows = tuned_audit
        base_by_id = {
            record.source_id: (record, prediction, row)
            for record, prediction, row in zip(base_records, base_predictions, base_rows)
        }
        tuned_by_id = {
            record.source_id: (record, prediction, row)
            for record, prediction, row in zip(tuned_records, tuned_predictions, tuned_rows)
        }
        for source_id in sorted(set(base_by_id) & set(tuned_by_id)):
            base_record, base_prediction, _ = base_by_id[source_id]
            tuned_record, tuned_prediction, _ = tuned_by_id[source_id]
            if base_record != tuned_record:
                continue
            base_success = evaluate_predictions(
                [tuned_record], [base_prediction], confidence=False
            ).task_success
            tuned_success = evaluate_predictions(
                [tuned_record], [tuned_prediction], confidence=False
            ).task_success
            if base_success == 0.0 and tuned_success == 1.0:
                output.append(
                    {
                        "source_id": source_id,
                        "base_prediction": base_prediction,
                        "tuned_prediction": tuned_prediction,
                    }
                )
                if len(output) == 5:
                    return {"status": "verified", "examples": output}
    return {
        "status": "audit unavailable" if audit_unavailable and not output else "verified",
        "examples": output,
    }


def _official_final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("workflow") == "final" or str(row.get("run", "")).startswith("final-")
    ]


def _write_final_evidence(
    output: Path,
    final_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    paired_intervals: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    """Write tracked, sanitized final evidence only after the raw-audit gate."""
    eligible = bool(final_rows) and audit.get("audit_status") == "verified"
    eligible = eligible and audit.get("all_row_counts_match") is True
    eligible = eligible and paired_intervals.get("status") != "audit unavailable"
    if not eligible:
        return {
            "status": "nonfinal",
            "reason": "official final raw audit is unavailable or invalid",
        }
    final = output / "final"
    tables = final / "tables"
    figures = final / "figures"
    findings = final / "findings"
    for directory in (tables, figures, findings):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "final-aggregate.csv", final_rows, fields)
    write_json(tables / "paired-delta-intervals.json", paired_intervals)
    _write_figure(
        figures / "quality-vs-rss-pareto.png",
        "Final quality vs RSS Pareto",
        nondominated_pareto(final_rows),
        "rss_gib",
        "task_success",
    )
    write_json(findings / "raw-to-table-audit.json", audit)
    (findings / "findings.md").write_text(
        "# Final findings\n\nStatus: audited final evidence.\n",
        encoding="utf-8",
    )
    return {"status": "final", "path": str(final)}


def build_report(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    run_root = base / "runs"
    discovered = (
        sum(1 for item in run_root.iterdir() if item.is_dir() and item.name not in {".ftlab.lock"})
        if run_root.exists()
        else 0
    )
    rows = aggregate_evaluations(run_root)
    headline = _headline_rows(rows)
    pareto = nondominated_pareto(headline)
    output = base / "reports"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate.json", rows)
    write_json(output / "pareto.json", pareto)
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
        "prompt_processing_tokens_per_second",
        "generation_throughput_tokens_per_second",
        "throughput_tokens_per_second",
        "memory_before_bytes",
        "memory_peak_bytes",
        "memory_after_bytes",
    ]
    _write_csv(output / "aggregate.csv", rows, fields)

    scaling = _controlled_axis_rows(rows, "dataset_size")
    capacity = [
        row
        for row in rows
        if row.get("axis") in {"adapter_rank", "adapter_layers", "adapter_targets"}
    ]
    model_frontier = _controlled_axis_rows(rows, "model_size")

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
        task_errors = row.get("task_errors", {})
        task_errors = task_errors if isinstance(task_errors, dict) else {}
        for category in TASK_ERROR_TAXONOMY:
            count = task_errors.get(category, 0)
            error_rows.append({"run": row["run"], "category": category, "count": count})

    _write_figure(
        output / "quality-vs-rss-pareto.png",
        "Quality vs RSS Pareto (preliminary)",
        pareto,
        "rss_gib",
        "task_success",
    )
    _write_precision_table(output / "headline-precision-table.png", headline)
    _write_heatmap(
        output / "rank-by-layer-heatmap.png",
        "Rank by layer (preliminary)",
        capacity,
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
        model_frontier,
        "generation_throughput_tokens_per_second",
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
    raw_audit = _audit_predictions(base, rows)
    paired_intervals = _paired_delta_intervals(base, rows)
    write_json(output / "three-seed-summary.json", _three_seed_summary(rows))
    write_json(output / "representative-examples.json", _representative_pairs(base, rows))
    write_json(output / "paired-delta-intervals.json", paired_intervals)
    write_json(output / "raw-to-table-audit.json", raw_audit)
    final_rows = _official_final_rows(rows)
    final_audit = _audit_predictions(base, final_rows)
    final_evidence = _write_final_evidence(
        output, final_rows, final_audit, paired_intervals, fields
    )
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
        "pareto": str(output / "pareto.json"),
        "final_evidence": final_evidence,
    }
