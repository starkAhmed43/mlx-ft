"""Function-call metrics with independent validity and correctness fields."""

from __future__ import annotations

import random
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .data import AcceptedRecord
from .parser import PRIMARY_TAXONOMY, ParsedCall, parse_prediction
from .rendering import TOOL_END, TOOL_START

TASK_ERROR_TAXONOMY = (
    "no_parseable_tool_call_boundary",
    "invalid_json",
    "nonexistent_tool",
    "wrong_supplied_tool",
    "missing_required_argument",
    "extra_hallucinated_argument",
    "wrong_type",
    "wrong_enum_or_value",
    "should_not_call_tool",
    "truncated_output",
)


def canonicalize(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value).strip()
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def _schema_at(schema: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    current: Any = schema
    for key in path:
        if not isinstance(current, dict):
            return None
        if current.get("type") == "object":
            current = current.get("properties", {}).get(key)
        elif current.get("type") == "array" and key.isdigit():
            current = current.get("items")
        else:
            return None
    return current if isinstance(current, dict) else None


def _same_value(left: Any, right: Any, schema: dict[str, Any] | None) -> bool:
    if schema is not None and "enum" in schema:
        # Enum members are protocol values. Do not apply the string
        # normalization used for ordinary argument strings.
        return type(left) is type(right) and left == right
    left, right = canonicalize(left), canonicalize(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if type(left) is type(right):
            return left == right
        return schema is not None and schema.get("type") == "number" and float(left) == float(right)
    return type(left) is type(right) and left == right


def _leaves(value: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], Any] = {}
        for key in sorted(value):
            result.update(_leaves(value[key], (*path, key)))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_leaves(item, (*path, str(index))))
        return result
    return {path: value}


def _has_extra_argument(gold: Any, predicted: Any) -> bool:
    if isinstance(predicted, dict):
        if not isinstance(gold, dict):
            return True
        return any(
            key not in gold or _has_extra_argument(gold[key], value)
            for key, value in predicted.items()
        )
    if isinstance(predicted, list):
        if not isinstance(gold, list) or len(predicted) > len(gold):
            return True
        return any(_has_extra_argument(gold[index], value) for index, value in enumerate(predicted))
    return False


def _schema_for_tool(record: AcceptedRecord) -> dict[str, Any]:
    return next(
        (tool["parameters"] for tool in record.tools if tool["name"] == record.function_name),
        {},
    )


def _required_leaves(
    value: Any, schema: dict[str, Any], path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    """Return gold values for schema-required arguments, including nested fields."""
    if not isinstance(value, dict) or schema.get("type") != "object":
        return {path: value}
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    result: dict[tuple[str, ...], Any] = {}
    for key in required:
        if key not in value:
            continue
        child_schema = properties.get(key, {})
        child_value = value[key]
        if isinstance(child_value, dict) and child_schema.get("type") == "object":
            nested = _required_leaves(child_value, child_schema, (*path, key))
            result.update(nested or {(*path, key): child_value})
        else:
            result[(*path, key)] = child_value
    return result


def _first_schema_error_category(record: AcceptedRecord, parsed: ParsedCall) -> str | None:
    """Classify schema/task failures in the published, ordered taxonomy."""
    if parsed.error is not None and parsed.error.category != "schema_invalid":
        if parsed.error.category in {"truncation", "partial_boundary"}:
            return "truncated_output"
        if parsed.error.category == "invalid_json":
            return "invalid_json"
        if parsed.error.category == "unknown_tool":
            return "nonexistent_tool"
        return "no_parseable_tool_call_boundary"
    if parsed.name != record.function_name:
        return "wrong_supplied_tool"

    schema = _schema_for_tool(record)
    arguments = parsed.arguments
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda error: (list(error.path), error.validator),
    )
    validators = {error.validator for error in errors}
    if "required" in validators:
        return "missing_required_argument"
    if "additionalProperties" in validators:
        return "extra_hallucinated_argument"
    if "type" in validators:
        return "wrong_type"
    if "enum" in validators:
        return "wrong_enum_or_value"

    gold_leaves = _leaves(record.arguments)
    predicted_leaves = _leaves(arguments)
    for path in sorted(set(gold_leaves) & set(predicted_leaves)):
        if not _same_value(gold_leaves[path], predicted_leaves[path], _schema_at(schema, path)):
            return "wrong_enum_or_value"
    return None


def _same_argument(
    left: Any, right: Any, schema: dict[str, Any], path: tuple[str, ...] = ()
) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            return False
        return all(_same_argument(left[key], right[key], schema, (*path, key)) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(
            _same_argument(left[index], right[index], schema, (*path, str(index)))
            for index in range(len(left))
        )
    return _same_value(left, right, _schema_at(schema, path))


def _f1(precision_count: int, recall_count: int, predicted: int, gold: int) -> float:
    if predicted == 0 and gold == 0:
        return 1.0
    if precision_count == 0 or recall_count == 0:
        return 0.0
    precision = precision_count / predicted
    recall = recall_count / gold
    return 2 * precision * recall / (precision + recall)


def _interval(samples: list[float], confidence: float) -> tuple[float, float]:
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return (
        samples[max(0, int(alpha * len(samples)))],
        samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))],
    )


def _micro_counts(value: float | tuple[int, int, int]) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(isinstance(item, int) and item >= 0 for item in value)
    ):
        raise ValueError("micro-F1 values must be (true_positive, predicted, gold) tuples")
    return value


def _scalar_value(value: float | tuple[int, int, int]) -> float:
    if isinstance(value, tuple):
        raise ValueError("scalar metric values must be numeric")
    return float(value)


def bootstrap_micro_f1_interval(
    counts: Sequence[tuple[int, int, int]],
    *,
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a micro-F1 by recomputing TP, predicted, and gold totals."""
    if not counts:
        return (0.0, 0.0)
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must be between zero and one")
    if any(matches < 0 or predicted < 0 or gold < 0 for matches, predicted, gold in counts):
        raise ValueError("micro-F1 counts must be non-negative")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [counts[rng.randrange(len(counts))] for _ in counts]
        matches = sum(item[0] for item in selected)
        predicted = sum(item[1] for item in selected)
        gold = sum(item[2] for item in selected)
        draws.append(_f1(matches, matches, predicted, gold))
    return _interval(draws, confidence)


@dataclass(frozen=True)
class Evaluation:
    count: int
    tool_accuracy: float
    json_validity: float
    schema_validity: float
    argument_key_f1: float
    argument_value_f1: float
    exact_match: float
    task_success: float
    parser_errors: dict[str, int]
    primary_categories: dict[str, int] | None = None
    task_error_categories: dict[str, int] | None = None
    extra_argument_rate: float = 0.0
    extra_argument_count: int = 0
    confidence_intervals: dict[str, tuple[float, float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "tool_accuracy": self.tool_accuracy,
            "json_validity": self.json_validity,
            "schema_validity": self.schema_validity,
            "argument_key_f1": self.argument_key_f1,
            "argument_value_f1": self.argument_value_f1,
            "exact_match": self.exact_match,
            "task_success": self.task_success,
            "parser_errors": dict(self.parser_errors),
            "primary_categories": {
                category: (self.primary_categories or {}).get(category, 0)
                for category in PRIMARY_TAXONOMY
            },
            "task_error_categories": {
                category: (self.task_error_categories or {}).get(category, 0)
                for category in TASK_ERROR_TAXONOMY
            },
            "extra_argument_rate": self.extra_argument_rate,
            "extra_argument_count": self.extra_argument_count,
            "confidence_intervals": {
                key: {"low": low, "high": high}
                for key, (low, high) in (self.confidence_intervals or {}).items()
            },
        }


def evaluate_predictions(
    records: Sequence[AcceptedRecord],
    predictions: Iterable[str],
    *,
    confidence: bool = True,
    prediction_diagnostics: Sequence[dict[str, Any]] | None = None,
) -> Evaluation:
    prediction_list = list(predictions)
    if len(prediction_list) != len(records):
        raise ValueError(
            f"prediction count {len(prediction_list)} does not match record count {len(records)}"
        )
    tool_hits = json_hits = schema_hits = exact_hits = success_hits = 0
    extra_argument_count = 0
    key_matches = predicted_keys = gold_keys = 0
    value_matches = predicted_values = gold_values = 0
    key_f1_counts: list[tuple[int, int, int]] = []
    value_f1_counts: list[tuple[int, int, int]] = []
    series: dict[str, list[float]] = {
        "tool_accuracy": [],
        "json_validity": [],
        "schema_validity": [],
        "argument_key_f1": [],
        "argument_value_f1": [],
        "exact_match": [],
        "task_success": [],
    }
    parser_errors: Counter[str] = Counter()
    primary_categories: Counter[str] = Counter()
    task_error_categories: Counter[str] = Counter()
    diagnostics = prediction_diagnostics or [{} for _ in prediction_list]
    if len(diagnostics) != len(prediction_list):
        raise ValueError("prediction diagnostics must match prediction count")
    for record, text, diagnostic in zip(records, prediction_list, diagnostics):
        gold_leaves = _leaves(record.arguments)
        parsed: ParsedCall = parse_prediction(
            text,
            record.tools,
            generation_truncated=bool(diagnostic.get("truncated", False)),
            max_generation_tokens=diagnostic.get("max_generation_tokens"),
            generated_tokens=diagnostic.get("generated_tokens"),
        )
        primary_categories[parsed.primary_category] += 1
        json_hits += int(parsed.json_valid)
        tool_hits += int(parsed.tool_valid and parsed.name == record.function_name)
        schema_hits += int(parsed.schema_valid)
        series["json_validity"].append(float(parsed.json_valid))
        series["tool_accuracy"].append(
            float(parsed.tool_valid and parsed.name == record.function_name)
        )
        series["schema_validity"].append(float(parsed.schema_valid))
        if parsed.error is not None:
            parser_errors[parsed.error.category] += 1
        task_error = _first_schema_error_category(record, parsed)
        if task_error is not None:
            task_error_categories[task_error] += 1

        comparable = parsed.error is None or parsed.error.category == "schema_invalid"
        predicted_leaves = _leaves(parsed.arguments) if comparable else {}
        tool_ok = parsed.tool_valid and parsed.name == record.function_name
        schema = _schema_for_tool(record)
        gold_paths, predicted_paths = set(gold_leaves), set(predicted_leaves)
        record_key_matches = len(gold_paths & predicted_paths)
        record_value_matches = sum(
            int(_same_value(gold_leaves[path], predicted_leaves[path], _schema_at(schema, path)))
            for path in gold_paths & predicted_paths
        )
        key_matches += record_key_matches
        predicted_keys += len(predicted_paths)
        gold_keys += len(gold_paths)
        value_matches += record_value_matches
        predicted_values += len(predicted_paths)
        gold_values += len(gold_paths)
        key_f1 = _f1(record_key_matches, record_key_matches, len(predicted_paths), len(gold_paths))
        value_f1 = _f1(
            record_value_matches, record_value_matches, len(predicted_paths), len(gold_paths)
        )
        key_f1_counts.append((record_key_matches, len(predicted_paths), len(gold_paths)))
        value_f1_counts.append((record_value_matches, len(predicted_paths), len(gold_paths)))
        extra_argument_count += int(_has_extra_argument(record.arguments, parsed.arguments))
        exact = (
            parsed.error is None
            and tool_ok
            and _same_argument(record.arguments, parsed.arguments, schema)
        )
        required_gold = _required_leaves(record.arguments, schema)
        required_correct = all(
            path in predicted_leaves
            and _same_value(gold_value, predicted_leaves[path], _schema_at(schema, path))
            for path, gold_value in required_gold.items()
        )
        task_success = parsed.error is None and tool_ok and parsed.schema_valid and required_correct
        exact_hits += int(exact)
        success_hits += int(task_success)
        series["argument_key_f1"].append(key_f1)
        series["argument_value_f1"].append(value_f1)
        series["exact_match"].append(float(exact))
        series["task_success"].append(float(task_success))
    count = len(records)
    return Evaluation(
        count=count,
        tool_accuracy=tool_hits / count if count else 0.0,
        json_validity=json_hits / count if count else 0.0,
        schema_validity=schema_hits / count if count else 0.0,
        argument_key_f1=_f1(key_matches, key_matches, predicted_keys, gold_keys),
        argument_value_f1=_f1(value_matches, value_matches, predicted_values, gold_values),
        exact_match=exact_hits / count if count else 0.0,
        task_success=success_hits / count if count else 0.0,
        parser_errors=dict(parser_errors),
        primary_categories=dict(primary_categories),
        task_error_categories={
            category: task_error_categories[category] for category in TASK_ERROR_TAXONOMY
        },
        extra_argument_rate=extra_argument_count / count if count else 0.0,
        extra_argument_count=extra_argument_count,
        confidence_intervals=(
            {
                **{
                    name: paired_bootstrap_interval(values)
                    for name, values in series.items()
                    if name not in {"argument_key_f1", "argument_value_f1"}
                },
                "argument_key_f1": bootstrap_micro_f1_interval(key_f1_counts),
                "argument_value_f1": bootstrap_micro_f1_interval(value_f1_counts),
            }
            if confidence
            else {}
        ),
    )


def paired_bootstrap_interval(
    values: Sequence[float], *, seed: int = 42, samples: int = 10_000, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a deterministic paired bootstrap interval for a proportion or F1 series."""
    if not values:
        return (0.0, 0.0)
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must be between zero and one")
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    ]
    return _interval(means, confidence)


def paired_bootstrap_delta_interval(
    base_values: Sequence[float],
    tuned_values: Sequence[float],
    *,
    base_source_ids: Sequence[str],
    tuned_source_ids: Sequence[str],
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a paired bootstrap interval for tuned minus base values.

    Stable source IDs are required because paired inference is invalid when
    predictions are compared by their incidental input order.
    """
    return paired_bootstrap_metric_delta_interval(
        base_values,
        tuned_values,
        metric="scalar",
        base_source_ids=base_source_ids,
        tuned_source_ids=tuned_source_ids,
        seed=seed,
        samples=samples,
        confidence=confidence,
    )


def paired_bootstrap_metric_delta_interval(
    base_values: Sequence[float | tuple[int, int, int]],
    tuned_values: Sequence[float | tuple[int, int, int]],
    *,
    metric: str,
    base_source_ids: Sequence[str],
    tuned_source_ids: Sequence[str],
    seed: int = 42,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap the aligned tuned-minus-base difference for a final metric.

    Use scalar values for proportion metrics. For ``argument_key_f1`` and
    ``argument_value_f1``, pass ``(true_positive, predicted, gold)`` counts
    for every record so every draw recomputes the published micro-F1.
    """
    if metric not in {
        "scalar",
        "tool_accuracy",
        "json_validity",
        "schema_validity",
        "exact_match",
        "task_success",
        "argument_key_f1",
        "argument_value_f1",
    }:
        raise ValueError(f"unsupported metric: {metric}")
    if len(base_values) != len(base_source_ids) or len(tuned_values) != len(tuned_source_ids):
        raise ValueError("each value series must have one source ID per value")
    if len(set(base_source_ids)) != len(base_source_ids):
        raise ValueError("base source IDs must be unique")
    if len(set(tuned_source_ids)) != len(tuned_source_ids):
        raise ValueError("tuned source IDs must be unique")
    if set(base_source_ids) != set(tuned_source_ids):
        raise ValueError("base and tuned source IDs must match exactly")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must be between zero and one")
    if not base_values:
        return (0.0, 0.0)

    if metric in {"argument_key_f1", "argument_value_f1"}:
        typed_base = [_micro_counts(value) for value in base_values]
        typed_tuned = [_micro_counts(value) for value in tuned_values]
        counts_by_id = dict(zip(tuned_source_ids, typed_tuned, strict=True))
        paired_counts = [
            (base, counts_by_id[source_id]) for source_id, base in zip(base_source_ids, typed_base)
        ]
    else:
        typed_base_scalars = [_scalar_value(value) for value in base_values]
        typed_tuned_scalars = [_scalar_value(value) for value in tuned_values]
        scalars_by_id = dict(zip(tuned_source_ids, typed_tuned_scalars, strict=True))
        paired_scalars = [
            (base, scalars_by_id[source_id])
            for source_id, base in zip(base_source_ids, typed_base_scalars)
        ]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        if metric in {"argument_key_f1", "argument_value_f1"}:
            selected_counts = [
                paired_counts[rng.randrange(len(paired_counts))] for _ in paired_counts
            ]
            base_counts = [item[0] for item in selected_counts]
            tuned_counts = [item[1] for item in selected_counts]
            base_f1 = _f1(
                sum(item[0] for item in base_counts),
                sum(item[0] for item in base_counts),
                sum(item[1] for item in base_counts),
                sum(item[2] for item in base_counts),
            )
            tuned_f1 = _f1(
                sum(item[0] for item in tuned_counts),
                sum(item[0] for item in tuned_counts),
                sum(item[1] for item in tuned_counts),
                sum(item[2] for item in tuned_counts),
            )
            draws.append(tuned_f1 - base_f1)
        else:
            selected_scalars = [
                paired_scalars[rng.randrange(len(paired_scalars))] for _ in paired_scalars
            ]
            draws.append(
                sum(tuned - base for base, tuned in selected_scalars) / len(selected_scalars)
            )
    return _interval(draws, confidence)


def three_seed_summary(evaluations: Sequence[Evaluation]) -> dict[str, dict[str, float]]:
    """Summarize three independent seeds with sample standard deviation."""
    if len(evaluations) != 3:
        raise ValueError("exactly three seed evaluations are required")
    fields = (
        "tool_accuracy",
        "json_validity",
        "schema_validity",
        "argument_key_f1",
        "argument_value_f1",
        "exact_match",
        "task_success",
    )
    output: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(getattr(item, field)) for item in evaluations]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        output[field] = {"mean": mean, "sample_std": variance**0.5}
    return output


def no_call_accuracy(predictions: Iterable[str]) -> float:
    values = list(predictions)
    if not values:
        return 0.0
    return sum(TOOL_START not in text and TOOL_END not in text for text in values) / len(values)
