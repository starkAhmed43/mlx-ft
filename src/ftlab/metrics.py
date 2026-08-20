"""Function-call metrics with independent validity and correctness fields."""

from __future__ import annotations

import random
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .data import AcceptedRecord
from .parser import PRIMARY_TAXONOMY, ParsedCall, parse_prediction
from .rendering import TOOL_END, TOOL_START


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
    key_scores: list[float] = []
    value_scores: list[float] = []
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
            if parsed.error.category == "schema_invalid":
                predicted_leaves = _leaves(parsed.arguments)
                extra_argument_count += int(set(predicted_leaves) - set(gold_leaves) != set())
            key_scores.append(0.0)
            value_scores.append(0.0)
            series["argument_key_f1"].append(0.0)
            series["argument_value_f1"].append(0.0)
            series["exact_match"].append(0.0)
            series["task_success"].append(0.0)
            continue
        tool_ok = parsed.tool_valid and parsed.name == record.function_name
        predicted_leaves = _leaves(parsed.arguments)
        gold_paths, predicted_paths = set(gold_leaves), set(predicted_leaves)
        key_matches = len(gold_paths & predicted_paths)
        predicted_keys = len(predicted_paths)
        schema: dict[str, Any] = next(
            (tool["parameters"] for tool in record.tools if tool["name"] == record.function_name),
            {},
        )
        value_matches = sum(
            int(_same_value(gold_leaves[path], predicted_leaves[path], _schema_at(schema, path)))
            for path in gold_paths & predicted_paths
        )
        key_f1 = _f1(key_matches, key_matches, predicted_keys, len(gold_paths))
        value_f1 = _f1(value_matches, value_matches, predicted_keys, len(gold_paths))
        key_scores.append(key_f1)
        value_scores.append(value_f1)
        extra_argument_count += int(set(predicted_leaves) - set(gold_leaves) != set())
        exact = tool_ok and _same_argument(record.arguments, parsed.arguments, schema)
        exact_hits += int(exact)
        success_hits += int(exact)
        series["argument_key_f1"].append(key_f1)
        series["argument_value_f1"].append(value_f1)
        series["exact_match"].append(float(exact))
        series["task_success"].append(float(exact))
    count = len(records)
    return Evaluation(
        count=count,
        tool_accuracy=tool_hits / count if count else 0.0,
        json_validity=json_hits / count if count else 0.0,
        schema_validity=schema_hits / count if count else 0.0,
        argument_key_f1=sum(key_scores) / count if count else 0.0,
        argument_value_f1=sum(value_scores) / count if count else 0.0,
        exact_match=exact_hits / count if count else 0.0,
        task_success=success_hits / count if count else 0.0,
        parser_errors=dict(parser_errors),
        primary_categories=dict(primary_categories),
        extra_argument_rate=extra_argument_count / count if count else 0.0,
        extra_argument_count=extra_argument_count,
        confidence_intervals=(
            {name: paired_bootstrap_interval(values) for name, values in series.items()}
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
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = means[max(0, int(alpha * samples))]
    upper = means[min(samples - 1, int((1.0 - alpha) * samples))]
    return lower, upper


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
