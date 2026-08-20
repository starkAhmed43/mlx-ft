from __future__ import annotations

import pytest

from ftlab.data import AcceptedRecord, normalize_record
from ftlab.metrics import canonicalize, evaluate_predictions
from ftlab.parser import PRIMARY_TAXONOMY, parse_prediction
from ftlab.rendering import render_json_call


def test_strict_parser_and_metrics(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    good = render_json_call(record.function_name, record.arguments)
    parsed = parse_prediction(good, record.tools)
    assert parsed.valid
    result = evaluate_predictions([record], [good])
    assert result.task_success == 1.0
    assert result.exact_match == 1.0


def test_parser_rejects_multiple_and_partial(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    good = render_json_call(record.function_name, record.arguments)
    assert parse_prediction(good + good, record.tools).error is not None
    assert parse_prediction(good[:-5], record.tools).error is not None
    assert parse_prediction(good + "<|im_end|>", record.tools).valid
    assert parse_prediction(good + "<|im_end|><|im_end|>", record.tools).error is not None


def test_parser_failures_are_false_for_all_metrics(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    result = evaluate_predictions(
        [record], ['<tool_call>{"name":"unknown","arguments":{}}</tool_call>']
    )
    assert result.json_validity == 1.0
    assert result.schema_validity == 0.0
    assert result.tool_accuracy == 0.0
    assert result.argument_key_f1 == 0.0
    assert result.argument_value_f1 == 0.0
    assert result.task_success == 0.0


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parser_rejects_non_finite_json_constants(
    sample_record: dict[str, object], constant: str
) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = (
        '<tool_call>{"name":"weather.get","arguments":{"city":' + constant + "}}</tool_call>"
    )
    parsed = parse_prediction(prediction, record.tools)
    assert parsed.error is not None
    assert parsed.error.category == "invalid_json"
    assert parsed.error.message == f"non-finite JSON number: {constant}"


def test_parser_rejects_nested_duplicate_json_keys(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = (
        '<tool_call>{"name":"weather.get","arguments":'
        '{"nested":{"city":"Paris","city":"Delhi"}}}</tool_call>'
    )
    parsed = parse_prediction(prediction, record.tools)
    assert parsed.error is not None
    assert parsed.error.category == "invalid_json"
    assert parsed.error.message == "duplicate JSON object key: city"


def test_metric_evaluation_counts_strict_json_failure_as_false(
    sample_record: dict[str, object],
) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = (
        '<tool_call>{"name":"weather.get","arguments":{"city":"Paris","city":"Delhi"}}</tool_call>'
    )
    result = evaluate_predictions([record], [prediction])
    assert result.json_validity == 0.0
    assert result.schema_validity == 0.0
    assert result.task_success == 0.0


def test_schema_failure_is_json_valid_but_not_schema_valid(
    sample_record: dict[str, object],
) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = render_json_call("weather.get", {"city": 42})
    result = evaluate_predictions([record], [prediction])
    assert result.json_validity == 1.0
    assert result.tool_accuracy == 1.0
    assert result.schema_validity == 0.0
    assert result.primary_categories["schema_invalid"] == 1


def test_schema_invalid_extra_arguments_are_counted(sample_record: dict[str, object]) -> None:
    tool = sample_record["tools"][0]
    assert isinstance(tool, dict)
    parameters = tool["parameters"]
    assert isinstance(parameters, dict)
    parameters["additionalProperties"] = False
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = (
        '<tool_call>{"name":"weather.get","arguments":{"city":"Paris","extra":1}}</tool_call>'
    )
    result = evaluate_predictions([record], [prediction])
    assert result.json_validity == 1.0
    assert result.schema_validity == 0.0
    assert result.extra_argument_count == 1


def test_primary_taxonomy_is_fixed() -> None:
    assert len(PRIMARY_TAXONOMY) == 10
    assert len(set(PRIMARY_TAXONOMY)) == 10


def test_nested_empty_extra_fails_exact_match(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = (
        '<tool_call>\n{"name":"weather.get","arguments":{"city":"Paris","nested":{}}}\n</tool_call>'
    )
    result = evaluate_predictions([record], [prediction])
    assert result.exact_match == 0.0
    assert result.task_success == 0.0


def test_unicode_trim_and_number_schema_rules(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    assert canonicalize("  Cafe\u0301 ") == "Café"
