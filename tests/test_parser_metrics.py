from __future__ import annotations

import pytest

from ftlab.data import AcceptedRecord, normalize_record
from ftlab.metrics import (
    TASK_ERROR_TAXONOMY,
    bootstrap_micro_f1_interval,
    canonicalize,
    evaluate_predictions,
    paired_bootstrap_delta_interval,
    paired_bootstrap_metric_delta_interval,
)
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
    assert result.task_success == 1.0
    assert result.extra_argument_count == 1


def test_unicode_trim_and_number_schema_rules(sample_record: dict[str, object]) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    assert canonicalize("  Cafe\u0301 ") == "Café"


def test_argument_f1_is_micro_aggregated(sample_record: dict[str, object]) -> None:
    first = normalize_record(sample_record, "first")
    second_input = {
        **sample_record,
        "id": "second",
        "answers": [{"name": "weather.get", "arguments": {"city": "Delhi", "days": 2}}],
    }
    second = normalize_record(second_input, "second")
    assert isinstance(first, AcceptedRecord)
    assert isinstance(second, AcceptedRecord)
    predictions = [
        render_json_call("weather.get", {"city": "Paris"}),
        render_json_call("weather.get", {"city": "Delhi"}),
    ]
    result = evaluate_predictions([first, second], predictions, confidence=False)
    assert result.argument_key_f1 == pytest.approx(0.8)
    assert result.argument_value_f1 == pytest.approx(0.8)


def test_task_success_uses_required_gold_arguments_not_strict_object_match(
    sample_record: dict[str, object],
) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = render_json_call("weather.get", {"city": "Paris", "days": 7})
    result = evaluate_predictions([record], [prediction], confidence=False)
    assert result.schema_validity == 1.0
    assert result.task_success == 1.0
    assert result.exact_match == 0.0
    assert result.task_error_categories["wrong_enum_or_value"] == 0


@pytest.mark.parametrize(
    ("prediction", "category"),
    [
        ("plain text", "no_parseable_tool_call_boundary"),
        ('<tool_call>{"name":}</tool_call>', "invalid_json"),
        ('<tool_call>{"name":"unknown","arguments":{}}</tool_call>', "nonexistent_tool"),
        (render_json_call("weather.get", {}), "missing_required_argument"),
        (render_json_call("weather.get", {"city": 1}), "wrong_type"),
        (render_json_call("weather.get", {"city": "Delhi"}), "wrong_enum_or_value"),
        ('<tool_call>{"name":"weather.get","arguments":{"city":"Paris"}}', "truncated_output"),
    ],
)
def test_task_error_taxonomy_maps_first_applicable_category(
    sample_record: dict[str, object], prediction: str, category: str
) -> None:
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    result = evaluate_predictions([record], [prediction], confidence=False)
    assert tuple(result.as_dict()["task_error_categories"]) == TASK_ERROR_TAXONOMY
    assert result.task_error_categories[category] == 1


def test_extra_argument_has_its_own_task_error_category(sample_record: dict[str, object]) -> None:
    tool = sample_record["tools"][0]
    assert isinstance(tool, dict)
    parameters = tool["parameters"]
    assert isinstance(parameters, dict)
    parameters["additionalProperties"] = False
    record = normalize_record(sample_record, "x")
    assert isinstance(record, AcceptedRecord)
    prediction = render_json_call("weather.get", {"city": "Paris", "extra": True})
    result = evaluate_predictions([record], [prediction], confidence=False)
    assert result.task_error_categories["extra_hallucinated_argument"] == 1
    assert result.extra_argument_count == 1


def test_wrong_supplied_tool_is_distinct_from_nonexistent_tool(
    sample_record: dict[str, object],
) -> None:
    tools = sample_record["tools"]
    assert isinstance(tools, list)
    other = {**tools[0], "name": "weather.forecast"}
    record_input = {**sample_record, "tools": [tools[0], other]}
    record = normalize_record(record_input, "x")
    assert isinstance(record, AcceptedRecord)
    result = evaluate_predictions(
        [record], [render_json_call("weather.forecast", {"city": "Paris"})], confidence=False
    )
    assert result.task_error_categories["wrong_supplied_tool"] == 1


def test_paired_delta_interval_aligns_by_source_id_and_rejects_bad_pairs() -> None:
    interval = paired_bootstrap_delta_interval(
        [0.0, 1.0],
        [1.0, 1.0],
        base_source_ids=["a", "b"],
        tuned_source_ids=["b", "a"],
        samples=100,
    )
    assert interval[0] >= 0.0
    assert interval[1] <= 1.0
    with pytest.raises(ValueError, match="unique"):
        paired_bootstrap_delta_interval(
            [0.0, 1.0], [1.0, 1.0], base_source_ids=["a", "a"], tuned_source_ids=["a", "b"]
        )
    with pytest.raises(ValueError, match="match exactly"):
        paired_bootstrap_delta_interval([0.0], [1.0], base_source_ids=["a"], tuned_source_ids=["b"])


def test_micro_f1_bootstrap_recomputes_counts_not_per_example_means() -> None:
    counts = [(10, 10, 10), (0, 0, 1)]
    interval = bootstrap_micro_f1_interval(counts, seed=4, samples=1)
    assert interval == pytest.approx((20 / 21, 20 / 21))
    assert interval[0] != pytest.approx(0.5)


def test_paired_metric_delta_recomputes_aligned_micro_f1() -> None:
    interval = paired_bootstrap_metric_delta_interval(
        [(0, 0, 10), (0, 0, 1)],
        [(10, 10, 10), (0, 0, 1)],
        metric="argument_value_f1",
        base_source_ids=["a", "b"],
        tuned_source_ids=["b", "a"],
        seed=4,
        samples=1,
    )
    assert interval == pytest.approx((20 / 21, 20 / 21))
    with pytest.raises(ValueError, match="micro-F1"):
        paired_bootstrap_metric_delta_interval(
            [0.0],
            [1.0],
            metric="argument_key_f1",
            base_source_ids=["a"],
            tuned_source_ids=["a"],
        )
