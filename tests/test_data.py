from __future__ import annotations

import pytest

from ftlab.data import (
    DATA_FILTER_VERSION,
    SPLIT_VERSION,
    AcceptedRecord,
    Rejection,
    allocate_schema_ood_test,
    audit_records,
    build_nested_allocation,
    context_eligible_records,
    dataset_manifest_v2,
    normalize_record,
    normalize_schema,
    normalized_query,
    schema_fingerprint,
    split_records,
    stratified_split_records,
    validate_dataset_manifest_v2,
)


def test_normalize_fixture_and_schema(sample_record: dict[str, object]) -> None:
    result = normalize_record(sample_record, "x")
    assert isinstance(result, AcceptedRecord)
    assert result.query == "Find the weather."
    assert result.tools[0]["parameters"]["type"] == "object"


def _legacy_record(
    parameters: dict[str, object], arguments: dict[str, object]
) -> dict[str, object]:
    return {
        "id": "legacy",
        "query": "Call the tool.",
        "tools": [{"name": "legacy.tool", "parameters": parameters}],
        "answers": [{"name": "legacy.tool", "arguments": arguments}],
    }


def test_legacy_empty_parameters_are_strict_optional_object() -> None:
    result = normalize_record(_legacy_record({}, {}), "legacy")
    assert isinstance(result, AcceptedRecord)
    assert result.tools[0]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_legacy_scalar_aliases_are_sorted() -> None:
    parameters = {
        "flag": {"type": "bool"},
        "ratio": {"type": "float"},
        "count": {"type": "int"},
        "name": {"type": "str"},
    }
    result = normalize_record(_legacy_record(parameters, {}), "legacy")
    assert isinstance(result, AcceptedRecord)
    properties = result.tools[0]["parameters"]["properties"]
    assert list(properties) == ["count", "flag", "name", "ratio"]
    assert [properties[key]["type"] for key in properties] == [
        "integer",
        "boolean",
        "string",
        "number",
    ]


def test_legacy_optional_defaults_and_nested_lists_allow_omission() -> None:
    parameters = {
        "labels": {"type": "List[List[str]], optional, default: []"},
        "limit": {"type": "int, optional", "default": 10},
    }
    result = normalize_record(_legacy_record(parameters, {}), "legacy")
    assert isinstance(result, AcceptedRecord)
    properties = result.tools[0]["parameters"]["properties"]
    assert properties["limit"] == {"type": "integer"}
    assert properties["labels"] == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "string"}},
    }


def test_legacy_gold_arguments_are_type_checked() -> None:
    result = normalize_record(
        _legacy_record({"count": {"type": "int"}}, {"count": "one"}), "legacy"
    )
    assert isinstance(result, Rejection)
    assert result.reason == "gold arguments fail schema: 'one' is not of type 'integer'"


def test_legacy_gold_extra_keys_are_rejected() -> None:
    result = normalize_record(
        _legacy_record({"count": {"type": "int"}}, {"count": 1, "extra": 2}), "legacy"
    )
    assert isinstance(result, Rejection)
    assert result.reason.startswith("gold arguments fail schema: Additional properties")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_text_rejects_non_finite_constants(constant: str) -> None:
    record = _legacy_record({"value": {"type": "float"}}, {})
    record["answers"] = '[{"name":"legacy.tool","arguments":{"nested":{"value":' + constant + "}}}]"
    result = normalize_record(record, "legacy")
    assert isinstance(result, Rejection)
    assert result.reason == f"non-finite JSON number: {constant}"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_python_objects_reject_non_finite_numbers(value: float) -> None:
    result = normalize_record(
        _legacy_record({"value": {"type": "float"}}, {"value": {"nested": value}}),
        "legacy",
    )
    assert isinstance(result, Rejection)
    assert result.reason == "answers contains non-finite number"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ((1,), "answers contains an unsupported tuple"),
        ({1}, "answers contains an unsupported set"),
        (frozenset({1}), "answers contains an unsupported frozenset"),
        (object(), "answers contains unsupported JSON value type: object"),
        ({1: "one"}, "answers contains a non-string object key"),
    ],
)
def test_nested_python_values_must_be_json_compatible(value: object, reason: str) -> None:
    result = normalize_record(
        _legacy_record({"value": {"type": "str"}}, {"value": {"nested": value}}),
        "legacy",
    )
    assert isinstance(result, Rejection)
    assert result.reason == reason


def test_nested_duplicate_json_keys_are_rejected() -> None:
    record = _legacy_record({"value": {"type": "int"}}, {})
    record["answers"] = '[{"name":"legacy.tool","arguments":{"nested":{"value":1,"value":2}}}]'
    result = normalize_record(record, "legacy")
    assert isinstance(result, Rejection)
    assert result.reason == "duplicate JSON object key: value"


@pytest.mark.parametrize("answers", [[], [{"name": "legacy.tool", "arguments": {}}] * 2])
def test_gold_answer_count_must_be_exactly_one(answers: list[dict[str, object]]) -> None:
    record = _legacy_record({}, {})
    record["answers"] = answers
    result = normalize_record(record, "legacy")
    assert isinstance(result, Rejection)
    assert result.reason == "exactly one gold answer is required"


def test_native_schema_is_preserved_through_record_normalization() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer", "enum": [1, 2]}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    result = normalize_record(_legacy_record(schema, {"items": [{"id": 1}]}), "native")
    assert isinstance(result, AcceptedRecord)
    assert result.tools[0]["parameters"] == schema


@pytest.mark.parametrize(
    "bad",
    [
        {"$ref": "#/x"},
        {"type": ["string", "null"]},
        {"type": "array", "items": [{"type": "string"}]},
        {"allOf": [{"type": "string"}]},
        {"type": "string", "pattern": "x"},
        {"type": "string", "format": "date-time"},
    ],
)
def test_schema_rejections(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        normalize_schema(bad)


@pytest.mark.parametrize(
    "type_label",
    ["list", "List", "Union[str, int]", "Tuple[int, int]", "Dict[str, int]", "set", "Callable"],
)
def test_legacy_unsupported_type_families_are_rejected(type_label: str) -> None:
    result = normalize_record(_legacy_record({"value": {"type": type_label}}, {}), "legacy")
    assert isinstance(result, Rejection)
    assert result.reason


def test_description_does_not_change_schema_fingerprint() -> None:
    first = {"type": "string", "description": "one"}
    second = {"type": "string", "description": "two"}
    assert schema_fingerprint(first) == schema_fingerprint(second)


def test_additional_properties_false_is_preserved() -> None:
    normalized = normalize_schema({"type": "object", "additionalProperties": False})
    assert normalized["additionalProperties"] is False


def test_duplicate_gold_conflict_is_rejected(sample_record: dict[str, object]) -> None:
    second = dict(sample_record)
    second["id"] = "fixture-2"
    second["answers"] = [{"name": "weather.get", "arguments": {"city": "Delhi"}}]
    from ftlab.data import normalize_records

    accepted, rejected = normalize_records([sample_record, second])
    assert not accepted
    assert all("duplicate-group gold conflict" in item.reason for item in rejected)


def test_split_is_deterministic(sample_record: dict[str, object]) -> None:
    records = []
    for index in range(4):
        value = dict(sample_record)
        value["id"] = str(index)
        value["query"] = f"q{index}"
        result = normalize_record(value, index)
        assert isinstance(result, AcceptedRecord)
        records.append(result)
    first = split_records(records, {"train": 2, "test": 2})
    second = split_records(records, {"train": 2, "test": 2})
    assert [[item.source_id for item in first[name]] for name in first] == [
        [item.source_id for item in second[name]] for name in second
    ]


def test_duplicate_query_key_is_nfc_trimmed_and_casefolded(
    sample_record: dict[str, object],
) -> None:
    assert normalized_query("  CAFE\u0301 ") == normalized_query("café")


def test_context_eligibility_rejects_long_records_at_short_context(
    sample_record: dict[str, object],
) -> None:
    result = normalize_record({**sample_record, "rendered_length": 700}, "long")
    assert isinstance(result, AcceptedRecord)
    short, short_rejected = context_eligible_records([result], 512)
    long, long_rejected = context_eligible_records([result], 1024)
    assert short == []
    assert [item.source_id for item in short_rejected] == ["fixture-1"]
    assert [item.source_id for item in long] == ["fixture-1"]
    assert long_rejected == []


def test_hash_rank_split_is_independent_of_source_order(sample_record: dict[str, object]) -> None:
    records = []
    for index in range(6):
        value = dict(sample_record)
        value["id"] = str(index)
        value["query"] = f"q{index}"
        result = normalize_record(value, index)
        assert isinstance(result, AcceptedRecord)
        records.append(result)
    first = split_records(records, {"train": 3, "test": 3})
    second = split_records(list(reversed(records)), {"train": 3, "test": 3})
    assert {name: sorted(item.source_id for item in values) for name, values in first.items()} == {
        name: sorted(item.source_id for item in values) for name, values in second.items()
    }


def test_stratified_split_and_audit_are_deterministic(sample_record: dict[str, object]) -> None:
    records = []
    for index, tool_count in enumerate((1, 2, 5) * 2):
        value = dict(sample_record)
        value["id"] = str(index)
        value["query"] = f"q{index}"
        value["tools"] = [
            {"name": f"fixture.{tool_index}", "parameters": {"type": "object", "properties": {}}}
            for tool_index in range(tool_count)
        ]
        value["answers"] = [{"name": "fixture.0", "arguments": {}}]
        result = normalize_record(value, index)
        assert isinstance(result, AcceptedRecord)
        records.append(result)
    split = stratified_split_records(records, {"validation": 3, "test": 3})
    assert {len(values) for values in split.values()} == {3}
    audit = audit_records(records, version="fixture", samples=3)
    assert audit["sample_count"] == 3
    assert "source_id" not in audit


def test_stratified_split_keeps_duplicate_component_across_strata(
    sample_record: dict[str, object],
) -> None:
    records = []
    for index, (query, tool_count) in enumerate(
        (
            ("crossing", 1),
            ("crossing", 2),
            ("single-one", 1),
            ("single-two", 2),
            ("five-a", 5),
            ("five-b", 5),
        )
    ):
        value = dict(sample_record)
        value["id"] = str(index)
        value["query"] = query
        value["tools"] = [
            {"name": f"fixture.{tool_index}", "parameters": {"type": "object", "properties": {}}}
            for tool_index in range(tool_count)
        ]
        value["answers"] = [{"name": "fixture.0", "arguments": {}}]
        result = normalize_record(value, index)
        assert isinstance(result, AcceptedRecord)
        records.append(result)
    split = stratified_split_records(records, {"train": 3, "test": 3})
    locations = {
        item.source_id: split_name for split_name, values in split.items() for item in values
    }
    assert locations["0"] == locations["1"]


def test_schema_ood_discards_transitive_component(sample_record: dict[str, object]) -> None:
    def make(source_id: str, query: str, name: str, schema: dict[str, object]) -> AcceptedRecord:
        value = {
            "id": source_id,
            "query": query,
            "tools": [{"name": name, "parameters": schema}],
            "answers": [{"name": name, "arguments": {}}],
        }
        result = normalize_record(value, source_id)
        assert isinstance(result, AcceptedRecord)
        return result

    schema_a = {"type": "object", "properties": {"a": {"type": "string"}}}
    schema_b = {"type": "object", "properties": {"b": {"type": "string"}}}
    first = make("first", "q1", "alpha.one", schema_a)
    middle = make("middle", "q2", "alpha.two", schema_b)
    last = make("last", "q3", "beta.one", schema_b)
    with pytest.raises(ValueError, match="cannot build strict"):
        allocate_schema_ood_test([first, middle, last], [first], target=1)


def _allocation_records(count: int, *, long_from: int | None = None) -> list[AcceptedRecord]:
    records: list[AcceptedRecord] = []
    for index in range(count):
        long = long_from is not None and index >= long_from
        schema = {"type": "object", "properties": {f"p{index}": {"type": "string"}}}
        records.append(
            AcceptedRecord(
                source_id=str(index),
                query=f"query {index}",
                tools=({"name": f"api{index}.call", "parameters": schema},),
                function_name=f"api{index}.call",
                arguments={},
                normalized_task=str(index),
                candidate_prefixes=frozenset({f"api{index}"}),
                candidate_fingerprints=frozenset({str(index)}),
                rendered_length=1024 if long else 256,
            )
        )
    return records


def test_allocation_uses_long_records_only_for_full_endpoint() -> None:
    short = _allocation_records(12_750)
    long = [*short, *_allocation_records(10_000, long_from=0)]
    # Give extension records distinct identities and group keys.
    long = [
        record
        if index < len(short)
        else AcceptedRecord(
            source_id=f"long-{index}",
            query=f"long query {index}",
            tools=record.tools,
            function_name=record.function_name,
            arguments=record.arguments,
            normalized_task=f"long-{index}",
            candidate_prefixes=frozenset({f"longapi{index}"}),
            candidate_fingerprints=frozenset({f"long-{index}"}),
            rendered_length=1024,
        )
        for index, record in enumerate(long)
    ]
    allocation = build_nested_allocation(short, long_records=long)
    assert {name: len(allocation[name]["train"]) for name in ("smoke", "day1", "core", "full")} == {
        "smoke": 128,
        "day1": 2000,
        "core": 10000,
        "full": 20000,
    }
    core_ids = {item.source_id for item in allocation["core"]["train"]}
    full_ids = {item.source_id for item in allocation["full"]["train"]}
    assert core_ids <= full_ids
    assert all(item.rendered_length <= 512 for item in allocation["core"]["train"])
    assert any(item.rendered_length > 512 for item in allocation["full"]["train"])
    ood_prefixes = set().union(
        *(item.candidate_prefixes for item in allocation["schema_ood"]["test"])
    )
    selected = [item for name in ("core", "full") for item in allocation[name]["train"]]
    assert not any(item.candidate_prefixes & ood_prefixes for item in selected)
    repeat = build_nested_allocation(short, long_records=long)
    assert [item.source_id for item in allocation["full"]["train"]] == [
        item.source_id for item in repeat["full"]["train"]
    ]


def test_allocation_failure_reports_token_tier_counts() -> None:
    with pytest.raises(ValueError, match=r"eligible_512=100, eligible_2048=100"):
        build_nested_allocation(_allocation_records(100))


def test_short_allocation_does_not_require_full_extension() -> None:
    short = _allocation_records(12_750)
    allocation = build_nested_allocation(short, require_full=False)
    assert {name: len(allocation[name]["train"]) for name in ("smoke", "day1", "core")} == {
        "smoke": 128,
        "day1": 2000,
        "core": 10000,
    }
    assert len(allocation["schema_ood"]["test"]) == 1000
    assert len(allocation["nested_10k"]["train"]) == 10000
    assert "full" not in allocation
    with pytest.raises(ValueError, match=r"2048-token full train extension.*eligible_512=12750"):
        build_nested_allocation(short)


def test_dataset_manifest_v2_is_sanitized_and_content_addressed() -> None:
    records = _allocation_records(2)
    manifest = dataset_manifest_v2(
        version="fixture",
        allocation={"train": records},
        eligibility_counts={"eligible_512": 2, "eligible_2048": 2},
        source_descriptors=[{"dataset": "fixture", "revision": "abc", "kind": "test"}],
        prompt_contract_version=1,
    )
    validate_dataset_manifest_v2(manifest)
    assert (manifest["filter_version"], manifest["split_version"]) == (2, 2)
    assert (DATA_FILTER_VERSION, SPLIT_VERSION) == (2, 2)
    assert "query" not in str(manifest)
    manifest["split_locks"]["train"]["count"] = 3
    with pytest.raises(ValueError, match="content hash"):
        validate_dataset_manifest_v2(manifest)
