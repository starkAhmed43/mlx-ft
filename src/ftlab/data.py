"""Dataset normalization, filtering, duplicate grouping, and deterministic splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from jsonschema import Draft202012Validator

from .exceptions import ExternalDependencyError

DATASET_ID = "Salesforce/xlam-function-calling-60k"
DATASET_REVISION = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
DATA_FILTER_VERSION = 1
SPLIT_VERSION = 1

# One immutable allocation plan is shared by all published dataset versions.
NESTED_TRAIN_TARGETS = {"smoke": 128, "day1": 2000, "core": 10000, "full": 20000}
LOCKED_VALIDATION_SIZE = 750
LOCKED_TEST_SIZE = 1000


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _validate_json_values(value: Any, field: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string object key")
            _validate_json_values(key, field)
            _validate_json_values(nested, field)
        return
    if isinstance(value, list):
        for nested in value:
            _validate_json_values(nested, field)
        return
    if isinstance(value, tuple):
        raise ValueError(f"{field} contains an unsupported tuple")
    if isinstance(value, set):
        raise ValueError(f"{field} contains an unsupported set")
    if isinstance(value, frozenset):
        raise ValueError(f"{field} contains an unsupported frozenset")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains non-finite number")
        return
    raise ValueError(f"{field} contains unsupported JSON value type: {type(value).__name__}")


def _json_value(value: Any, field: str) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON: {exc.msg}") from exc
    _validate_json_values(value, field)
    return value


def _type_name(value: Any) -> str | None:
    if isinstance(value, list):
        raise ValueError("nullable unions are not supported")
    aliases = {"int": "integer", "float": "number", "list": "array", "dict": "object"}
    if isinstance(value, str):
        return aliases.get(value, value)
    return None


def normalize_schema(schema: Any) -> dict[str, Any]:
    """Normalize a supported Draft 2020-12 schema or raise a useful error."""
    _validate_json_values(schema, "schema")
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    supported = {
        "$schema",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
    }
    unsupported = set(schema) - supported
    if unsupported:
        raise ValueError(f"unsupported schema keyword: {sorted(unsupported)[0]}")
    result: dict[str, Any] = {}
    raw_type = schema.get("type")
    if raw_type == "list" and "items" not in schema:
        raise ValueError("bare list types are not supported")
    if raw_type == "dict" and "properties" not in schema:
        raise ValueError("dict types without properties are not supported")
    schema_type = _type_name(schema.get("type"))
    if schema_type is None:
        if "properties" in schema or "required" in schema:
            schema_type = "object"
        else:
            raise ValueError("schema type is required")
    allowed_types = {"object", "string", "integer", "number", "boolean", "array"}
    if schema_type not in allowed_types:
        raise ValueError(f"unsupported schema type: {schema_type}")
    result["type"] = schema_type
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            raise ValueError("enum must be a non-empty array")
        result["enum"] = enum
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("object properties must be an object")
        result["properties"] = {str(k): normalize_schema(v) for k, v in sorted(properties.items())}
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("required must be an array of strings")
        result["required"] = sorted(set(required))
        additional = schema.get("additionalProperties", True)
        if not isinstance(additional, bool):
            raise ValueError("additionalProperties must be boolean")
        result["additionalProperties"] = additional
    elif schema_type == "array":
        if isinstance(schema.get("items"), list):
            raise ValueError("tuple arrays are not supported")
        if not isinstance(schema.get("items"), dict):
            raise ValueError("homogeneous array items are required")
        result["items"] = normalize_schema(schema["items"])
    return result


_LEGACY_DESCRIPTOR_KEYS = {"description", "type", "default"}


def _is_legacy_parameters(parameters: Any) -> bool:
    if not isinstance(parameters, dict):
        return False
    if not parameters:
        return True
    # Native schemas use these keys at the root. A legacy parameter can still
    # have a name such as ``type`` when its value is a descriptor.
    if "properties" in parameters or "required" in parameters:
        return False
    if "type" in parameters and not isinstance(parameters["type"], dict):
        return False
    return all(isinstance(value, dict) for value in parameters.values())


def _legacy_type_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("legacy parameter type must be a string")
    label = re.sub(r",\s*default\b.*$", "", value.strip(), flags=re.IGNORECASE)
    label = re.sub(r",\s*optional\b", "", label, flags=re.IGNORECASE).strip()
    if not label:
        raise ValueError("legacy parameter type is empty")

    aliases = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }
    if label in aliases:
        return {"type": aliases[label]}
    if label in {"string", "integer", "number", "boolean"}:
        return {"type": label}

    list_match = re.fullmatch(r"(?i:list)\s*\[(.*)\]", label)
    if list_match is not None:
        item_label = list_match.group(1).strip()
        if not item_label:
            raise ValueError("homogeneous list item type is required")
        return {"type": "array", "items": _legacy_type_schema(item_label)}

    base = label.split("[", 1)[0].strip().casefold()
    if base in {"list", "typing.list"}:
        raise ValueError("bare list types are not supported")
    if base in {"union", "optional"} or "|" in label:
        raise ValueError("union types are not supported")
    if base in {"tuple", "typing.tuple"}:
        raise ValueError("tuple types are not supported")
    if base in {"dict", "typing.dict"}:
        raise ValueError("dict types without properties are not supported")
    if base in {"set", "typing.set"}:
        raise ValueError("set types are not supported")
    if base in {"callable", "typing.callable"}:
        raise ValueError("callable types are not supported")
    if base in {"object", "array"}:
        raise ValueError(f"{base} types without properties are not supported")
    raise ValueError(f"unsupported legacy parameter type: {label}")


def _legacy_schema(parameters: dict[Any, Any]) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    for name, descriptor in sorted(parameters.items(), key=lambda item: str(item[0])):
        if not isinstance(descriptor, dict):
            raise ValueError("legacy parameter descriptor must be an object")
        unsupported = set(descriptor) - _LEGACY_DESCRIPTOR_KEYS
        if unsupported:
            raise ValueError(f"unsupported legacy descriptor key: {sorted(unsupported)[0]}")
        if "type" not in descriptor:
            raise ValueError("legacy parameter type is required")
        properties[str(name)] = _legacy_type_schema(descriptor["type"])
    return {
        "type": "object",
        "properties": properties,
        "required": [],
        "additionalProperties": False,
    }


def _normalize_parameters(parameters: Any) -> dict[str, Any]:
    if _is_legacy_parameters(parameters):
        return normalize_schema(_legacy_schema(parameters))
    return normalize_schema(parameters)


def schema_fingerprint(schema: dict[str, Any]) -> str:
    """Hash structure only; descriptions and titles do not affect the result."""
    return hashlib.sha256(canonical_json(normalize_schema(schema)).encode()).hexdigest()


def normalized_api_prefix(name: str) -> str:
    """Return a conservative API prefix for OOD grouping."""
    clean = re.sub(r"[^a-zA-Z0-9]+", ".", name.strip()).strip(".").lower()
    return clean.split(".", 1)[0] if clean else ""


def normalized_query(query: str) -> str:
    """Return the duplicate-group key for a user query."""
    return unicodedata.normalize("NFC", query).strip().casefold()


def _tool_parts(tool: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool.get("function", tool)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("tool function name is required")
    parameters = function.get("parameters", function.get("schema"))
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    return function["name"], _normalize_parameters(parameters)


def _parse_tools(value: Any) -> list[dict[str, Any]]:
    parsed = _json_value(value, "tools")
    if not isinstance(parsed, list):
        raise ValueError("tools must be an array")
    output = []
    names: set[str] = set()
    for tool in parsed:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be an object")
        name, schema = _tool_parts(tool)
        if name in names:
            raise ValueError("candidate function names are not unique")
        names.add(name)
        function = tool.get("function", tool)
        output.append(
            {"name": name, "description": function.get("description", ""), "parameters": schema}
        )
    return sorted(output, key=lambda item: item["name"])


def _parse_answer(value: Any) -> tuple[str, dict[str, Any]]:
    parsed = _json_value(value, "answers")
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError("exactly one gold answer is required")
    answer = parsed[0]
    if not isinstance(answer, dict):
        raise ValueError("gold answer must be an object")
    function = answer.get("function")
    function_name = function.get("name") if isinstance(function, dict) else function
    name = answer.get("name") or function_name
    function_arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
    arguments = answer.get("arguments", function_arguments)
    arguments = _json_value(arguments, "arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("gold function and object arguments are required")
    return name, arguments


@dataclass(frozen=True)
class Rejection:
    source_id: str
    reason: str


@dataclass(frozen=True)
class AcceptedRecord:
    source_id: str
    query: str
    tools: tuple[dict[str, Any], ...]
    function_name: str
    arguments: dict[str, Any]
    normalized_task: str
    candidate_prefixes: frozenset[str]
    candidate_fingerprints: frozenset[str]
    rendered_length: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "source_id": self.source_id,
            "query": self.query,
            "tools": list(self.tools),
            "answer": {"name": self.function_name, "arguments": self.arguments},
        }
        if self.rendered_length is not None:
            value["rendered_length"] = self.rendered_length
        return value


def normalize_record(
    record: dict[str, Any],
    source_id: str | int,
    *,
    max_seq_length: int | None = None,
) -> AcceptedRecord | Rejection:
    """Normalize one xLAM-like record and retain the exact rejection reason."""
    source = str(record.get("id", record.get("source_id", source_id)))
    try:
        query = record.get("query", record.get("prompt", record.get("question")))
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is missing")
        tools = _parse_tools(record.get("tools", record.get("functions")))
        function_name, arguments = _parse_answer(record.get("answers", record.get("answer")))
        candidate = {tool["name"]: tool for tool in tools}
        if function_name not in candidate:
            raise ValueError("gold function is not in candidate tools")
        schema = candidate[function_name]["parameters"]
        errors = sorted(
            Draft202012Validator(schema).iter_errors(arguments), key=lambda item: list(item.path)
        )
        if errors:
            raise ValueError(f"gold arguments fail schema: {errors[0].message}")
        rendered_length = record.get("rendered_length")
        if rendered_length is not None:
            if not isinstance(rendered_length, int):
                raise ValueError("rendered_length must be an integer")
            if max_seq_length is not None and rendered_length > max_seq_length:
                raise ValueError("rendered length exceeds max_seq_length")
        normalized_tools = tuple(tools)
        clean_query = unicodedata.normalize("NFC", query).strip()
        normalized_task = canonical_json({"query": clean_query, "tools": list(normalized_tools)})
        prefixes = frozenset(normalized_api_prefix(tool["name"]) for tool in normalized_tools)
        fingerprints = frozenset(
            schema_fingerprint(tool["parameters"]) for tool in normalized_tools
        )
        return AcceptedRecord(
            source_id=source,
            query=clean_query,
            tools=normalized_tools,
            function_name=function_name,
            arguments=arguments,
            normalized_task=normalized_task,
            candidate_prefixes=frozenset(item for item in prefixes if item),
            candidate_fingerprints=fingerprints,
            rendered_length=rendered_length,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return Rejection(source, str(exc))


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    max_seq_length: int | None = None,
    length_fn: Callable[[AcceptedRecord], int] | None = None,
) -> tuple[list[AcceptedRecord], list[Rejection]]:
    accepted: list[AcceptedRecord] = []
    rejected: list[Rejection] = []
    for index, record in enumerate(records):
        result = normalize_record(
            record, index, max_seq_length=None if length_fn is not None else max_seq_length
        )
        if isinstance(result, AcceptedRecord) and length_fn is not None:
            try:
                length = int(length_fn(result))
                if length > (max_seq_length or length):
                    result = Rejection(result.source_id, "rendered length exceeds max_seq_length")
                else:
                    result = replace(result, rendered_length=length)
            except (TypeError, ValueError) as exc:
                result = Rejection(result.source_id, f"rendered length failed: {exc}")
        if isinstance(result, AcceptedRecord):
            accepted.append(result)
        else:
            rejected.append(result)
    conflicts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in accepted:
        gold = canonical_json({"name": item.function_name, "arguments": item.arguments})
        conflicts[("query", normalized_query(item.query))].add(gold)
        conflicts[("task", item.normalized_task)].add(gold)
    conflicting_keys = {key for key, values in conflicts.items() if len(values) > 1}
    if conflicting_keys:
        kept: list[AcceptedRecord] = []
        for item in accepted:
            if ("query", normalized_query(item.query)) in conflicting_keys or (
                "task",
                item.normalized_task,
            ) in conflicting_keys:
                rejected.append(Rejection(item.source_id, "duplicate-group gold conflict"))
            else:
                kept.append(item)
        accepted = kept
    return accepted, rejected


def context_eligible_records(
    records: Sequence[AcceptedRecord], max_seq_length: int
) -> tuple[list[AcceptedRecord], list[Rejection]]:
    """Return a deterministic context view without changing locked records."""
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    accepted: list[AcceptedRecord] = []
    rejected: list[Rejection] = []
    for record in records:
        if record.rendered_length is not None and record.rendered_length > max_seq_length:
            rejected.append(Rejection(record.source_id, "rendered length exceeds max_seq_length"))
        else:
            accepted.append(record)
    return accepted, rejected


def build_full_allocation(
    records: Sequence[AcceptedRecord],
    locked_validation: Sequence[AcceptedRecord],
    locked_test: Sequence[AcceptedRecord],
    *,
    target: int = 20_000,
    max_seq_length: int = 2048,
    seed: int = 42,
) -> tuple[dict[str, list[AcceptedRecord]], list[AcceptedRecord], list[Rejection]]:
    """Build the exact full train split from a declared context eligibility cap."""
    eligible, rejected = context_eligible_records(records, max_seq_length)
    candidates = _without_locked_components(eligible, [*locked_validation, *locked_test])
    train = stratified_split_records(candidates, {"train": target}, seed=seed)["train"]
    return (
        {"train": train, "validation": list(locked_validation), "test": list(locked_test)},
        eligible,
        rejected,
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        first, second = self.find(left), self.find(right)
        if first != second:
            self.parent[second] = first


def _groups(records: Sequence[AcceptedRecord], *, schema_ood: bool = False) -> list[list[int]]:
    union = _UnionFind(len(records))
    seen: dict[str, int] = {}
    for index, item in enumerate(records):
        keys = (
            (f"prefix:{value}" for value in item.candidate_prefixes)
            if schema_ood
            else (f"query:{normalized_query(item.query)}", f"task:{item.normalized_task}")
        )
        if schema_ood:
            keys = (*keys, *(f"fingerprint:{value}" for value in item.candidate_fingerprints))
        for key in keys:
            if key in seen:
                union.union(index, seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)
    return list(groups.values())


def _rank_groups(
    groups: Sequence[list[int]], records: Sequence[AcceptedRecord], seed: int
) -> list[list[int]]:
    """Order groups by a stable content hash, independent of source order."""
    return sorted(
        (sorted(group, key=lambda index: records[index].source_id) for group in groups),
        key=lambda group: hashlib.sha256(
            f"{seed}:".encode()
            + "\n".join(sorted(records[index].source_id for index in group)).encode()
        ).hexdigest(),
    )


def duplicate_groups(records: Sequence[AcceptedRecord]) -> list[list[int]]:
    return _groups(records)


def schema_ood_groups(records: Sequence[AcceptedRecord]) -> list[list[int]]:
    return _groups(records, schema_ood=True)


def allocate_schema_ood_test(
    records: Sequence[AcceptedRecord],
    locked_train_validation: Sequence[AcceptedRecord],
    target: int = 1000,
    *,
    seed: int = 42,
) -> list[AcceptedRecord]:
    """Select a strict OOD test against an already locked train/validation set."""
    used_ids = {item.source_id for item in locked_train_validation}
    candidates = [item for item in records if item.source_id not in used_ids]
    combined = [*locked_train_validation, *candidates]
    all_groups = schema_ood_groups(combined)
    locked_indexes = set(range(len(locked_train_validation)))
    groups = [
        [index - len(locked_train_validation) for index in group]
        for group in all_groups
        if not set(group).intersection(locked_indexes)
    ]
    groups = _rank_groups(groups, candidates, seed)
    selected: list[AcceptedRecord] = []
    for group in groups:
        if len(selected) + len(group) <= target:
            selected.extend(candidates[index] for index in group)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError(
            f"cannot build strict schema-OOD test of {target}; "
            f"only {len(selected)} records fit outside locked train/validation"
        )
    return selected


def split_records(
    records: Sequence[AcceptedRecord],
    sizes: dict[str, int],
    *,
    seed: int = 42,
    schema_ood_test: bool = False,
) -> dict[str, list[AcceptedRecord]]:
    """Allocate complete duplicate groups with a deterministic seed."""
    if any(size < 0 for size in sizes.values()):
        raise ValueError("split sizes must be non-negative")
    if sum(sizes.values()) > len(records):
        raise ValueError(f"requested {sum(sizes.values())} records, only {len(records)} accepted")
    groups = schema_ood_groups(records) if schema_ood_test else duplicate_groups(records)
    groups = _rank_groups(groups, records, seed)
    output: dict[str, list[AcceptedRecord]] = {name: [] for name in sizes}
    used: set[int] = set()
    for name, target in sizes.items():
        for group in groups:
            if any(index in used for index in group):
                continue
            if len(output[name]) + len(group) <= target:
                output[name].extend(records[index] for index in group)
                used.update(group)
            if len(output[name]) == target:
                break
        if len(output[name]) != target:
            raise ValueError(
                f"cannot allocate strict {name} split of {target}; complete groups prevent exact allocation"
            )
    if schema_ood_test and "test" in output:
        train_valid = [item for name, split in output.items() if name != "test" for item in split]
        prefixes = set().union(*(item.candidate_prefixes for item in train_valid))
        fingerprints = set().union(*(item.candidate_fingerprints for item in train_valid))
        overlap = [
            item.source_id
            for item in output["test"]
            if item.candidate_prefixes & prefixes or item.candidate_fingerprints & fingerprints
        ]
        if overlap:
            raise ValueError(f"schema-OOD test overlaps train or validation: {overlap[:5]}")
    return output


def stratified_split_records(
    records: Sequence[AcceptedRecord],
    sizes: dict[str, int],
    *,
    seed: int = 42,
    schema_ood_test: bool = False,
) -> dict[str, list[AcceptedRecord]]:
    """Split records while retaining the 1, 2-4, and >=5 tool strata."""
    if not records:
        return {name: [] for name in sizes}
    strata = stratify_records(records)
    total = len(records)
    quotas: dict[str, dict[str, int]] = {name: {} for name in sizes}
    for split_name, target in sizes.items():
        exact = {key: target * len(values) / total for key, values in strata.items()}
        base = {key: int(value) for key, value in exact.items()}
        remainder = target - sum(base.values())
        for key in sorted(exact, key=lambda item: (-(exact[item] - base[item]), item))[:remainder]:
            base[key] += 1
        quotas[split_name] = base
    output: dict[str, list[AcceptedRecord]] = {name: [] for name in sizes}
    # Rank global duplicate components once. A component can contain records
    # from different tool-count strata, so never split it per stratum.
    groups = _rank_groups(
        schema_ood_groups(records) if schema_ood_test else duplicate_groups(records), records, seed
    )
    remaining = {name: {stratum: quotas[name][stratum] for stratum in strata} for name in sizes}
    for group in groups:
        counts = Counter(tool_count_stratum(records[index]) for index in group)
        choices = [
            name
            for name in sizes
            if all(count <= remaining[name][stratum] for stratum, count in counts.items())
        ]
        if not choices:
            continue
        name = min(choices, key=lambda candidate: (sum(remaining[candidate].values()), candidate))
        output[name].extend(records[index] for index in group)
        for stratum, count in counts.items():
            remaining[name][stratum] -= count
    if any(remaining[name][stratum] for name in sizes for stratum in strata):
        raise ValueError("cannot allocate strict stratified splits without duplicate leakage")
    for values in output.values():
        values.sort(
            key=lambda item: hashlib.sha256(f"{seed}:{item.source_id}".encode()).hexdigest()
        )
    if schema_ood_test and "test" in output:
        locked = [item for name, values in output.items() if name != "test" for item in values]
        prefixes = set().union(*(item.candidate_prefixes for item in locked))
        fingerprints = set().union(*(item.candidate_fingerprints for item in locked))
        if any(
            item.candidate_prefixes & prefixes or item.candidate_fingerprints & fingerprints
            for item in output["test"]
        ):
            raise ValueError("schema-OOD test overlaps train or validation")
    return output


def _without_locked_components(
    records: Sequence[AcceptedRecord], locked: Sequence[AcceptedRecord]
) -> list[AcceptedRecord]:
    """Remove every duplicate component touching a locked record."""
    combined = [*locked, *records]
    locked_indexes = set(range(len(locked)))
    blocked = {
        index
        for group in duplicate_groups(combined)
        if set(group) & locked_indexes
        for index in group
    }
    return [
        item
        for index, item in enumerate(combined[len(locked) :], len(locked))
        if index not in blocked
    ]


def _nested_subset(
    records: Sequence[AcceptedRecord], target: int, *, seed: int
) -> list[AcceptedRecord]:
    if target == len(records):
        return list(records)
    selected: list[AcceptedRecord] = []
    for group in _rank_groups(duplicate_groups(records), records, seed):
        if len(selected) + len(group) > target:
            continue
        selected.extend(records[index] for index in group)
        if len(selected) == target:
            return selected
    raise ValueError(f"cannot build nested training subset of {target}")


def build_nested_allocation(
    records: Sequence[AcceptedRecord], *, seed: int = 42
) -> dict[str, dict[str, list[AcceptedRecord]]]:
    """Build the frozen train nesting and shared validation/test locks.

    Validation and test are selected once from the complete accepted pool.
    Every training size is a deterministic subset of the full training pool.
    """
    locked = stratified_split_records(
        records,
        {"validation": LOCKED_VALIDATION_SIZE, "test": LOCKED_TEST_SIZE},
        seed=seed,
    )
    locked_values = [*locked["validation"], *locked["test"]]
    candidates = _without_locked_components(records, locked_values)
    full_train = stratified_split_records(
        candidates, {"train": NESTED_TRAIN_TARGETS["full"]}, seed=seed
    )["train"]
    train_versions = {
        name: _nested_subset(full_train, target, seed=seed)
        for name, target in NESTED_TRAIN_TARGETS.items()
    }
    # Smoke/day1 use smaller views of the immutable locks. The core locks are
    # never independently reallocated for an alias.
    validation_views = {
        "full": locked["validation"],
        "core": locked["validation"],
        "day1": _nested_subset(locked["validation"], 250, seed=seed),
        "smoke": _nested_subset(locked["validation"], 32, seed=seed),
    }
    test_views = {
        "full": locked["test"],
        "core": locked["test"],
        "day1": _nested_subset(locked["test"], 250, seed=seed),
        "smoke": _nested_subset(locked["test"], 64, seed=seed),
    }
    output: dict[str, dict[str, list[AcceptedRecord]]] = {}
    for name in NESTED_TRAIN_TARGETS:
        output[name] = {
            "train": train_versions[name],
            "validation": validation_views[name],
            "test": test_views[name],
        }
    alias_targets = {
        "nested_1k": 1000,
        "nested-1k": 1000,
        "nested_5k": 5000,
        "nested-5k": 5000,
        "nested_10k": 10000,
        "nested-10k": 10000,
        "nested_20k": 20000,
        "nested-20k": 20000,
    }
    for alias, target_size in alias_targets.items():
        output[alias] = {
            "train": _nested_subset(full_train, target_size, seed=seed),
            "validation": locked["validation"],
            "test": locked["test"],
        }
    return output


def tool_count_stratum(record: AcceptedRecord) -> str:
    """Return the frozen validation/test stratum by candidate-tool count."""
    count = len(record.tools)
    return "1" if count == 1 else "2-4" if count <= 4 else ">=5"


def stratify_records(records: Sequence[AcceptedRecord]) -> dict[str, list[AcceptedRecord]]:
    """Group records into the required 1, 2-4, and >=5 tool strata."""
    output: dict[str, list[AcceptedRecord]] = {name: [] for name in ("1", "2-4", ">=5")}
    for record in records:
        output[tool_count_stratum(record)].append(record)
    return output


def audit_records(
    records: Sequence[AcceptedRecord], *, version: str, samples: int = 100, seed: int = 42
) -> dict[str, Any]:
    """Produce deterministic aggregate audit data without retaining raw records."""
    if samples < 0:
        raise ValueError("samples must be non-negative")
    ranked = sorted(
        records,
        key=lambda item: hashlib.sha256(f"{seed}:{item.source_id}".encode()).hexdigest(),
    )
    sample = ranked[:samples]
    stratum_counts = Counter(tool_count_stratum(item) for item in records)
    return {
        "version": version,
        "audit_version": 1,
        "seed": seed,
        "sample_count": len(sample),
        "population_count": len(records),
        "sample_id_hash": hashlib.sha256(
            "\n".join(item.source_id for item in sample).encode()
        ).hexdigest(),
        "strata": {name: stratum_counts.get(name, 0) for name in ("1", "2-4", ">=5")},
    }


def audit_queue(
    records: Sequence[AcceptedRecord], *, version: str, samples: int = 100, seed: int = 42
) -> list[dict[str, Any]]:
    """Return the deterministic local inspection queue with full records."""
    if samples < 0:
        raise ValueError("samples must be non-negative")
    ranked = sorted(
        records,
        key=lambda item: hashlib.sha256(f"{seed}:{item.source_id}".encode()).hexdigest(),
    )
    return [
        {"version": version, "rank": rank, "source_id": item.source_id, "record": item.as_dict()}
        for rank, item in enumerate(ranked[:samples])
    ]


def source_flow(
    accepted: Sequence[AcceptedRecord], rejected: Sequence[Rejection]
) -> dict[str, int]:
    reasons = Counter(item.reason for item in rejected)
    result = {
        "source": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
    }
    result.update({f"rejected:{reason}": count for reason, count in reasons.items()})
    return result


def load_xlam(
    *, revision: str = DATASET_REVISION, cache_root: str | None = None
) -> list[dict[str, Any]]:
    """Load the pinned dataset. Never substitute a local or alternate dataset."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ExternalDependencyError(
            "datasets and huggingface_hub are required for xLAM access; install the data extra"
        ) from exc
    try:
        HfApi().whoami()
    except Exception as exc:
        raise ExternalDependencyError(
            "Hugging Face authentication is required for the pinned xLAM dataset"
        ) from exc
    try:
        if cache_root is not None:
            cache = os.path.abspath(cache_root)
            os.makedirs(cache, exist_ok=True)
            os.environ.update(
                {
                    "HF_HOME": cache,
                    "HF_HUB_CACHE": os.path.join(cache, "hub"),
                    "HF_DATASETS_CACHE": os.path.join(cache, "datasets"),
                }
            )
        if cache_root is None:
            raise ExternalDependencyError("a project cache root is required for xLAM access")
        dataset = load_dataset(
            DATASET_ID,
            revision=revision,
            split="train",
            cache_dir=os.path.join(os.path.abspath(cache_root), "huggingface", "datasets"),
        )
    except Exception as exc:
        raise ExternalDependencyError(
            "xLAM access failed; accept the dataset terms and authenticate with Hugging Face"
        ) from exc
    return [dict(row) for row in dataset]
