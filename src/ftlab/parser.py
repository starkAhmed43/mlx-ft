"""Strict parser for the Qwen single-tool-call output contract."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .rendering import TOOL_END, TOOL_START

PRIMARY_TAXONOMY = (
    "valid",
    "no_call",
    "thinking",
    "truncated",
    "multiple_calls",
    "extra_output",
    "invalid_json",
    "invalid_call",
    "unknown_tool",
    "schema_invalid",
)


@dataclass(frozen=True)
class ParseError:
    category: str
    message: str


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]
    raw: str
    error: ParseError | None = None

    @property
    def valid(self) -> bool:
        return self.error is None

    @property
    def json_valid(self) -> bool:
        return self.error is None or self.error.category in {
            "json_fragment",
            "non_object_arguments",
            "invalid_call",
            "unknown_tool",
            "schema_invalid",
        }

    @property
    def tool_valid(self) -> bool:
        """Whether the payload names a supplied tool, independent of schema."""
        return self.error is None or self.error.category == "schema_invalid"

    @property
    def schema_valid(self) -> bool:
        return self.error is None

    @property
    def primary_category(self) -> str:
        if self.error is None:
            return "valid"
        mapping = {
            "non_text": "extra_output",
            "thinking_block": "thinking",
            "partial_boundary": "truncated",
            "truncation": "truncated",
            "no_call": "no_call",
            "multiple_calls": "multiple_calls",
            "extra_output": "extra_output",
            "invalid_json": "invalid_json",
            "json_fragment": "invalid_call",
            "non_object_arguments": "invalid_call",
            "invalid_call": "invalid_call",
            "unknown_tool": "unknown_tool",
            "schema_invalid": "schema_invalid",
        }
        return mapping.get(self.error.category, "extra_output")


_BOUNDARY = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _tool_names(tools: Iterable[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _tool_schema(tools: Iterable[Any], name: str) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(function, dict) and function.get("name") == name:
            schema = function.get("parameters", function.get("schema"))
            return schema if isinstance(schema, dict) else None
    return None


def parse_prediction(
    text: str,
    tools: Iterable[Any],
    *,
    generation_truncated: bool = False,
    max_generation_tokens: int | None = None,
    generated_tokens: int | None = None,
) -> ParsedCall:
    """Parse one complete boundary and reject every recovery path."""
    tool_list = list(tools)
    if not isinstance(text, str):
        return ParsedCall("", {}, "", ParseError("non_text", "prediction is not text"))
    if generation_truncated or (
        max_generation_tokens is not None
        and generated_tokens is not None
        and generated_tokens >= max_generation_tokens
    ):
        return ParsedCall(
            "", {}, text, ParseError("truncation", "generation reached its length limit")
        )
    if "<think>" in text or "</think>" in text:
        return ParsedCall(
            "", {}, text, ParseError("thinking_block", "thinking blocks are not allowed")
        )
    count_start = text.count(TOOL_START)
    count_end = text.count(TOOL_END)
    if count_start != 1 or count_end != 1:
        if count_start > 1 or count_end > 1:
            category = "multiple_calls"
        elif count_start == 0 and count_end == 0:
            category = "no_call"
        else:
            category = "partial_boundary"
        return ParsedCall(
            "", {}, text, ParseError(category, "one complete tool boundary is required")
        )
    clean_text = text.strip()
    if clean_text.endswith("<|im_end|>"):
        clean_text = clean_text[: -len("<|im_end|>")].rstrip()
        if "<|im_end|>" in clean_text:
            return ParsedCall(
                "", {}, text, ParseError("extra_output", "multiple template terminators")
            )
    match = _BOUNDARY.fullmatch(clean_text)
    if match is None:
        return ParsedCall(
            "",
            {},
            text,
            ParseError("extra_output", "output outside the tool boundary is not allowed"),
        )
    payload = match.group(1).strip()
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        return ParsedCall("", {}, text, ParseError("invalid_json", exc.msg))
    except ValueError as exc:
        return ParsedCall("", {}, text, ParseError("invalid_json", str(exc)))
    if not isinstance(value, dict):
        return ParsedCall(
            "", {}, text, ParseError("json_fragment", "tool call must be a JSON object")
        )
    name = value.get("name")
    arguments = value.get("arguments")
    if set(value) != {"name", "arguments"}:
        return ParsedCall(
            str(name or ""),
            arguments if isinstance(arguments, dict) else {},
            text,
            ParseError("invalid_call", "call must contain only name and arguments"),
        )
    if not isinstance(name, str) or not name:
        return ParsedCall("", {}, text, ParseError("invalid_call", "tool name must be a string"))
    if not isinstance(arguments, dict):
        return ParsedCall(
            name, {}, text, ParseError("non_object_arguments", "arguments must be an object")
        )
    if name not in _tool_names(tool_list):
        return ParsedCall(
            name, arguments, text, ParseError("unknown_tool", f"unknown tool: {name}")
        )
    schema = _tool_schema(tool_list, name)
    if schema is not None:
        errors = list(Draft202012Validator(schema).iter_errors(arguments))
        if errors:
            return ParsedCall(
                name, arguments, text, ParseError("schema_invalid", errors[0].message)
            )
    return ParsedCall(name, arguments, text)
