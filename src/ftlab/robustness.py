"""Deterministic no-call robustness suite, kept separate from call metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parser import parse_prediction
from .rendering import TOOL_END, TOOL_START


@dataclass(frozen=True)
class NoCallCase:
    case_id: str
    query: str
    expected_no_call: bool = True


@dataclass(frozen=True)
class RobustnessCase:
    case_id: str
    query: str
    tools: tuple[dict[str, Any], ...]
    expected_name: str | None
    distractors: int


CALL_TEMPLATES = (
    "Find the weather in {city}.",
    "Set a reminder to call {person} tomorrow.",
    "Search for flights from {city} to {city2}.",
    "Convert {amount} USD to EUR.",
    "Look up the stock price for {ticker}.",
    "Book a table for {people} at {place}.",
    "Translate {word} into French.",
    "Check whether {service} is available.",
    "Send a message to {person} saying hello.",
    "Find the nearest {place}.",
    "Calculate {amount} plus {amount2}.",
    "Get the population of {city}.",
    "Create a task named {task}.",
    "Delete the draft named {task}.",
    "List events for {city} this weekend.",
    "Get the exchange rate for {ticker}.",
    "Check the status of order {order}.",
    "Find a recipe using {word}.",
    "Open a support ticket for {service}.",
    "Schedule a meeting with {person}.",
)


def robustness_cases() -> list[RobustnessCase]:
    cases: list[RobustnessCase] = []
    values = {
        "city": "Paris",
        "city2": "Delhi",
        "person": "Alex",
        "amount": 3,
        "amount2": 4,
        "ticker": "ACME",
        "people": 2,
        "place": "cafe",
        "word": "cache",
        "service": "weather",
        "task": "weekly review",
        "order": "A-17",
    }
    for template_index, template in enumerate(CALL_TEMPLATES):
        for distractors in (0, 1, 3, 7):
            tools = [{"name": "fixture.target", "parameters": {"type": "object", "properties": {}}}]
            tools.extend(
                {
                    "name": f"fixture.distractor.{index}",
                    "parameters": {"type": "object", "properties": {}},
                }
                for index in range(distractors)
            )
            cases.append(
                RobustnessCase(
                    f"call-{template_index:02d}-{distractors}",
                    template.format(**values),
                    tuple(tools),
                    "fixture.target",
                    distractors,
                )
            )
    return cases


def synthetic_no_call_cases() -> list[NoCallCase]:
    return [
        NoCallCase("no-call-0", "What is the purpose of this test?"),
        NoCallCase("no-call-1", "Respond with a short greeting only."),
        NoCallCase("no-call-2", "Explain the word cache."),
        NoCallCase("no-call-3", "Do not select a tool for this question."),
    ]


def robustness_no_call_cases() -> list[NoCallCase]:
    prompts = [
        "What is the purpose of this test?",
        "Respond with a short greeting only.",
        "Explain the word cache.",
        "Do not select a tool for this question.",
        "Summarize this sentence without taking an action.",
        "What does offline mean?",
        "Write one sentence about rain.",
        "Say thank you.",
        "Explain a queue simply.",
        "Give a harmless example of a list.",
        "What is two plus two?",
        "Define latency.",
        "Describe a blue sky.",
        "Reply with the word okay.",
        "Explain NFC normalization.",
        "What is a JSON object?",
        "Give a short study tip.",
        "Name a fruit.",
        "Describe a quiet room.",
        "Do not call any function.",
    ]
    return [NoCallCase(f"no-call-{index}", prompt) for index, prompt in enumerate(prompts)]


def no_call_accuracy(predictions: list[str]) -> float:
    if len(predictions) != len(synthetic_no_call_cases()):
        raise ValueError("no-call predictions must match the deterministic suite size")
    return sum(
        TOOL_START not in prediction and TOOL_END not in prediction for prediction in predictions
    ) / len(predictions)


def all_robustness_cases() -> list[RobustnessCase | NoCallCase]:
    return [*robustness_cases(), *robustness_no_call_cases()]


def prepare_robustness(path: str | Path) -> Path:
    """Write 80 call cases and 20 separate no-call cases."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in all_robustness_cases():
        if isinstance(case, RobustnessCase):
            rows.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "tools": list(case.tools),
                    "expected_name": case.expected_name,
                    "distractors": case.distractors,
                }
            )
        else:
            rows.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "tools": [],
                    "expected_name": None,
                    "distractors": 0,
                }
            )
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    return target


def evaluate_robustness(predictions: list[str] | dict[str, str]) -> dict[str, Any]:
    cases = all_robustness_cases()
    if isinstance(predictions, list) and len(predictions) != len(cases):
        raise ValueError("robustness predictions must match the 100-case suite size")
    values = (
        {case.case_id: predictions[index] for index, case in enumerate(cases)}
        if isinstance(predictions, list)
        else predictions
    )
    calls = [case for case in cases if isinstance(case, RobustnessCase)]
    no_calls = [case for case in cases if isinstance(case, NoCallCase)]
    successes = 0
    for case in calls:
        parsed = parse_prediction(values.get(case.case_id, ""), case.tools)
        successes += int(parsed.valid and parsed.name == case.expected_name)
    by_distractor: dict[str, list[int]] = {str(level): [] for level in (0, 1, 3, 7)}
    for case in calls:
        parsed = parse_prediction(values.get(case.case_id, ""), case.tools)
        by_distractor[str(case.distractors)].append(
            int(parsed.valid and parsed.name == case.expected_name)
        )
    no_call_values = [values.get(case.case_id, "") for case in no_calls]
    no_call_failures: list[dict[str, str]] = []
    parser_diagnostics: dict[str, int] = {}
    for no_call_case, value in zip(no_calls, no_call_values):
        if TOOL_START not in value and TOOL_END not in value:
            continue
        parsed = parse_prediction(value, [])
        diagnostic = parsed.error.category if parsed.error is not None else "valid"
        parser_diagnostics[diagnostic] = parser_diagnostics.get(diagnostic, 0) + 1
        no_call_failures.append(
            {
                "case_id": no_call_case.case_id,
                "error_category": "should_not_call_tool",
                "parser_diagnostic": diagnostic,
            }
        )
    return {
        "call_count": len(calls),
        "call_success": successes / len(calls),
        "call_success_by_distractors": {
            key: sum(items) / len(items) if items else 0.0 for key, items in by_distractor.items()
        },
        "no_call_count": len(no_calls),
        "no_call_accuracy": sum(
            int(TOOL_START not in value and TOOL_END not in value) for value in no_call_values
        )
        / len(no_call_values),
        "no_call_error_categories": {
            "should_not_call_tool": len(no_call_failures),
        },
        "no_call_failures": no_call_failures,
        "no_call_parser_diagnostics": parser_diagnostics,
        "case_hash": hashlib.sha256("\n".join(case.case_id for case in cases).encode()).hexdigest(),
    }
