from __future__ import annotations

import pytest

from ftlab.robustness import evaluate_robustness, no_call_accuracy, synthetic_no_call_cases


def test_no_call_suite_is_deterministic() -> None:
    cases = synthetic_no_call_cases()
    assert [case.case_id for case in cases] == [f"no-call-{index}" for index in range(4)]
    assert no_call_accuracy(["Normal assistant response."] * len(cases)) == 1.0
    assert no_call_accuracy(["<tool_call>"] + ["Normal"] * 3) == 0.75


def test_no_call_suite_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        no_call_accuracy([])


def test_no_call_tool_attempt_uses_task_taxonomy_and_keeps_parser_diagnostic() -> None:
    predictions = ['<tool_call>{"name":"fixture.target","arguments":{}}</tool_call>'] * 80
    predictions += ["<tool_call>{}</tool_call>"] + ["No action."] * 19
    result = evaluate_robustness(predictions)
    assert result["no_call_error_categories"] == {"should_not_call_tool": 1}
    failure = result["no_call_failures"][0]
    assert failure["case_id"] == "no-call-0"
    assert failure["error_category"] == "should_not_call_tool"
    assert failure["parser_diagnostic"] == "invalid_call"
