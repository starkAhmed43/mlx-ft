from __future__ import annotations

import pytest

from ftlab.robustness import no_call_accuracy, synthetic_no_call_cases


def test_no_call_suite_is_deterministic() -> None:
    cases = synthetic_no_call_cases()
    assert [case.case_id for case in cases] == [f"no-call-{index}" for index in range(4)]
    assert no_call_accuracy(["Normal assistant response."] * len(cases)) == 1.0
    assert no_call_accuracy(["<tool_call>"] + ["Normal"] * 3) == 0.75


def test_no_call_suite_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        no_call_accuracy([])
