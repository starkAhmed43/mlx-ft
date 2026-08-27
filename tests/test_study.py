from __future__ import annotations

import pytest

from ftlab.study import candidate_key, create_final_lock, nondominated


def test_candidate_key_uses_published_tie_break_order() -> None:
    fast = {
        "fingerprint": "z",
        "task_success": 1.0,
        "argument_value_f1": 1.0,
        "schema_validity": 1.0,
        "rss_gib": 2.0,
        "generation_throughput_tokens_per_second": 20.0,
    }
    slow = {**fast, "fingerprint": "a", "generation_throughput_tokens_per_second": 10.0}
    assert candidate_key(fast) > candidate_key(slow)


def test_nondominated_excludes_strictly_worse_candidate() -> None:
    best = {
        "run_id": "best",
        "task_success": 1.0,
        "argument_value_f1": 1.0,
        "schema_validity": 1.0,
        "rss_gib": 1.0,
    }
    worse = {
        "run_id": "worse",
        "task_success": 0.5,
        "argument_value_f1": 0.5,
        "schema_validity": 0.5,
        "rss_gib": 2.0,
    }
    assert nondominated([best, worse]) == [best]


def test_final_lock_never_treats_missing_controlled_flag_as_controlled(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("ftlab.study._clean_tree", lambda _: True)
    with pytest.raises(ValueError, match="no controlled"):
        create_final_lock(tmp_path, [{"run_id": "unproven", "status": "completed"}])
