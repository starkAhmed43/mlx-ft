from __future__ import annotations

import math

import pytest

from ftlab.storage import write_json


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_write_json_rejects_non_finite_floats(tmp_path, value: float) -> None:
    target = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="non-finite float"):
        write_json(target, {"nested": {"metric": value}})
    assert not target.exists()


def test_write_json_persists_finite_metrics(tmp_path) -> None:
    target = write_json(tmp_path / "metrics.json", {"metric": 1.5})
    assert target.read_text(encoding="utf-8") == '{\n  "metric": 1.5\n}\n'
