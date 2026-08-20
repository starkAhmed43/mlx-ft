from __future__ import annotations

import os
import sys

import pytest

from ftlab.environment import configure_uv_project_environment, validate_conda_environment


def test_conda_environment_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "mlx-ft")
    monkeypatch.setenv("CONDA_PREFIX", sys.prefix)
    status = validate_conda_environment()
    assert status.name == "mlx-ft"


def test_conda_environment_rejects_wrong_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setenv("CONDA_PREFIX", sys.prefix)
    with pytest.raises(RuntimeError, match="CONDA_DEFAULT_ENV"):
        validate_conda_environment()


def test_uv_environment_uses_conda_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "mlx-ft")
    monkeypatch.setenv("CONDA_PREFIX", sys.prefix)
    configure_uv_project_environment()
    assert os.environ["UV_PROJECT_ENVIRONMENT"] == sys.prefix
