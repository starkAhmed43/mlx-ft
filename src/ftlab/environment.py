"""Conda environment contract used by every reproducible project command."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ENVIRONMENT_NAME = "mlx-ft"
PYTHON_VERSION = (3, 14, 6)


@dataclass(frozen=True)
class EnvironmentStatus:
    name: str
    prefix: str
    python: tuple[int, int, int]
    uv_project_environment: str


def validate_conda_environment() -> EnvironmentStatus:
    """Validate the exact Conda interpreter required by the project."""
    name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = os.environ.get("CONDA_PREFIX")
    if name != ENVIRONMENT_NAME:
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be {ENVIRONMENT_NAME!r}")
    if not prefix:
        raise RuntimeError("CONDA_PREFIX must be set")
    if os.path.realpath(prefix) != os.path.realpath(sys.prefix):
        raise RuntimeError("CONDA_PREFIX must equal sys.prefix")
    if sys.version_info[:3] != PYTHON_VERSION:
        actual = ".".join(str(value) for value in sys.version_info[:3])
        expected = ".".join(str(value) for value in PYTHON_VERSION)
        raise RuntimeError(f"Python {expected} is required (found {actual})")
    return EnvironmentStatus(
        name=name,
        prefix=prefix,
        python=sys.version_info[:3],
        uv_project_environment=os.environ.get("UV_PROJECT_ENVIRONMENT", ""),
    )


def configure_uv_project_environment() -> str:
    """Set uv's project environment to the active Conda prefix."""
    status = validate_conda_environment()
    os.environ["UV_PROJECT_ENVIRONMENT"] = status.prefix
    return status.prefix
