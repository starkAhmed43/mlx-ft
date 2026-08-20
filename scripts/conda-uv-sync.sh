#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "mlx-ft" ]]; then
  echo "activate the mlx-ft Conda environment before syncing" >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is required" >&2
  exit 1
fi
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
exec uv sync --locked --extra apple --extra data --extra tracking --extra reporting --group dev
