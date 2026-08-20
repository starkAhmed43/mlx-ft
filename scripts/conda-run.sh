#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "mlx-ft" || -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate the mlx-ft Conda environment before running ftlab" >&2
  exit 1
fi
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
exec "$@"
