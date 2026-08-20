.DEFAULT_GOAL := test

CONDA_ENV ?= mlx-ft

.PHONY: sync preflight lint format-check typecheck test audit

sync:
	conda run -n $(CONDA_ENV) scripts/conda-uv-sync.sh

preflight:
	conda run -n $(CONDA_ENV) python -m ftlab.cli preflight

lint:
	conda run -n $(CONDA_ENV) python -m ruff check src tests

format-check:
	conda run -n $(CONDA_ENV) python -m ruff format --check src tests

typecheck:
	conda run -n $(CONDA_ENV) python -m mypy src/ftlab

test:
	conda run -n $(CONDA_ENV) python -m pytest -m 'not real'

audit:
	conda run -n $(CONDA_ENV) python -m ftlab.cli data audit --version smoke --samples 100
