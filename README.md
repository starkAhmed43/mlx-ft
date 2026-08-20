# mlx-ft

`mlx-ft` is a small, learning-focused lab for function-calling fine-tuning on Apple Silicon.
The first milestone validates the full Qwen3 0.6B 4-bit pipeline. It does not claim a quality gain.

Use the dedicated Conda environment from `environment.yml`, then run the locked sync:

```sh
scripts/conda-uv-sync.sh
```
Equivalent direct command:

```sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --extra apple --extra data --extra tracking --extra reporting --group dev
```
The wrapper sets `UV_PROJECT_ENVIRONMENT` to the active Conda prefix. The
preflight command requires Conda environment `mlx-ft` and Python 3.14.6:

```sh
conda run -n mlx-ft python -m ftlab.cli preflight
```
The `apple` extra is required for model work. Core parsing and data tests do not import MLX.
The project keeps BFCL in a separate Python 3.12 environment because its pinned
NumPy requirement conflicts with the project environment. See `environment-bfcl.yml`.

Read the beginner [tutorial](docs/tutorial.md) for the mental model and
milestone commands. The [tool-calling lab notebook](docs/tool-calling-lab.ipynb)
is a safe, fixture-only walkthrough. It does not download data or models,
train, execute tools, or initialize W&B.

Read the [data card](cards/data-card.md), [model card](cards/model-card.md),
[limitations](cards/limitations.md), and [interview notes](cards/interview-notes.md)
before describing results.

Run a deterministic aggregate audit without writing raw records:

```sh
ftlab data audit --version smoke --samples 100
```

Reproduce the corrected smoke workflow with:

```sh
scripts/reproduce-smoke.sh
```

The best-run wrapper refuses to run until a selected completed controlled
artifact and resolved configuration are supplied:

```sh
scripts/reproduce-best.sh --run runs/SELECTED_RUN --config configs/experiments/selected.yaml
```

Reports are preliminary until final controlled runs have commit and provenance
metadata. The four legacy prompt-contract-0 runs are excluded from reports.
No W&B or GitHub upload occurs by default. Private-project and repository-auth
checks remain manual external gates.
