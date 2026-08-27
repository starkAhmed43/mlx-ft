# mlx-ft

`mlx-ft` is a learning-focused lab for function-calling fine-tuning on Apple Silicon.
It contains the study implementation, not completed study results.

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

The study uses an intentional 30-run, 8-layer, 512-token baseline matrix.
Small datasets use records rendered at 512 tokens or less. Only the 20k
training endpoint can add records rendered at up to 2,048 tokens.

Official final test evaluation is locked and test-once. Exact reruns are
allowed, but the system labels them as replicas. BFCL validates the selected
validation winner; it never selects or tunes a winner.

The final reproduction wrapper requires a committed selection lock:

```sh
scripts/reproduce-best.sh --candidate RUN_ID
```

Tracked evidence is limited to sanitized dataset descriptors, selection locks,
and final tables, figures, and findings. Raw records, models, adapters,
predictions, and run directories remain local and ignored. The four legacy
prompt-contract-0 runs are excluded. No result, prepared dataset, or quality
gain is claimed here.
