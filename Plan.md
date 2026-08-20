# mlx-ft audit snapshot — 2026-08-20

## Overall verdict

This is the pre-initial-commit audit snapshot recorded on 2026-08-20. The repository has a substantial framework for the planned study, but the study is not complete. Most experiment code exists, but almost all real data preparation, training, evaluation, BFCL work, and final reporting remain unexecuted. A data-allocation defect currently blocks the corrected smoke and core pipelines.

## Status

| Area | Status |
| --- | --- |
| Environment and project structure | Mostly implemented |
| Prompt contract | Implemented and real-tested |
| Evaluation semantics | Mostly implemented |
| Data preparation | Incorrect and blocked |
| Runner and artifact contract | Partial |
| Sweep configuration | Defined; execution has defects |
| Experiments | Not run |
| BFCL | Installed; integration is not functional |
| Robustness | Suite implemented; not run |
| Reports | Generators exist; no valid results |
| Documentation | Present; partly ahead of evidence |
| Final delivery gates | Failing or incomplete |

## Implemented

- The `mlx-ft` Conda environment uses Python 3.14.6. Dependency compatibility was verified for 86 packages, and the unwanted `.venv` is absent.
- Conda-only preflight and wrappers exist. A separate Python 3.12 `mlx-ft-bfcl` environment contains `bfcl-eval==2026.3.23`.
- The renderer preserves Qwen non-thinking controls, masks template-owned tokens, and starts the target at `<tool_call>`. The real MLX integration tests pass.
- Strict schema normalization, transitive duplicate grouping, SHA ranking, Schema-OOD primitives, parsing, independent validity metrics, the ten-category taxonomy, bootstrap intervals, and seed summaries exist.
- CLI support exists for data audit, benchmark probe, sweep, robustness, and BFCL. Five pinned model profiles are registered.
- Child-process execution, exclusive job locking, and one-second system sampling exist. The local robustness suite has 80 call-required and 20 no-call cases.
- The controlled-training matrix expands to 30 entries. License, Makefile, cards, notebook, tutorial, CI, and reproduction scripts exist.
- Hugging Face, GitHub, and W&B authentication work. The `starkAhmed43/mlx-ft` repository and `starkahmed43/mlx-ft` W&B project were verified private.

## Partial or incorrect

### Data allocation

Partial 2,048-token full-allocation support exists. The normal less-than-or-equal-to-512-token branch still always attempts the full 20,000/750/1,000 allocation. The documented accepted pool has 21,718 records, fewer than the 21,750 minimum before duplicate exclusions. This blocks smoke, day1, and core preparation. Those versions must allocate independently at 512 tokens; the nested 1k–20k study must use the 2,048-token pool.

The current smoke manifest is legacy prompt-contract version 0. It lacks current hashes and context views, and `ftlab data validate --version smoke` fails because its test split-ID hash does not match.

### Sweep and safety logic

- Probe reuse is keyed only by model and precision. It can reuse an insufficient probe after rank, layer, target, context, or microbatch changes.
- Winner-seed processing can reselect the winner and overwrite its learning rate.
- Model-size exposure does not tie the 0.6B and 4B endpoints to a measured 1.7B reference.
- Sweep evaluation always uses validation. The locked IID and Schema-OOD final evaluation flow, separate base evaluation, and Pareto selection are absent.
- The safety classifier treats current free disk as projected free disk instead of estimating completed-run size.

### Artifact and runner contracts

- Validation permits empty declared artifacts.
- Tokenizer hashes and checkpoint-byte fields are not always populated.
- The runner replaces the original user command with the worker command.
- Worker requests, exception results, and logs can contain absolute paths or sensitive exception text.
- Timeout handling can wait forever after `SIGTERM`.
- Standalone training is always marked uncontrolled, so a smoke training run cannot become report-eligible.

### BFCL and reports

The official BFCL executable fails to import because `soundfile` is missing. The adapter can bypass the pinned BFCL environment, uses incomplete official metadata, uses a BFCL v3-style result name although the data is v4, and lacks a validation-winner gate.

Report generators create the seven expected filenames, but there are no valid controlled runs. Pareto/frontier calculations, study-axis filtering, sweep RSS aggregation, and comparable base-to-tuned examples are incomplete. Findings remain preliminary.

Documentation describes full, nested, and Schema-OOD datasets as if they exist and states a 512-token normalization limit that conflicts with the required 2,048-token full pool.

## Missing or unexecuted work

- Prepare and validate day1, core, full, Schema-OOD, and nested 1k/5k/10k/20k data versions.
- Produce the deterministic 100-record audit queue and record aggregate findings.
- Regenerate corrected smoke data, base evaluation, training, and tuned evaluation.
- Run safety probes, learning-rate pilots, headline precision, capacity, data, context, batch, size, and winner-seed studies.
- Run the three 1.7B bases, locked IID and Schema-OOD evaluation, validation-only winner selection, and final Pareto evaluation.
- Run robustness and official BFCL preparation, export, and evaluation.
- Produce valid confidence summaries, examples, final reports, measured findings, and sanitized W&B uploads.

## Current checks (2026-08-20 pre-commit audit results)

Passed:

- Conda preflight; Python 3.14.6; package compatibility; `uv lock --check`; Ruff lint; `uv build`.
- Real MLX integration tests: 3 passed.
- Sweep dry-run: 30 entries.
- GitHub, W&B, and Hugging Face authentication/privacy checks. `gh repo view` reported PRIVATE, the W&B GraphQL privacy verifier passed, and `hf auth whoami` passed.

Failed:

- Fixture tests: 117 passed, 1 failed, 3 deselected.
- Mypy: one error in `src/ftlab/worker.py`.
- Ruff formatting: three files need formatting.
- Smoke data validation.
- BFCL executable import.

Incomplete:

- The full ignored-cache `gitleaks --no-git` scan was interrupted.
- `git diff --check` passes, but cannot inspect untracked content.

## External gates and repository state (pre-initial-commit snapshot)

At the time of this audit, no commit existed and source files were untracked. Generated data, models, adapters, predictions, caches, raw records, and run directories remain outside the initial source commit. Current local smoke data and four local runs are legacy prompt-contract-0 artifacts and must remain excluded from final reports.

Hugging Face and W&B access are available. W&B uploads must wait for valid final measurements. GitHub remote privacy is verified. Existing legacy manifests remain null or lack a commit field. After the authorized initial commit, new manifests should record that commit; results remain preliminary until the study completes.

## Prioritized next work

1. Repair data allocation and manifest validation; update data-card claims and add focused tests.
2. Repair sweep selection, probe identity, model-size exposure, runner safety, artifact privacy, and BFCL integration; then pass lint, format, type, and fixture tests.
3. Regenerate and validate each immutable data version, then inspect the audit queue.
4. Run the corrected smoke gate: matching prompts, no base thinking blocks, required artifacts, 128 microbatches, 16 updates, and real MLX gate.
5. Run sequential safety-gated experiments and locked final evaluation. Do not tune on BFCL.
6. Generate reports only from eligible controlled runs, validate raw-to-table outputs, and upload sanitized W&B results only after final measurement checks.
7. Complete all delivery checks, including the full secret scan, then review each changed file before a study-completion commit.
