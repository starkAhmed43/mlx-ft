# mlx-ft implementation and execution status

## Purpose

This repository implements an Apple Silicon function-calling fine-tuning
study. It does not contain completed experiment evidence or a quality claim.

## Implemented study contract

- Qwen3 MLX-LM training uses LoRA with the native non-thinking prompt format.
- The intentional study matrix contains 30 validation runs and uses the
  8-layer, 512-token baseline.
- Data preparation separates eligibility tiers. Smoke, day1, core, and nested
  1k/5k/10k pools use records at or below 512 rendered tokens. The 20k endpoint
  retains the 10k pool and can extend it with records at or below 2,048 tokens.
- Schema-OOD records are reserved before IID locks. IID validation and test
  locks are shared. Duplicate, API-prefix, and schema-fingerprint leakage is
  excluded where the split contract requires it.
- `task_success` requires the correct tool, schema-valid arguments, and every
  schema-required gold argument to be correct. `exact_match` remains a separate
  complete-object metric. Argument key and value F1 are micro-F1 scores over
  all evaluated argument leaves.
- Parser diagnostics describe output parsing states. The task-error taxonomy
  separately reports task failures such as a wrong tool, missing argument,
  extra argument, wrong type, or wrong value.
- Winner selection uses validation only. A committed final-selection lock gates
  the official IID and Schema-OOD test evaluation. The first valid result is
  official; exact reruns are labeled replicas.
- BFCL evaluates the validation-selected winner only. It does not tune models
  or affect winner selection.
- Sanitized manifests, selection locks, and final report evidence are tracked.
  Raw records, models, adapters, predictions, and run directories stay local.

## Implementation verification: 2026-08-27

Implementation checks pass: fixture tests, Ruff lint, Ruff format, Mypy,
lockfile validation, package build, the 30-run sweep dry expansion, and the
BFCL preflight. Three real-asset tests are skipped because their model assets
are unavailable. Before any study run, prepare fresh data, inspect the audit
queue, and validate every generated manifest. Do not use the local legacy
prompt-contract-0 smoke data or the legacy run directories.

## Execution roadmap

1. Pass fixture, lint, format, type, lockfile, and real-model preflight gates.
2. Prepare and validate fresh smoke, day1, core, full, Schema-OOD, and nested
   dataset manifests. Record only sanitized descriptors and aggregate audits.
3. Run the smoke gate, then bounded safety probes and the 30 validation runs.
4. Select and commit the validation winner lock. Run the official IID and
   Schema-OOD test evaluation once; label exact repeats as replicas.
5. Run robustness and BFCL against that locked winner.
6. Generate and review controlled-axis reports, paired confidence intervals,
   tables, figures, findings, and representative errors.

## Evidence status

Real datasets, training runs, final IID and Schema-OOD measurements,
robustness results, and BFCL measurements remain unexecuted. No benchmark
result or model-quality finding is claimed as valid evidence. Generated local
artifacts are ignored until the required sanitized final evidence is
deliberately reviewed and added.
