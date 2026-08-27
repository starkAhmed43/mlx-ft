# Model card

Project license: [MIT](../LICENSE). Qwen checkpoint terms remain those of the
pinned upstream model snapshots.

The registry contains pinned Qwen3 MLX checkpoints in 0.6B, 1.7B, and 4B
families with 4-bit, 8-bit, and bf16 precision where available. The intentional
validation study has 30 runs and uses an 8-layer, 512-token baseline. It uses
LoRA adapters and preserves the Qwen non-thinking prompt contract.

The model work runs in a fresh child process under a project lock. It does not
execute tools. Winner selection uses validation only. Official final test work
uses a committed lock and is test-once; exact reruns are labeled replicas. BFCL
evaluates the selected validation winner and never tunes a model. Real MLX
measurements are not included. Legacy prompt-contract-0 runs are excluded.
