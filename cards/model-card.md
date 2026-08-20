# Model card

Project license: [MIT](../LICENSE). Qwen checkpoint terms remain those of the
pinned upstream model snapshots.

The registry contains pinned Qwen3 MLX checkpoints in 0.6B, 1.7B, and 4B
families with 4-bit, 8-bit, and bf16 precision where available. The study uses
LoRA adapters and preserves the Qwen non-thinking prompt contract.

The model work runs in a fresh child process under a project lock. It does not
execute tools. Real MLX measurements are not included in this preliminary card.
Legacy prompt-contract-0 runs are excluded from reports because their prompt
contract differed between training and evaluation.
