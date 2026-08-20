#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN_DIR=""
CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "usage: $0 --run RUN_DIR --config CONFIG" >&2; exit 2 ;;
  esac
done
if [[ -z "$RUN_DIR" || -z "$CONFIG" ]]; then
  echo "usage: $0 --run RUN_DIR --config CONFIG" >&2
  exit 2
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "selected run directory does not exist" >&2
  exit 1
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
if [[ ! -f "$CONFIG" ]]; then
  echo "selected configuration does not exist" >&2
  exit 1
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
if [[ ! -f "$RUN_DIR/manifest.json" || ! -f "$RUN_DIR/status.json" ]]; then
  echo "selected run must contain an artifact-contract manifest and status" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/resolved_config.yaml" ]]; then
  echo "selected run must contain resolved_config.yaml" >&2
  exit 1
fi

scripts/conda-run.sh python - "$RUN_DIR" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

run = Path(sys.argv[1])
supplied_path = Path(sys.argv[2])
manifest = json.loads((run / "manifest.json").read_text())
status = json.loads((run / "status.json").read_text())
if manifest.get("artifact_version") != 1 or manifest.get("prompt_contract_version") != 1:
    raise SystemExit("selected run is not a contract-v1 artifact")
if manifest.get("controlled") is not True:
    raise SystemExit("selected run is not explicitly controlled")
if status.get("status") != "completed" or manifest.get("status") != "completed":
    raise SystemExit("selected run is not completed")
if not (manifest.get("git") or {}).get("commit"):
    raise SystemExit("selected run has no commit provenance")
resolved = yaml.safe_load((run / "resolved_config.yaml").read_text()) or {}
supplied = yaml.safe_load(supplied_path.read_text()) or {}
for key in ("model", "model_revision", "dataset"):
    if key in supplied and supplied[key] != resolved.get(key):
        raise SystemExit(f"supplied config does not match resolved {key}")
for section, keys in {
    "training": ("max_seq_length", "batch_size", "grad_accumulation_steps", "num_layers", "lora_parameters"),
    "rendering": ("enable_thinking", "mask_prompt", "max_seq_length"),
}.items():
    selected = supplied.get(section, {})
    actual = resolved.get(section, {})
    for key in keys:
        if key in selected and selected[key] != actual.get(key):
            raise SystemExit(f"supplied config does not match resolved {section}.{key}")
PY

exec scripts/conda-run.sh ftlab train --config "$RUN_DIR/resolved_config.yaml"
