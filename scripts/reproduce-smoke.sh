#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FREE_GIB="$(df -kP . | awk 'NR==2 {print $4 / 1024 / 1024}')"
awk 'BEGIN { if (ARGV[1] < 30) exit 1 }' "$FREE_GIB" || {
  echo "at least 30 GiB of free disk space is required" >&2
  exit 1
}
scripts/conda-run.sh ftlab data prepare --version smoke
scripts/conda-run.sh ftlab data validate --version smoke
scripts/conda-run.sh ftlab preflight --real-model
PROBE_RESULT="$(scripts/conda-run.sh ftlab benchmark probe --config configs/experiments/smoke-06b.yaml --steps 32)"
echo "$PROBE_RESULT"
if ! grep -q '"status": "safe"' <<<"$PROBE_RESULT"; then
  echo "resource safety probe is not safe; refusing to run training" >&2
  exit 1
fi
BASE_STARTED_AFTER="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
scripts/conda-run.sh ftlab evaluate --model Qwen/Qwen3-0.6B-MLX-4bit --split test
BASE_RUN="$(scripts/conda-run.sh python - "$ROOT" "$BASE_STARTED_AFTER" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
started_after = sys.argv[2]
candidates = []
for path in (root / "runs").iterdir():
    if not path.is_dir() or path.name.startswith("."):
        continue
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if manifest.get("kind") == "evaluation" and manifest.get("config", {}).get("workflow") == "base" and status.get("status") == "completed" and manifest.get("started_at", "") >= started_after:
        candidates.append((manifest.get("started_at", ""), path))
if not candidates:
    raise SystemExit("base evaluation did not produce a completed artifact")
print(max(candidates)[1])
PY
)"
TRAIN_STARTED_AFTER="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
scripts/conda-run.sh ftlab train --config configs/experiments/smoke-06b.yaml
TRAIN_RUN="$(scripts/conda-run.sh python - "$ROOT" "$TRAIN_STARTED_AFTER" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
started_after = sys.argv[2]
candidates = []
for path in (root / "runs").iterdir():
    if not path.is_dir() or path.name.startswith("."):
        continue
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "status.json").read_text())
        training = json.loads((path / "training.json").read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if (
        manifest.get("kind") == "training"
        and status.get("status") == "completed"
        and training.get("microbatches") == 128
        and training.get("optimizer_updates") == 16
        and manifest.get("started_at", "") >= started_after
    ):
        candidates.append((manifest.get("finished_at", ""), path))
if not candidates:
    raise SystemExit("training did not produce the required 128 microbatches and 16 updates")
print(max(candidates)[1])
PY
)"
scripts/conda-run.sh python - "$TRAIN_RUN" <<'PY'
import json
import sys
from pathlib import Path

from ftlab.artifacts import validate_run_artifacts

run = Path(sys.argv[1])
missing = validate_run_artifacts(run, kind="training")
if missing:
    raise SystemExit(f"training contract is incomplete: {missing}")
PY
TUNED_STARTED_AFTER="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
scripts/conda-run.sh ftlab evaluate --model Qwen/Qwen3-0.6B-MLX-4bit --split test \
  --adapter-path "$TRAIN_RUN/adapter"
TUNED_RUN="$(scripts/conda-run.sh python - "$ROOT" "$TUNED_STARTED_AFTER" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
started_after = sys.argv[2]
candidates = []
for path in (root / "runs").iterdir():
    if not path.is_dir() or path.name.startswith("."):
        continue
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if manifest.get("kind") == "evaluation" and manifest.get("config", {}).get("workflow") == "tuned" and status.get("status") == "completed" and manifest.get("started_at", "") >= started_after:
        candidates.append((manifest.get("started_at", ""), path))
if not candidates:
    raise SystemExit("tuned evaluation did not produce a completed artifact")
print(max(candidates)[1])
PY
)"
scripts/conda-run.sh python - "$BASE_RUN" "$TUNED_RUN" <<'PY'
import json
import sys
from pathlib import Path

from ftlab.artifacts import validate_run_artifacts

base = Path(sys.argv[1])
tuned = Path(sys.argv[2])
for run in (base, tuned):
    missing = validate_run_artifacts(run, kind="evaluation")
    if missing:
        raise SystemExit(f"evaluation contract is incomplete for {run}: {missing}")
def rows(run):
    return [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines() if line.strip()]
base_rows = rows(base)
tuned_rows = rows(tuned)
if len(base_rows) != len(tuned_rows):
    raise SystemExit("base and tuned evaluation counts differ")
if any("<think>" in str(row.get("prediction", "")).casefold() for row in base_rows):
    raise SystemExit("base generation contains a thinking block")
if [row.get("prompt_hash") for row in base_rows] != [row.get("prompt_hash") for row in tuned_rows]:
    raise SystemExit("base and tuned prompt hashes differ")
if [row.get("decoding") for row in base_rows] != [row.get("decoding") for row in tuned_rows]:
    raise SystemExit("base and tuned decoding settings differ")
PY
scripts/conda-run.sh ftlab data audit --version smoke --samples 100
scripts/conda-run.sh ftlab report build
