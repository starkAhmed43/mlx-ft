#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 2 || "$1" != "--candidate" ]]; then
  echo "usage: $0 --candidate RUN_ID" >&2
  exit 2
fi

CANDIDATE="$2"
scripts/conda-run.sh ftlab study lock-winner
exec scripts/conda-run.sh ftlab study reproduce-final --candidate "$CANDIDATE"
