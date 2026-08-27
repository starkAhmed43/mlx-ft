"""Immutable selection locks for the final, test-once study stage."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

LOCK_VERSION = 1


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _clean_tree(root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("a git worktree is required to lock final selection") from exc
    return not result.stdout.strip()


def candidate_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    """The published deterministic selection order."""
    selection = row.get("selection", row)
    if not isinstance(selection, dict):
        selection = row
    return (
        float(selection.get("task_success", 0.0)),
        float(selection.get("argument_value_f1", 0.0)),
        float(selection.get("schema_validity", 0.0)),
        -float(selection.get("rss_gib", selection.get("peak_rss_gib", float("inf")))),
        float(
            selection.get(
                "generation_throughput_tokens_per_second",
                selection.get("throughput_tokens_per_second", 0.0),
            )
        ),
        str(row.get("fingerprint", row.get("run_id", ""))),
    )


def nondominated(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return quality-high, RSS-low Pareto candidates in stable order."""

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        a, b = left.get("selection", left), right.get("selection", right)
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        quality = ("task_success", "argument_value_f1", "schema_validity")
        not_worse = all(float(a.get(key, 0.0)) >= float(b.get(key, 0.0)) for key in quality)
        rss_a = _number(a.get("rss_gib", a.get("peak_rss_gib")), float("inf"))
        rss_b = _number(b.get("rss_gib", b.get("peak_rss_gib")), float("inf"))
        strictly = (
            any(float(a.get(key, 0.0)) > float(b.get(key, 0.0)) for key in quality) or rss_a < rss_b
        )
        return not_worse and rss_a <= rss_b and strictly

    return sorted(
        [
            row
            for row in rows
            if not any(dominates(other, row) for other in rows if other is not row)
        ],
        key=candidate_key,
        reverse=True,
    )


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float, str)) else default


def create_final_lock(root: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a content-addressed selection lock from validation-only rows."""
    base = Path(root).resolve()
    if not _clean_tree(base):
        raise RuntimeError("final selection requires a clean committed git tree")
    eligible = [
        row
        for row in rows
        if row.get("status", "completed") == "completed"
        and row.get("validation_only", True)
        # A missing field is not evidence that a run was controlled.  Locks
        # are only valid for rows that explicitly passed the artifact gate.
        and row.get("controlled") is True
        and row.get("run_id")
    ]
    if not eligible:
        raise ValueError("no controlled validation-only runs are available for final selection")
    winner = max(eligible, key=candidate_key)
    competitors = [row for row in nondominated(eligible) if row is not winner][:2]
    candidate_rows = [winner, *competitors]
    payload: dict[str, Any] = {
        "lock_version": LOCK_VERSION,
        "winner": winner["run_id"],
        "pareto_competitors": [row["run_id"] for row in competitors],
        "candidates": [row["run_id"] for row in candidate_rows],
        "candidate_recipes": {
            row["run_id"]: {
                "source_run": row["run_id"],
                "resolved_fingerprint": row.get("resolved_fingerprint", row.get("fingerprint", "")),
                "recipe": row.get("recipe", {}),
                "hashes": row.get("hashes", {}),
                "validation_metrics": row.get("selection", {}),
                "git_commit": row.get("git_commit"),
                "immutable": row.get("immutable", {}),
            }
            for row in candidate_rows
        },
        "selection_rule": "task_success,value_f1,schema_validity,lower_rss,higher_generation_throughput,lexical_fingerprint",
        "selection_source": "validation_only_controlled_runs",
    }
    payload["content_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    target = base / "reports" / "final-selection.lock.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("final selection lock already exists and is immutable")
        return existing
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_final_lock(root: str | Path) -> dict[str, Any]:
    target = Path(root).resolve() / "reports" / "final-selection.lock.json"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("a final-selection lock is required") from exc
    digest = payload.pop("content_sha256", None)
    if payload.get("lock_version") != LOCK_VERSION or not isinstance(digest, str):
        raise ValueError("final-selection lock has an unsupported format")
    if hashlib.sha256(_canonical(payload).encode()).hexdigest() != digest:
        raise ValueError("final-selection lock content hash does not match")
    payload["content_sha256"] = digest
    return payload
