"""Lazy adapter for the pinned Berkeley Function Calling benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data import normalize_schema

BFCL_VERSION = "2026.3.23"
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_CATEGORIES = ("simple_python", "irrelevance")


def bfcl_preflight(environment: str = "mlx-ft-bfcl") -> dict[str, str]:
    """Verify the dedicated pinned environment without importing it locally."""
    command = [
        "conda",
        "run",
        "-n",
        environment,
        "python",
        "-c",
        (
            "import importlib.metadata as m, subprocess; "
            f"assert m.version('bfcl-eval') == '{BFCL_VERSION}'; "
            "subprocess.run(['bfcl', '--help'], check=True, capture_output=True); "
            "print(m.version('bfcl-eval'))"
        ),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("BFCL preflight failed in the dedicated Conda environment") from exc
    installed = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if installed != BFCL_VERSION:
        raise RuntimeError("BFCL preflight did not report the pinned version")
    return {"environment": environment, "bfcl_version": installed, "source_commit": BFCL_COMMIT}


def _case_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_bfcl_cases(
    rows: Iterable[dict[str, Any]], *, per_category: int = 200, seed: int = 42
) -> list[dict[str, Any]]:
    """Select exactly 200 cases per supported category by stable hash."""
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in BFCL_CATEGORIES}
    for row in rows:
        category = row.get("category", row.get("type"))
        if category in grouped and isinstance(row, dict) and _supported_schema(row):
            grouped[category].append(row)
    selected: list[dict[str, Any]] = []
    for category in BFCL_CATEGORIES:
        ranked = sorted(
            grouped[category],
            key=lambda row: hashlib.sha256(f"{seed}:{_case_hash(row)}".encode()).hexdigest(),
        )
        if len(ranked) < per_category:
            raise ValueError(f"BFCL category {category} has only {len(ranked)} supported cases")
        selected.extend(ranked[:per_category])
    return selected


def _schema_values(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Find schema objects in both fixture and official BFCL row shapes."""
    found: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"parameters", "schema"} and isinstance(child, dict):
                    found.append(child)
                elif key == "function" and isinstance(child, dict):
                    visit(child)
                elif key in {"functions", "tools"} and isinstance(child, list):
                    visit(child)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(row)
    return found


def _supported_schema(row: dict[str, Any]) -> bool:
    """Keep only simple JSON-schema cases supported by the study parser."""
    # A schema-less fixture is accepted as the empty object used by the local
    # parser tests. Real BFCL rows must pass recursive normalization.
    schemas = _schema_values(row)
    if not schemas:
        return True
    for schema in schemas:
        if schema == {}:
            schema = {"type": "object", "properties": {}}
        try:
            normalize_schema(schema)
        except ValueError:
            return False
    return True


def _category_candidates(root: Path) -> dict[str, list[Path]]:
    candidates: dict[str, list[Path]] = {category: [] for category in BFCL_CATEGORIES}
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.suffix not in {".json", ".jsonl"}:
            continue
        for category in BFCL_CATEGORIES:
            if category in item.name:
                candidates[category].append(item)
    return candidates


def _find_installed_sources() -> dict[str, Path]:
    try:
        if importlib.metadata.version("bfcl-eval") != BFCL_VERSION:
            raise RuntimeError(f"bfcl-eval must be pinned to {BFCL_VERSION}")
        root = importlib.resources.files("bfcl_eval")
    except ImportError, ModuleNotFoundError, importlib.metadata.PackageNotFoundError:
        try:
            result = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    "mlx-ft-bfcl",
                    "python",
                    "-c",
                    (
                        "import importlib.metadata as m, importlib.resources as r; "
                        f"assert m.version('bfcl-eval') == '{BFCL_VERSION}'; "
                        "print(r.files('bfcl_eval'))"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            package_path = Path(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "BFCL data is unavailable; install environment-bfcl.yml or pass --source"
            ) from exc
    else:
        package_path = Path(str(root))
    candidates = _category_candidates(package_path)
    missing = [category for category, values in candidates.items() if len(values) != 1]
    if missing:
        raise RuntimeError(
            "BFCL pinned category files are ambiguous or missing: " + ", ".join(missing)
        )
    return {category: values[0] for category, values in candidates.items()}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        values = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else value.get("rows", [])
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise ValueError(f"BFCL source must contain object rows: {path}")
    return values


def prepare_bfcl(
    source: str | Path | None,
    output: str | Path,
    *,
    per_category: int = 200,
) -> Path:
    """Select and hash local BFCL rows without importing or downloading BFCL."""
    if source is None:
        paths = _find_installed_sources()
    else:
        source_path = Path(source)
        if source_path.is_dir():
            discovered = _category_candidates(source_path)
            missing = [category for category, values in discovered.items() if len(values) != 1]
            if missing:
                raise ValueError(
                    "explicit BFCL source must contain one file per category: " + ", ".join(missing)
                )
            paths = {category: values[0] for category, values in discovered.items()}
        else:
            inferred = next(
                (category for category in BFCL_CATEGORIES if category in source_path.name), None
            )
            paths = {inferred or "": source_path}
    rows: list[dict[str, Any]] = []
    for category, path in paths.items():
        rows.extend(
            {
                **row,
                "category": row.get("category", category),
            }
            for row in _load_rows(path)
        )
    if any(row.get("category") not in BFCL_CATEGORIES for row in rows):
        raise ValueError("BFCL source rows must identify simple_python or irrelevance")
    selected = select_bfcl_cases(rows, per_category=per_category)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bfcl_version": BFCL_VERSION,
        "source_commit": BFCL_COMMIT,
        "count": len(selected),
        "rows": [
            {
                "id": row.get("id", index),
                "hash": _case_hash(row),
                "category": row.get("category", row.get("type", "")),
            }
            for index, row in enumerate(selected)
        ],
        "records_file": target.with_name(target.stem + ".records.jsonl").name,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records_path = target.with_name(target.stem + ".records.jsonl")
    records_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected) + "\n",
        encoding="utf-8",
    )
    return target


def export_bfcl_responses(
    responses: Iterable[dict[str, Any] | str], output: str | Path, *, run_id: str | None = None
) -> Path:
    """Export MLX responses in the official JSONL-shaped result format."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(responses):
            if isinstance(value, str):
                value = {"model_response": value}
            # Official BFCL rows do not define a project run_id field.
            response = value.get("prediction", value.get("model_response", value.get("result", "")))
            row = {"id": value.get("id", index), "result": response}
            for key in ("inference_log", "tokens", "latency"):
                if key in value:
                    row[key] = value[key]
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def build_bfcl_eval_commands(
    *,
    model_name: str = "mlx",
    result_dir: str | Path | None = None,
    partial_eval: bool = True,
    executable: str = "bfcl",
) -> list[list[str]]:
    """Build one official partial-evaluation command per selected category."""
    suffix = ["--partial-eval"] if partial_eval else []
    if result_dir is not None:
        suffix += ["--result-dir", str(result_dir)]
    return [
        [executable, "evaluate", "--model", model_name, "--test-category", category, *suffix]
        for category in BFCL_CATEGORIES
    ]


def evaluate_bfcl(
    run_dir: str | Path, *, project_root: str | Path | None = None, partial_eval: bool = True
) -> list[subprocess.CompletedProcess[str]]:
    """Invoke the pinned BFCL evaluator in its dedicated Conda environment."""
    run_path = Path(run_dir)
    root = Path(project_root or run_path).resolve()
    if not run_path.is_absolute():
        run_path = root / "runs" / run_path
    from .study import load_final_lock

    lock = load_final_lock(root)
    parent = run_path.name.removesuffix("-bfcl")
    try:
        run_manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        parent = str((run_manifest.get("config") or {}).get("parent_run", parent))
    except OSError, json.JSONDecodeError:
        pass
    if parent not in lock["candidates"]:
        raise ValueError("BFCL evaluation requires an exactly selected final candidate")
    manifest_path = run_path / "bfcl.manifest.json"
    if not manifest_path.exists():
        manifest_path = root / "data" / "bfcl.manifest.json"
    if not manifest_path.exists():
        raise ValueError("BFCL selected-case manifest is required before evaluation")
    selected = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_path = manifest_path.parent / str(selected.get("records_file", ""))
    if not records_path.exists():
        raise ValueError("BFCL selected records file is missing")
    selected_records = {str(row.get("id")): row for row in _load_rows(records_path)}
    for item in selected.get("rows", []):
        case_id = str(item.get("id"))
        if case_id not in selected_records or _case_hash(selected_records[case_id]) != item.get(
            "hash"
        ):
            raise ValueError("BFCL selected record hash does not match its manifest")
    predictions_path = run_path / "predictions.jsonl"
    if not predictions_path.exists():
        raise ValueError("run predictions are required before BFCL evaluation")
    allowed = {str(item["id"]): item for item in selected.get("rows", [])}
    responses = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    response_ids = [str(row.get("id")) for row in responses]
    if any(case_id not in allowed for case_id in response_ids):
        raise ValueError("run predictions contain IDs outside the selected BFCL cases")
    if len(response_ids) != len(set(response_ids)) or set(response_ids) != set(allowed):
        raise ValueError("run predictions must contain each selected BFCL case exactly once")
    model_name = "mlx_ftlab"
    result_root = root / "result" / model_name
    result_root.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in BFCL_CATEGORIES}
    for row in responses:
        category = allowed[str(row["id"])].get("category", "")
        if category in by_category:
            by_category[category].append(row)
    for category, category_rows in by_category.items():
        target = result_root / f"BFCL_v4_{category}_result.json"
        staged: list[dict[str, Any]] = []
        for row in category_rows:
            one = {"id": row["id"], "result": row.get("prediction", row.get("result", ""))}
            for key in ("inference_log", "tokens", "latency"):
                if key in row:
                    one[key] = row[key]
            staged.append(one)
        target.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in staged),
            encoding="utf-8",
        )
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(root)
    commands = build_bfcl_eval_commands(
        model_name=model_name,
        result_dir=root / "result",
        partial_eval=partial_eval,
        executable="bfcl",
    )
    bfcl_preflight()
    commands = [["conda", "run", "-n", "mlx-ft-bfcl", *command] for command in commands]
    try:
        return [
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)
            for command in commands
        ]
    except FileNotFoundError as exc:
        raise RuntimeError("install the optional bfcl-eval dependency to evaluate BFCL") from exc
