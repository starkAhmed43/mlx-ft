"""Typer command line for the learning lab."""

from __future__ import annotations

import csv
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

import typer

from .artifacts import (
    append_training_metric,
    controlled_run_eligible,
    create_run_contract,
    update_manifest,
    write_status,
)
from .bfcl import evaluate_bfcl, export_bfcl_responses, prepare_bfcl
from .config import MODEL_ID, get_model_spec, load_config
from .data import (
    AcceptedRecord,
    allocate_schema_ood_test,
    audit_queue,
    audit_records,
    build_full_allocation,
    build_nested_allocation,
    context_eligible_records,
    duplicate_groups,
    load_xlam,
    normalize_records,
    source_flow,
    stratified_split_records,
    stratify_records,
)
from .environment import validate_conda_environment
from .hashing import sha256_file, sha256_records
from .metrics import evaluate_predictions
from .report import build_report
from .robustness import evaluate_robustness, prepare_robustness
from .runner import classify_probe_safety, run_isolated
from .storage import (
    cache_root,
    cleanup_candidates,
    ensure_project_dirs,
    require_free_space,
    storage_status,
    system_metrics,
    write_json,
)
from .sweep import SweepConfig, SweepRun, expand_matrix, load_sweep, run_sweep, select_best
from .tracking import finish_tracking, init_tracking, sanitized_row, validate_metric_allowlist

app = typer.Typer(
    help="Learning-focused MLX function-calling fine-tuning lab.", no_args_is_help=True
)
data_app = typer.Typer(help="Prepare and validate pinned xLAM data.")
storage_app = typer.Typer(help="Inspect local artifact storage.")
report_app = typer.Typer(help="Build local reports.")
robustness_app = typer.Typer(help="Prepare and score call/no-call robustness fixtures.")
benchmark_app = typer.Typer(help="Run bounded benchmark safety probes.")
bfcl_app = typer.Typer(help="Use the optional pinned BFCL benchmark adapter.")
app.add_typer(data_app, name="data")
app.add_typer(storage_app, name="storage")
app.add_typer(report_app, name="report")
app.add_typer(robustness_app, name="robustness")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(bfcl_app, name="bfcl")


def _root() -> Path:
    return Path.cwd()


def _fail(exc: Exception) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1)


@app.command()
def preflight(
    real_model: bool = typer.Option(False, help="Load the pinned model and generate one response."),
) -> None:
    """Check core imports and optionally run the real Apple model gate."""
    try:
        environment = validate_conda_environment()
        import numpy
        import psutil
        import pydantic

        typer.echo(f"environment={environment.name}")
        typer.echo(f"python={platform.python_version()}")
        typer.echo(f"jsonschema={package_version('jsonschema')}")
        typer.echo(f"numpy={numpy.__version__}")
        typer.echo(f"pydantic={pydantic.__version__}")
        typer.echo(f"psutil={psutil.__version__}")
        typer.echo(f"pyyaml={package_version('PyYAML')}")
        if real_model:
            import datasets
            import matplotlib
            import pandas
            import transformers

            import wandb

            typer.echo(
                "optional="
                + ",".join(
                    f"{module.__name__}={getattr(module, '__version__', 'installed')}"
                    for module in (datasets, matplotlib, pandas, transformers, wandb)
                )
            )

            from .modeling import generate_predictions, load_model
            from .rendering import load_tokenizer, render_tools_prompt

            model_spec = get_model_spec(MODEL_ID)
            cache = cache_root(_root())
            model, _ = load_model(
                model_spec.model_id, revision=model_spec.revision, cache_root=cache
            )
            tokenizer = load_tokenizer(
                model_spec.model_id, revision=model_spec.revision, cache_root=cache
            )
            tools = [{"name": "fixture.echo", "parameters": {"type": "object", "properties": {}}}]
            prompt = render_tools_prompt(tools, "Say hello.", tokenizer)
            result = generate_predictions(model, tokenizer, [prompt], max_tokens=16)
            typer.echo(f"model_response={result[0][:80]}")
        else:
            typer.echo("model=skipped (pass --real-model for the Apple integration gate)")
    except Exception as exc:
        _fail(exc)


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("fixture must contain a JSON array or JSONL records")
    return value


def _exclude_locked_duplicate_components(
    records: list[AcceptedRecord], locked: list[AcceptedRecord]
) -> list[AcceptedRecord]:
    """Remove every duplicate component connected to a locked record."""
    combined = [*locked, *records]
    locked_indexes = set(range(len(locked)))
    blocked_indexes = {
        index
        for group in duplicate_groups(combined)
        if set(group) & locked_indexes
        for index in group
    }
    return [
        item
        for index, item in enumerate(combined[len(locked) :], len(locked))
        if index not in blocked_indexes
    ]


@data_app.command("prepare")
def data_prepare(
    version: str = typer.Option("smoke", help="Locked manifest version."),
    source: Path | None = typer.Option(
        None, help="Fixture JSON or JSONL. Omit to load pinned xLAM."
    ),
    max_seq_length: int = typer.Option(512, min=1),
) -> None:
    """Normalize data and write a deterministic manifest."""
    sizes = {
        "smoke": {"train": 128, "validation": 32, "test": 64},
        "day1": {"train": 2000, "validation": 250, "test": 250},
        "core": {"train": 10000, "validation": 750, "test": 1000},
        "full": {"train": 20000},
        "schema_ood": {"train": 0, "validation": 0, "test": 1000},
        "nested_1k": {"train": 1000, "validation": 750, "test": 1000},
        "nested_5k": {"train": 5000, "validation": 750, "test": 1000},
        "nested_10k": {"train": 10000, "validation": 750, "test": 1000},
        "nested_20k": {"train": 20000, "validation": 750, "test": 1000},
        "nested-1k": {"train": 1000, "validation": 750, "test": 1000},
        "nested-5k": {"train": 5000, "validation": 750, "test": 1000},
        "nested-10k": {"train": 10000, "validation": 750, "test": 1000},
        "nested-20k": {"train": 20000, "validation": 750, "test": 1000},
    }
    if version not in sizes:
        _fail(ValueError(f"unknown manifest version: {version}"))
    try:
        data_model = get_model_spec(MODEL_ID)
        records = (
            _load_fixture(source) if source else load_xlam(cache_root=str(cache_root(_root())))
        )
        tokenizer = None
        if source is None:
            from .rendering import load_tokenizer, render_record

            tokenizer = load_tokenizer(
                data_model.model_id,
                revision=data_model.revision,
                cache_root=cache_root(_root()),
            )
            accepted, rejected = normalize_records(
                records,
                max_seq_length=None if max_seq_length > 512 else max_seq_length,
                length_fn=lambda item: len(render_record(item, tokenizer).full_tokens),
            )
        else:
            accepted, rejected = normalize_records(
                records, max_seq_length=None if max_seq_length > 512 else max_seq_length
            )
        context_base: dict[str, Any] | None = None
        if max_seq_length > 512:
            can_build_full = version in {
                "full",
                "nested_20k",
                "nested-20k",
            }
            if can_build_full:
                core_path = _root() / "data" / "core" / "manifest.json"
                if not core_path.exists():
                    raise ValueError(f"2048 full eligibility requires locked core manifest: {core_path}")
                core_payload = json.loads(core_path.read_text(encoding="utf-8"))
                base_validation, base_validation_rejected = normalize_records(
                    core_payload.get("splits", {}).get("validation", [])
                )
                base_test, base_test_rejected = normalize_records(
                    core_payload.get("splits", {}).get("test", [])
                )
                if base_validation_rejected or base_test_rejected:
                    raise ValueError("locked core validation or test split is invalid")
                split, eligible_all, context_rejected = build_full_allocation(
                    accepted,
                    base_validation,
                    base_test,
                    max_seq_length=max_seq_length,
                )
                rejected = [*rejected, *context_rejected]
            else:
                base_path = _root() / "data" / version / "manifest.json"
                if not base_path.exists():
                    raise ValueError(
                        f"context {max_seq_length} requires the locked 512-token manifest: {base_path}"
                    )
                context_base = json.loads(base_path.read_text(encoding="utf-8"))
                base_train, base_train_rejected = normalize_records(
                    context_base.get("splits", {}).get("train", [])
                )
                base_validation, base_validation_rejected = normalize_records(
                    context_base.get("splits", {}).get("validation", [])
                )
                base_test, base_test_rejected = normalize_records(
                    context_base.get("splits", {}).get("test", [])
                )
                if base_train_rejected or base_validation_rejected or base_test_rejected:
                    raise ValueError("locked 512-token manifest contains invalid records")
                eligible_all, context_rejected = context_eligible_records(accepted, max_seq_length)
                eligible_train = _exclude_locked_duplicate_components(
                    eligible_all, [*base_validation, *base_test]
                )
                eligible_train.sort(key=lambda item: sha256_records([{"source_id": item.source_id}]))
                split = {
                    "train": eligible_train,
                    "validation": base_validation,
                    "test": base_test,
                }
                rejected = [*rejected, *context_rejected]
        elif version == "schema_ood":
            full_manifest = _root() / "data" / "full" / "manifest.json"
            if not full_manifest.exists():
                raise ValueError(f"schema_ood requires locked full manifest: {full_manifest}")
            full_payload = json.loads(full_manifest.read_text(encoding="utf-8"))
            full_train, rejected_train = normalize_records(full_payload["splits"]["train"])
            full_validation, rejected_validation = normalize_records(
                full_payload["splits"]["validation"]
            )
            if rejected_train or rejected_validation:
                raise ValueError("locked full train or validation split is invalid")
            test = allocate_schema_ood_test(
                accepted,
                [*full_train, *full_validation],
                target=1000,
            )
            split = {"train": [], "validation": [], "test": test}
        elif version == "full":
            core_manifest = _root() / "data" / "core" / "manifest.json"
            if not core_manifest.exists():
                raise ValueError(f"full requires locked core manifest: {core_manifest}")
            core_payload = json.loads(core_manifest.read_text(encoding="utf-8"))
            core_validation, rejected_validation = normalize_records(
                core_payload["splits"]["validation"]
            )
            core_test, rejected_test = normalize_records(core_payload["splits"]["test"])
            if rejected_validation or rejected_test:
                raise ValueError("locked core validation or test split is invalid")
            candidates = _exclude_locked_duplicate_components(
                accepted, [*core_validation, *core_test]
            )
            full_train = stratified_split_records(candidates, {"train": 20000}, seed=42)["train"]
            split = {"train": full_train, "validation": core_validation, "test": core_test}
        elif len(accepted) < 1750:
            # Tiny fixtures cannot contain the immutable 750/1000 locks. Keep
            # fixture tests useful while real pinned pools use the shared plan.
            split = stratified_split_records(accepted, sizes[version], seed=42)
        else:
            allocation = build_nested_allocation(accepted, seed=42)
            split = allocation[version]
        root = _root()
        ensure_project_dirs(root)
        output_dir = root / "data" / version
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "manifest.json"
        payload: dict[str, Any] = {
            "version": version,
            "seed": 42,
            "filter_version": 1,
            "split_version": 1,
            "prompt_contract_version": 1,
            "revisions": {
                "dataset": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
                "model": data_model.revision,
            },
            "filter": {"version": 1},
            "split": {"version": 1, "seed": 42},
            "prompt": {"version": 1},
            "source": "fixture" if source else "Salesforce/xlam-function-calling-60k",
            "dataset_revision": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
            "model_revision": data_model.revision,
            "hashes": {
                "input_records": sha256_records(records),
                "source": sha256_records(records),
                "uv.lock": sha256_file(root / "uv.lock") if (root / "uv.lock").exists() else "",
                "accepted_records": sha256_records([record.as_dict() for record in accepted]),
                "model_revision_identity": sha256_records(
                    [{"model": data_model.model_id, "revision": data_model.revision}]
                ),
            },
            "flow": source_flow(accepted, rejected),
            "counts": {
                "source": len(records),
                "accepted": len(accepted),
                "rejected": len(rejected),
            },
            "rejection_summary": dict(source_flow(accepted, rejected)),
            "strata": {name: len(values) for name, values in stratify_records(accepted).items()},
            "splits": {
                name: [record.as_dict() for record in values] for name, values in split.items()
            },
            "rejections": [
                {"source_id": item.source_id, "reason": item.reason} for item in rejected
            ],
        }
        if context_base is not None:
            payload = context_base
        if tokenizer is not None and context_base is None:
            from .rendering import tokenizer_fingerprint

            payload["hashes"].update(tokenizer_fingerprint(tokenizer))
            payload["hashes"]["tokenizer_snapshot_revision"] = data_model.revision
        if source is not None and context_base is None:
            payload["hashes"]["source_file"] = sha256_file(source)
        write_json(output, payload)
        context_dir = output_dir / "contexts" / str(max_seq_length)
        context_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = output_dir if context_base is None else context_dir
        for name, values in split.items():
            completion_path = processed_dir / f"{name}.jsonl"
            with completion_path.open("w", encoding="utf-8") as handle:
                for record in values:
                    from .rendering import to_completion_record

                    completion = to_completion_record(record, tokenizer)
                    handle.write(
                        json.dumps(
                            {**completion, "tools": list(record.tools)},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if context_base is None:
                payload["hashes"][f"processed:{name}"] = sha256_file(completion_path)
                payload["hashes"][f"manifest:{name}"] = sha256_records(
                    [record.as_dict() for record in values]
                )
                payload["hashes"][f"split_ids:{name}"] = sha256_records(
                    [{"source_id": record.source_id} for record in values]
                )
        context_hashes: dict[str, str] = {}
        for name, values in split.items():
            context_path = context_dir / f"{name}.jsonl"
            context_path.write_text(
                "".join(
                    json.dumps(
                        {**to_completion_record(record, tokenizer), "tools": list(record.tools)},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                    for record in values
                ),
                encoding="utf-8",
            )
            context_hashes[name] = sha256_file(context_path)
        payload.setdefault("context_views", {})[str(max_seq_length)] = {
            "max_seq_length": max_seq_length,
            "accepted": len(split.get("train", [])),
            "rejected": len(rejected),
            "rejection_summary": dict(source_flow(split.get("train", []), rejected)),
            "counts": {name: len(values) for name, values in split.items()},
            "hashes": context_hashes,
        }
        write_json(output, payload)
        write_json(root / "data" / f"{version}.manifest.json", payload)
        manifest_digest = sha256_file(output)
        (output_dir / "manifest.sha256").write_text(manifest_digest + "\n", encoding="utf-8")
        (root / "data" / f"{version}.manifest.sha256").write_text(
            manifest_digest + "\n", encoding="utf-8"
        )
        typer.echo(f"wrote {output}")
    except Exception as exc:
        _fail(exc)


@data_app.command("audit")
def data_audit(
    version: str = typer.Option(..., help="Prepared manifest version."),
    samples: int = typer.Option(100, min=0, help="Deterministic audit sample size."),
    record_findings: Path | None = typer.Option(
        None, help="Optional aggregate findings JSON. Raw records are never stored."
    ),
) -> None:
    """Write an ignored full-record queue and a tracked aggregate audit."""
    try:
        path = _root() / "data" / version / "manifest.json"
        if not path.exists():
            path = _root() / "data" / f"{version}.manifest.json"
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        records, rejected = normalize_records(
            [raw for values in payload.get("splits", {}).values() for raw in values]
        )
        if rejected:
            raise ValueError(f"manifest contains invalid records: {rejected[0].reason}")
        audit = audit_records(records, version=version, samples=samples)
        queue = audit_queue(records, version=version, samples=samples)
        queue_dir = _root() / "data" / version
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_path = queue_dir / "audit-queue.jsonl"
        queue_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in queue),
            encoding="utf-8",
        )
        audit["queue_count"] = len(queue)
        audit["queue_path"] = f"data/{version}/audit-queue.jsonl"
        if record_findings is not None:
            findings = json.loads(record_findings.read_text(encoding="utf-8"))
            if not isinstance(findings, dict):
                raise ValueError("record findings must be a JSON object of aggregate fields")
            audit["findings"] = {
                str(key): value
                for key, value in findings.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        output = path.parent / "audit.json"
        write_json(output, audit)
        typer.echo(json.dumps(audit, sort_keys=True))
    except Exception as exc:
        _fail(exc)


@data_app.command("validate")
def data_validate(version: str = typer.Option("smoke")) -> None:
    """Validate a prepared manifest without changing it."""
    path = _root() / "data" / version / "manifest.json"
    if not path.exists():
        path = _root() / "data" / f"{version}.manifest.json"
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        sidecar = (
            path.parent / "manifest.sha256"
            if path.name == "manifest.json"
            else path.parent / f"{version}.manifest.sha256"
        )
        data_dir = path.parent if path.name == "manifest.json" else path.parent / version
        expected_sizes: dict[str, dict[str, int | None]] = {
            "smoke": {"train": 128, "validation": 32, "test": 64},
            "day1": {"train": 2000, "validation": 250, "test": 250},
            "core": {"train": 10000, "validation": 750, "test": 1000},
            "full": {"train": 20000, "validation": 750, "test": 1000},
            "schema_ood": {"train": None, "validation": None, "test": 1000},
            "nested_1k": {"train": 1000, "validation": 750, "test": 1000},
            "nested_5k": {"train": 5000, "validation": 750, "test": 1000},
            "nested_10k": {"train": 10000, "validation": 750, "test": 1000},
            "nested_20k": {"train": 20000, "validation": 750, "test": 1000},
            "nested-1k": {"train": 1000, "validation": 750, "test": 1000},
            "nested-5k": {"train": 5000, "validation": 750, "test": 1000},
            "nested-10k": {"train": 10000, "validation": 750, "test": 1000},
            "nested-20k": {"train": 20000, "validation": 750, "test": 1000},
        }
        expected = expected_sizes[version]
        splits = cast(dict[str, list[dict[str, Any]]], payload.get("splits", {}))
        sizes = {name: len(values) for name, values in splits.items()}
        if not splits:
            raise ValueError("manifest has no splits")
        for name, target in expected.items():
            if target is not None and sizes.get(name) != target:
                raise ValueError(f"{name} size {sizes.get(name)} does not match locked {target}")
        if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
            raise ValueError("manifest.sha256 sidecar is missing or does not match")
        normalized: dict[str, list[AcceptedRecord]] = {}
        for name, raw_values in splits.items():
            split_records_value, rejected = normalize_records(raw_values)
            if rejected:
                raise ValueError(f"{name} contains invalid records: {rejected[0].reason}")
            normalized[name] = split_records_value
            expected_hash = payload.get("hashes", {}).get(f"manifest:{name}")
            if not expected_hash:
                raise ValueError(f"{name} manifest hash is missing")
            if expected_hash != sha256_records(
                [record.as_dict() for record in split_records_value]
            ):
                raise ValueError(f"{name} manifest hash does not match")
            expected_ids_hash = payload.get("hashes", {}).get(f"split_ids:{name}")
            actual_ids_hash = sha256_records(
                [{"source_id": record.source_id} for record in split_records_value]
            )
            if expected_ids_hash != actual_ids_hash:
                raise ValueError(f"{name} split ID hash does not match")
            processed = data_dir / f"{name}.jsonl"
            processed_hash = payload.get("hashes", {}).get(f"processed:{name}")
            if not processed.exists() or not processed_hash:
                raise ValueError(f"{name} processed file or hash is missing")
            if processed_hash != sha256_file(processed):
                raise ValueError(f"{name} processed hash does not match")
        flat: list[AcceptedRecord] = []
        flat_names: list[str] = []
        for name, normalized_values in normalized.items():
            flat.extend(normalized_values)
            flat_names.extend([name] * len(normalized_values))
        if len({item.source_id for item in flat}) != len(flat):
            raise ValueError("source record appears in more than one split")
        groups = duplicate_groups(flat)
        for group in groups:
            if len({flat_names[index] for index in group}) > 1:
                raise ValueError("duplicate group leaks across splits")
        if version == "schema_ood":
            full_path = _root() / "data" / "full" / "manifest.json"
            if not full_path.exists():
                raise ValueError("locked full manifest is required for schema-OOD validation")
            full_payload = json.loads(full_path.read_text(encoding="utf-8"))
            full_train, full_train_rejected = normalize_records(full_payload["splits"]["train"])
            full_validation, full_validation_rejected = normalize_records(
                full_payload["splits"]["validation"]
            )
            if full_train_rejected or full_validation_rejected:
                raise ValueError("locked full train or validation split is invalid")
            locked = [*full_train, *full_validation]
            prefixes = set().union(*(item.candidate_prefixes for item in locked))
            fingerprints = set().union(*(item.candidate_fingerprints for item in locked))
            if any(
                item.candidate_prefixes & prefixes or item.candidate_fingerprints & fingerprints
                for item in normalized["test"]
            ):
                raise ValueError("schema-OOD test overlaps train or validation")
        if version == "full":
            core_path = _root() / "data" / "core" / "manifest.json"
            if not core_path.exists():
                raise ValueError("locked core manifest is required")
            core_payload = json.loads(core_path.read_text(encoding="utf-8"))
            names = ("validation", "test") if version == "full" else ("train", "validation")
            for name in names:
                expected_ids = [item.get("source_id") for item in core_payload["splits"][name]]
                actual_ids = [item.source_id for item in normalized[name]]
                if actual_ids != expected_ids:
                    raise ValueError(f"{version} {name} does not match locked core split")
        uv_hash = payload.get("hashes", {}).get("uv.lock")
        if (
            uv_hash
            and (_root() / "uv.lock").exists()
            and sha256_file(_root() / "uv.lock") != uv_hash
        ):
            raise ValueError("uv.lock hash does not match manifest")
        hashes = payload.get("hashes", {})
        required_hashes = {"source", "accepted_records", "uv.lock", "model_revision_identity"}
        required_hashes.update({f"manifest:{name}" for name in expected})
        required_hashes.update({f"processed:{name}" for name in expected})
        required_hashes.update({f"split_ids:{name}" for name in expected})
        if payload.get("source") == "fixture":
            required_hashes.add("source_file")
        if payload.get("source") != "fixture":
            required_hashes.update(
                {"tokenizer_files", "chat_template", "golden_tokens", "tokenizer_snapshot_revision"}
            )
        missing = sorted(name for name in required_hashes if not hashes.get(name))
        if missing:
            raise ValueError(f"manifest is missing required hashes: {missing}")
        for key in ("filter_version", "split_version", "prompt_contract_version"):
            if key not in payload:
                raise ValueError(f"manifest is missing {key}")
        typer.echo(json.dumps({"valid": True, "sizes": sizes}, sort_keys=True))
    except Exception as exc:
        _fail(exc)


@app.command()
def evaluate(
    model: str = typer.Option(..., help="Model identifier or local adapter."),
    split: str = typer.Option("test"),
    manifest: Path = typer.Option(Path("data/smoke/manifest.json")),
    predictions: Path | None = typer.Option(None, help="Local JSON array of model outputs."),
    adapter_path: Path | None = typer.Option(None, help="Optional tuned adapter directory."),
    wandb_online: bool = typer.Option(False, help="Enable verified private W&B."),
    wandb_entity: str | None = typer.Option(None, help="W&B entity for an online run."),
    wandb_project: str = typer.Option("mlx-ft", help="Private W&B project for an online run."),
) -> None:
    """Evaluate predictions against one locked manifest split."""
    run = None
    run_dir: Path | None = None
    contract = None
    try:
        root = _root()
        try:
            model_spec = get_model_spec(model)
            model_revision = model_spec.revision
            registered_model = True
        except ValueError:
            # Fixture-only prediction scoring may use a synthetic model label.
            # Real model work must use a registered pinned identity.
            model_revision = ""
            registered_model = False
        require_free_space(root, 30.0, estimate_gib=0.5)
        if not manifest.exists() and manifest == Path("data/smoke/manifest.json"):
            manifest = root / "data" / "smoke.manifest.json"
        label = "tuned" if adapter_path is not None else "base"
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{label}-{split}"
        ensure_project_dirs(root)
        contract = create_run_contract(
            root,
            run_id,
            kind="evaluation",
            config={
                "model": model,
                "model_revision": model_revision,
                "split": split,
                "workflow": label,
                "manifest": str(manifest),
                "adapter_path": str(adapter_path) if adapter_path else "",
                "decoding": {"max_tokens": 128},
            },
            model_hash=sha256_records([{"model": model, "revision": model_revision}]),
            command=["ftlab", "evaluate", "--model", model, "--split", split],
        )
        run_dir = contract.run_dir
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records, rejected = normalize_records(payload["splits"][split])
        if rejected:
            raise ValueError(f"manifest contains invalid records: {rejected[0].reason}")
        from .artifacts import append_system_metric

        append_system_metric(contract, system_metrics())
        input_hashes = {
            "manifest": sha256_file(manifest),
            "model_identity": sha256_records([{"model": model, "revision": model_revision}]),
            "uv.lock": sha256_file(root / "uv.lock"),
        }
        if adapter_path is not None:
            adapter_file = adapter_path / "adapters.safetensors"
            if not adapter_file.exists():
                raise ValueError(f"adapter file does not exist: {adapter_file}")
            input_hashes["adapter"] = sha256_file(adapter_file)
        write_json(run_dir / "input.hashes.json", input_hashes)
        if wandb_online:
            run = init_tracking(
                online=True,
                entity=wandb_entity,
                project=wandb_project,
                run_name=run_id,
                directory=str(root / "wandb"),
                config={
                    "run_label": f"pipeline validation {label}",
                    "seed": 42,
                    "model_revision": model_revision,
                    "dataset_revision": payload.get("dataset_revision", ""),
                    "split": split,
                    "max_seq_length": 512,
                },
            )
        tokenizer = None
        tokenizer_identity: dict[str, str] = {}
        prompt_lengths = [record.rendered_length or 0 for record in records]
        if payload.get("source") != "fixture":
            from .rendering import load_tokenizer, render_record, tokenizer_fingerprint

            tokenizer = load_tokenizer(model, revision=model_revision, cache_root=cache_root(root))
            tokenizer_identity = tokenizer_fingerprint(tokenizer)
            write_json(run_dir / "tokenizer.hashes.json", tokenizer_identity)
            rendered = [render_record(record, tokenizer) for record in records]
            prompt_lengths = [len(item.prompt_tokens) for item in rendered]
        if predictions is None:
            worker_result = run_isolated(
                contract,
                {
                    "mode": "evaluate",
                    "model": model,
                    "model_revision": model_revision,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "cache_root": str(cache_root(root)),
                    "manifest": str(manifest),
                    "split": split,
                    "predictions_path": str(run_dir / "predictions.jsonl"),
                    "evaluation_path": str(run_dir / "evaluation.json"),
                },
            )
            if worker_result.status != "completed" or worker_result.result is None:
                raise RuntimeError(worker_result.error or "evaluation worker failed")
            performance = worker_result.result
            prediction_rows = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            values = [row["prediction"] for row in prediction_rows]
            prediction_diagnostics = [
                {
                    "truncated": bool(row.get("generation_truncated", False)),
                    "generated_tokens": row.get("generated_tokens"),
                    "max_generation_tokens": row.get("max_generation_tokens"),
                }
                for row in prediction_rows
            ]
            decoding_hash = sha256_records(
                [{"decoding": row.get("decoding", {})} for row in prediction_rows]
            )
        else:
            performance = {}
            values = json.loads(predictions.read_text(encoding="utf-8"))
            prediction_diagnostics = None
            decoding_hash = ""
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("predictions must be a JSON array of strings")
        result = evaluate_predictions(
            records, values, prediction_diagnostics=prediction_diagnostics
        ).as_dict()
        if performance:
            for key in (
                "load_time_seconds",
                "warmup_steps_excluded",
                "latency_p50",
                "latency_p95",
                "latency_p99",
                "prompt_tokens",
                "completion_tokens",
                "throughput_tokens_per_second",
                "throughput_quartiles",
                "mlx_allocator_peak_bytes",
            ):
                if key in performance:
                    result[key] = performance[key]
        if predictions is not None:
            with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for record, value in zip(records, values):
                    handle.write(
                        json.dumps(
                            {"source_id": record.source_id, "prediction": value},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        write_json(run_dir / "predictions.raw.json", values)
        if decoding_hash:
            result["decoding_hash"] = decoding_hash
        write_json(run_dir / "evaluation.json", result)
        rows = []
        for record, value, prompt_tokens in zip(records, values, prompt_lengths):
            one = evaluate_predictions([record], [value], confidence=False)
            parsed = not bool(one.parser_errors)
            error = next(iter(one.parser_errors), "")
            rows.append(
                sanitized_row(
                    run_id,
                    record.source_id,
                    prompt_tokens=prompt_tokens,
                    candidate_count=len(record.tools),
                    parsed=parsed,
                    task_success=one.task_success == 1.0,
                    error_category=error,
                )
            )
        numeric = {key: value for key, value in result.items() if isinstance(value, (int, float))}
        tracking_payload = {
            "config": {
                "run_label": f"pipeline validation {label}",
                "seed": 42,
                "model_revision": model_revision,
                "dataset_revision": payload.get("dataset_revision", ""),
                "split": split,
                "max_seq_length": 512,
            },
            "metrics": numeric,
            "rows": rows,
        }
        write_json(run_dir / "tracking.sanitized.json", tracking_payload)
        dataset_rendering = {
            key: str(payload.get("hashes", {}).get(key, ""))
            for key in ("tokenizer_files", "chat_template", "golden_tokens")
        }
        rendering_matches = bool(tokenizer_identity) and all(
            tokenizer_identity.get(key) == value and value
            for key, value in dataset_rendering.items()
        )
        model_identity_hash = sha256_records([{"model": model, "revision": model_revision}])
        eligibility_manifest = {
            **contract.manifest,
            "hashes": {**contract.manifest.get("hashes", {}), **dataset_rendering},
            "counts": {**contract.manifest.get("counts", {}), "examples": len(records)},
            "dataset_prompt_contract_version": payload.get("prompt_contract_version"),
        }
        controlled = (
            predictions is None
            and controlled_run_eligible(
                eligibility_manifest,
                registered_model=registered_model,
                model_hash=model_identity_hash,
                rendering_hashes=dataset_rendering,
                expected_examples=len(records),
                comparable_rendering=rendering_matches,
            )
            and rendering_matches
            and bool(decoding_hash)
        )
        update_manifest(
            contract,
            model_revision=model_revision,
            controlled=controlled,
            comparable_rendering=rendering_matches,
            rendering_hashes=tokenizer_identity,
            dataset_prompt_contract_version=payload.get("prompt_contract_version"),
            counts={
                "examples": len(records),
                "prompt_tokens": performance.get("prompt_tokens"),
                "target_tokens": performance.get("completion_tokens"),
            },
            hashes={
                "manifest": sha256_file(manifest),
                "predictions": sha256_file(run_dir / "predictions.jsonl"),
                "evaluation": sha256_file(run_dir / "evaluation.json"),
            },
            allocator_peak_bytes=performance.get("mlx_allocator_peak_bytes"),
            decoding_hash=decoding_hash,
        )
        finish_tracking(run, metrics=numeric, rows=rows)
        append_system_metric(contract, system_metrics())
        write_status(contract, "completed", evaluation=result)
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        if run is not None:
            try:
                finish_tracking(run, metrics={"exit_code": 1})
            except Exception:
                pass
        if run_dir is not None:
            if contract is not None:
                write_status(
                    contract,
                    "failed",
                    error_category=type(exc).__name__,
                    error=str(exc),
                )
        _fail(exc)


@app.command()
def train(
    config: Path = typer.Option(..., "--config"),
    wandb_online: bool = typer.Option(False, help="Enable verified private W&B."),
    wandb_entity: str | None = typer.Option(None, help="W&B entity for an online run."),
    wandb_project: str = typer.Option("mlx-ft", help="Private W&B project for an online run."),
) -> None:
    """Run the pinned MLX-LM LoRA training configuration."""
    run = None
    run_dir: Path | None = None
    try:
        experiment = load_config(config)
        root = _root()
        require_free_space(root, experiment.storage.min_free_gib, estimate_gib=1.0)
        ensure_project_dirs(root)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-training"
        contract = create_run_contract(
            root,
            run_id,
            kind="training",
            config=experiment.resolved(),
            model_hash=sha256_records(
                [{"model": experiment.model, "revision": experiment.model_revision}]
            ),
            dataset_hash=sha256_file(root / "data" / experiment.dataset.version / "manifest.json"),
            lock_hash=sha256_file(root / "uv.lock"),
            seed=experiment.training.seed,
            command=["ftlab", "train", "--config", str(config)],
        )
        run_dir = contract.run_dir
        train_file = root / "data" / experiment.dataset.version / "train.jsonl"
        valid_file = root / "data" / experiment.dataset.version / "validation.jsonl"
        manifest_file = root / "data" / experiment.dataset.version / "manifest.json"
        write_json(
            run_dir / "input.hashes.json",
            {
                "manifest": sha256_file(manifest_file),
                "train": sha256_file(train_file),
                "validation": sha256_file(valid_file),
                "uv.lock": sha256_file(root / "uv.lock"),
                "model_identity": sha256_records(
                    [{"model": experiment.model, "revision": experiment.model_revision}]
                ),
            },
        )
        if wandb_online:
            run = init_tracking(
                online=True,
                entity=wandb_entity or experiment.tracking.entity,
                project=wandb_project,
                run_name=run_id,
                directory=str(root / "wandb"),
                config={
                    "run_label": experiment.provenance.label,
                    "seed": experiment.training.seed,
                    "model_revision": experiment.model_revision,
                    "dataset_revision": experiment.dataset.revision,
                    "split": "train",
                    "max_seq_length": experiment.training.max_seq_length,
                },
            )
        worker_result = run_isolated(
            contract,
            {
                "mode": "train",
                "config": experiment.training.model_dump(mode="json"),
                "model_path": experiment.model,
                "model_revision": experiment.model_revision,
                "cache_root": str(cache_root(root)),
                "train_path": str(train_file),
                "valid_path": str(valid_file),
                "adapter_path": str(run_dir / "adapter"),
            },
        )
        if worker_result.status != "completed" or worker_result.result is None:
            raise RuntimeError(worker_result.error or "training worker failed")
        result = worker_result.result
        training_metrics = result.get("training_metrics", [])
        if not isinstance(training_metrics, list) or not training_metrics:
            raise RuntimeError("training worker produced no training metrics")
        result["adapter"] = str(run_dir / "adapter")
        result["adapter_hash"] = sha256_file(run_dir / "adapter" / "adapters.safetensors")
        write_json(run_dir / "training.json", result)
        write_json(run_dir / "training.metrics.json", result.get("training_metrics", []))
        for metric in training_metrics:
            if contract is not None:
                append_training_metric(contract, metric)
        training_series = result.get("training_metrics", [])
        if run is not None:
            for series in training_series:
                metrics = {
                    key: value
                    for key, value in series.items()
                    if key != "iteration" and isinstance(value, (int, float))
                }
                validate_metric_allowlist(metrics)
                if metrics:
                    run.log(metrics)
        summary = {
            "microbatches": result["microbatches"],
            "optimizer_updates": result["optimizer_updates"],
        }
        tracking_payload = {
            "config": {
                "run_label": experiment.provenance.label,
                "seed": experiment.training.seed,
                "model_revision": experiment.model_revision,
                "dataset_revision": experiment.dataset.revision,
                "split": "train",
                "max_seq_length": experiment.training.max_seq_length,
            },
            "metrics": summary,
            "training_series": training_series,
        }
        write_json(run_dir / "tracking.sanitized.json", tracking_payload)
        adapter_file = run_dir / "adapter" / "adapters.safetensors"
        update_manifest(
            contract,
            controlled=False,
            comparable_rendering=False,
            rendering_hashes=result.get("tokenizer_identity", {}),
            trainable_parameters=(
                result.get("trainable_parameters")
                or result.get("manifest", {}).get("trainable_parameters")
            ),
            counts={
                "examples": sum(
                    bool(line.strip())
                    for line in train_file.read_text(encoding="utf-8").splitlines()
                ),
                "updates": result.get("optimizer_updates"),
                "prompt_tokens": result.get("prompt_tokens"),
                "target_tokens": result.get("target_tokens"),
            },
            bytes={
                "adapter": adapter_file.stat().st_size,
                "checkpoint": result.get("checkpoint_bytes"),
            },
            allocator_peak_bytes=result.get("mlx_allocator_peak_bytes"),
            hashes={"adapter": result["adapter_hash"]},
        )
        from .artifacts import append_system_metric

        append_system_metric(contract, system_metrics())
        write_status(contract, "completed", result=summary)
        finish_tracking(run, metrics=summary)
        typer.echo(json.dumps(result, sort_keys=True))
    except Exception as exc:
        if run is not None:
            try:
                finish_tracking(run, metrics={"exit_code": 1})
            except Exception:
                pass
        if run_dir is not None:
            if contract is not None:
                write_status(
                    contract,
                    "failed",
                    error_category=type(exc).__name__,
                    error=str(exc),
                )
        _fail(exc)


@report_app.command("build")
def report_build() -> None:
    """Build the local aggregate table and plot."""
    try:
        typer.echo(json.dumps(build_report(_root()), sort_keys=True))
    except Exception as exc:
        _fail(exc)


@robustness_app.command("prepare")
def robustness_prepare(output: Path = typer.Option(Path("data/robustness.jsonl"))) -> None:
    """Prepare the deterministic 80-call and 20-no-call suite."""
    try:
        typer.echo(str(prepare_robustness(output)))
    except Exception as exc:
        _fail(exc)


@robustness_app.command("evaluate")
def robustness_evaluate(
    model: str = typer.Option(...),
    adapter_path: Path | None = typer.Option(None),
    predictions: Path | None = typer.Option(None),
    cases: Path = typer.Option(Path("data/robustness.jsonl")),
) -> None:
    """Evaluate call success and no-call accuracy as separate metrics."""
    contract = None
    worker_payload: dict[str, Any] = {}
    try:
        root = _root()
        model_spec = get_model_spec(model)
        ensure_project_dirs(root)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-robustness"
        contract = create_run_contract(
            root,
            run_id,
            kind="robustness",
            config={
                "model": model,
                "model_revision": model_spec.revision,
                "adapter_path": str(adapter_path or ""),
                "cases": str(cases),
            },
            command=["ftlab", "robustness", "evaluate", "--model", model],
        )
        if predictions is None:
            worker_result = run_isolated(
                contract,
                {
                    "mode": "robustness",
                    "model": model,
                    "model_revision": model_spec.revision,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                    "cache_root": str(cache_root(root)),
                    "predictions_path": str(contract.path("predictions.jsonl")),
                    "evaluation_path": str(contract.path("evaluation.json")),
                },
            )
            if worker_result.status != "completed" or worker_result.result is None:
                raise RuntimeError(worker_result.error or "robustness worker failed")
            worker_payload = worker_result.result
            rows = [
                json.loads(line)
                for line in contract.path("predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            values = [row["prediction"] for row in rows]
        else:
            values = json.loads(predictions.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("predictions must be a JSON array of strings")
        result = evaluate_robustness(values)
        output = _root() / "reports" / "robustness.json"
        write_json(output, {"model": model, "adapter_path": str(adapter_path or ""), **result})
        if predictions is not None:
            contract.path("predictions.jsonl").write_text(
                "\n".join(json.dumps({"prediction": value}) for value in values) + "\n",
                encoding="utf-8",
            )
        write_json(contract.path("evaluation.json"), result)
        update_manifest(
            contract,
            controlled=False,
            comparable_rendering=False,
            counts={"examples": len(values)},
            hashes={
                "predictions": sha256_file(contract.path("predictions.jsonl")),
                "evaluation": sha256_file(contract.path("evaluation.json")),
            },
            allocator_peak_bytes=worker_payload.get("mlx_allocator_peak_bytes"),
        )
        from .artifacts import append_system_metric

        append_system_metric(contract, system_metrics())
        write_status(contract, "completed", evaluation=result)
        typer.echo(json.dumps(result, sort_keys=True))
    except Exception as exc:
        if contract is not None:
            write_status(contract, "failed", error_category=type(exc).__name__, error=str(exc))
        _fail(exc)


@app.command("sweep")
def sweep_command(
    config: Path = typer.Option(..., "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    run: bool = typer.Option(False, "--run"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Expand or sequentially execute a deterministic sweep matrix."""
    try:
        if dry_run and run:
            raise ValueError("choose either --dry-run or --run")
        expanded = expand_matrix(config)
        if dry_run or not run:
            typer.echo(
                json.dumps(
                    [
                        {
                            "index": item.index,
                            "fingerprint": item.fingerprint,
                            "values": item.values,
                        }
                        for item in expanded
                    ],
                    sort_keys=True,
                )
            )
            return
        plan = load_sweep(config)
        pilot_results: dict[str, list[dict[str, Any]]] = {}
        probe_results: dict[tuple[str, str], dict[str, Any]] = {}
        completed_selections: list[dict[str, Any]] = []
        size_exposure: tuple[int, int] | None = None
        active_contract: Any = None
        previous_path = _root() / "runs" / "sweep.jsonl"
        if resume and previous_path.exists():
            for line in previous_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                previous = json.loads(line)
                selection = previous.get("selection")
                if previous.get("status") == "completed" and isinstance(selection, dict):
                    completed_selections.append(selection)
                    precision = str(selection.get("precision", ""))
                    if selection.get("study_role") == "lr_pilot":
                        pilot_results.setdefault(precision, []).append(selection)

        def execute(one: SweepRun) -> dict[str, Any]:
            nonlocal active_contract, size_exposure
            values = dict(one.values)
            model = str(values["model"])
            spec = get_model_spec(model)
            precision = str(values.get("precision", spec.precision))
            probe_key = (model, precision)
            if probe_key not in probe_results:
                probe_results[probe_key] = _run_sweep_probe(plan, values, root=_root(), spec=spec)
            probe = probe_results[probe_key]
            experiment = _sweep_experiment(plan, values, root=_root())
            run_id = (
                f"sweep-{one.index:02d}-{one.fingerprint[:12]}-"
                f"{str(values.get('study_role', 'run'))}"
            )
            existing_run = _root() / "runs" / run_id
            if existing_run.exists():
                try:
                    existing_status = json.loads(
                        (existing_run / "status.json").read_text(encoding="utf-8")
                    ).get("status")
                except OSError, json.JSONDecodeError:
                    existing_status = None
                if existing_status != "completed":
                    run_id += "-retry-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            contract = create_run_contract(
                _root(),
                run_id,
                kind="training",
                config=experiment.resolved(),
                model_hash=sha256_records(
                    [{"model": experiment.model, "revision": experiment.model_revision}]
                ),
                dataset_hash=sha256_file(
                    _root() / "data" / experiment.dataset.version / "manifest.json"
                ),
                lock_hash=sha256_file(_root() / "uv.lock"),
                seed=experiment.training.seed,
                command=["ftlab", "sweep", "--run", "--config", str(config)],
            )
            active_contract = contract
            train_file, valid_file = _sweep_dataset_paths(
                _root(), experiment, int(values.get("context", 512))
            )
            expected_examples = int(values.get("examples", 0))
            available_examples = (
                sum(
                    bool(line.strip())
                    for line in train_file.read_text(encoding="utf-8").splitlines()
                )
                if train_file.exists()
                else 0
            )
            if expected_examples <= 0 or available_examples < expected_examples:
                raise RuntimeError(
                    f"training data has {available_examples} records; requested {expected_examples}"
                )
            train_input = contract.path("selected-train.jsonl")
            source_lines = train_file.read_text(encoding="utf-8").splitlines()
            train_input.write_text(
                "\n".join(source_lines[:expected_examples]) + "\n", encoding="utf-8"
            )
            worker_result = run_isolated(
                contract,
                {
                    "mode": "train",
                    "config": experiment.training.model_dump(mode="json"),
                    "model_path": experiment.model,
                    "model_revision": experiment.model_revision,
                    "cache_root": str(cache_root(_root())),
                    "train_path": str(train_input),
                    "valid_path": str(valid_file),
                    "adapter_path": str(contract.path("adapter")),
                },
            )
            if worker_result.status != "completed" or worker_result.result is None:
                raise RuntimeError(worker_result.error or "sweep training worker failed")
            worker = worker_result.result
            training_metrics = worker.get("training_metrics", [])
            if not isinstance(training_metrics, list) or not training_metrics:
                raise RuntimeError("training worker produced no training metrics")
            for metric in training_metrics:
                if isinstance(metric, dict):
                    append_training_metric(contract, metric)
            measured_target_tokens = worker.get("target_tokens")
            measured_prompt_tokens = worker.get("prompt_tokens")
            if values.get("study_role") == "size_endpoint":
                if not isinstance(measured_target_tokens, int) or not isinstance(
                    measured_prompt_tokens, int
                ):
                    raise RuntimeError("size endpoint requires measured prompt and target tokens")
                exposure = (measured_prompt_tokens, measured_target_tokens)
                if size_exposure is None:
                    size_exposure = exposure
                elif exposure != size_exposure:
                    raise RuntimeError(
                        "size endpoint exposure differs from the measured reference recipe"
                    )
            adapter = contract.path("adapter") / "adapters.safetensors"
            if not adapter.is_file():
                raise RuntimeError("training worker completed without adapters.safetensors")
            actual_microbatches = int(worker.get("microbatches", -1))
            actual_updates = int(worker.get("optimizer_updates", -1))
            expected_updates = int(values.get("updates", -1))
            expected_microbatches = expected_updates * experiment.training.grad_accumulation_steps
            if actual_microbatches != expected_microbatches or actual_updates != expected_updates:
                raise RuntimeError(
                    "training worker counts do not match the requested microbatch/update arithmetic"
                )
            evaluation_result = run_isolated(
                contract,
                {
                    "mode": "evaluate",
                    "model": experiment.model,
                    "model_revision": experiment.model_revision,
                    "adapter_path": str(adapter),
                    "cache_root": str(cache_root(_root())),
                    "manifest": str(
                        _root() / "data" / experiment.dataset.version / "manifest.json"
                    ),
                    "split": "validation",
                    "predictions_path": str(contract.path("predictions.jsonl")),
                    "evaluation_path": str(contract.path("evaluation.json")),
                },
            )
            if evaluation_result.status != "completed" or evaluation_result.result is None:
                raise RuntimeError(evaluation_result.error or "validation worker failed")
            evaluation_worker = evaluation_result.result
            evaluation = evaluation_worker.get("evaluation", {})
            measured_rss = 0.0
            if contract.path("system_metrics.csv").exists():
                with contract.path("system_metrics.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    measured_rss = max(
                        (float(row.get("rss_gib", 0.0)) for row in csv.DictReader(handle)),
                        default=0.0,
                    )
            required_metrics = ("task_success", "argument_value_f1", "schema_validity")
            if any(not isinstance(evaluation.get(key), (int, float)) for key in required_metrics):
                raise RuntimeError(
                    "validation evaluation did not produce measured selection metrics"
                )
            dataset_payload = json.loads(
                (_root() / "data" / experiment.dataset.version / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            rendering_hashes = {
                key: str(dataset_payload.get("hashes", {}).get(key, ""))
                for key in ("tokenizer_files", "chat_template", "golden_tokens")
            }
            loaded_rendering = evaluation_worker.get("tokenizer_identity", {})
            rendering_matches = bool(loaded_rendering) and all(
                rendering_hashes.get(key) and loaded_rendering.get(key) == rendering_hashes.get(key)
                for key in rendering_hashes
            )
            if isinstance(loaded_rendering, dict):
                rendering_hashes = {
                    key: str(loaded_rendering.get(key, rendering_hashes.get(key, "")))
                    for key in rendering_hashes
                }
            prompt_hashes = [
                json.loads(line).get("prompt_hash")
                for line in contract.path("predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            prompt_hash = sha256_records([{"prompt_hash": value} for value in prompt_hashes])
            effective_values = dict(values)
            effective_values["learning_rate"] = float(
                values.get("learning_rate", experiment.training.learning_rate)
            )
            resolved_fingerprint = sha256_records([effective_values])
            recipe = {
                key: values.get(key)
                for key in (
                    "model",
                    "precision",
                    "dataset_version",
                    "context",
                    "effective_batch",
                    "batch_size",
                    "grad_accumulation_steps",
                    "adapter_rank",
                    "adapter_layers",
                    "adapter_targets",
                    "examples",
                    "updates",
                    "lora_scale",
                )
            }
            evaluation_payload = {
                **evaluation,
                "study_role": values.get("study_role"),
                "axis": values.get("axis"),
                "learning_rate": float(
                    values.get("learning_rate", experiment.training.learning_rate)
                ),
                "rss_gib": measured_rss,
                "safety_probe": probe,
                "load_time_seconds": evaluation_worker.get("load_time_seconds"),
                "warmup_steps_excluded": evaluation_worker.get("warmup_steps_excluded"),
                "latency_p50": evaluation_worker.get("latency_p50"),
                "latency_p95": evaluation_worker.get("latency_p95"),
                "latency_p99": evaluation_worker.get("latency_p99"),
                "prompt_tokens": evaluation_worker.get("prompt_tokens"),
                "completion_tokens": evaluation_worker.get("completion_tokens"),
                "throughput_tokens_per_second": evaluation_worker.get(
                    "throughput_tokens_per_second"
                ),
                "throughput_quartiles": evaluation_worker.get("throughput_quartiles"),
                "prompt_hash": prompt_hash,
                "tokenizer_identity": evaluation_worker.get("tokenizer_identity", {}),
                "rendering_hashes": rendering_hashes,
                "controlled": False,
                "winner_source_run": values.get("winner_source_run"),
            }
            write_json(contract.path("evaluation.json"), evaluation_payload)
            measured_selection = {
                "study_role": values.get("study_role"),
                "axis": values.get("axis"),
                "model": model,
                "precision": precision,
                "learning_rate": float(
                    values.get("learning_rate", experiment.training.learning_rate)
                ),
                "prompt_hash": prompt_hash,
                "rendering_hashes": rendering_hashes,
                "throughput_tokens_per_second": evaluation_worker.get(
                    "throughput_tokens_per_second"
                ),
                "validation": evaluation,
                "rss_gib": measured_rss,
                "run_id": run_id,
                "resolved_fingerprint": resolved_fingerprint,
                "probe": probe,
                "recipe": recipe,
                "winner_source_run": values.get("winner_source_run"),
                "prompt_tokens": measured_prompt_tokens,
                "target_tokens": measured_target_tokens,
            }
            eligibility_manifest = {
                **contract.manifest,
                "hashes": {**contract.manifest.get("hashes", {}), **rendering_hashes},
                "counts": {
                    **contract.manifest.get("counts", {}),
                    "examples": expected_examples,
                    "updates": actual_updates,
                },
                "comparable_rendering": rendering_matches,
                "dataset_prompt_contract_version": dataset_payload.get("prompt_contract_version"),
                "safety_probe": probe,
            }
            controlled = controlled_run_eligible(
                eligibility_manifest,
                registered_model=True,
                model_hash=sha256_records(
                    [{"model": experiment.model, "revision": experiment.model_revision}]
                ),
                rendering_hashes=rendering_hashes,
                expected_examples=expected_examples,
                expected_updates=actual_updates,
                require_safe_probe=True,
                comparable_rendering=rendering_matches,
            )
            update_manifest(
                contract,
                trainable_parameters=(
                    worker.get("trainable_parameters")
                    or worker.get("manifest", {}).get("trainable_parameters")
                ),
                counts={
                    "examples": expected_examples,
                    "updates": actual_updates,
                    "prompt_tokens": worker.get("prompt_tokens"),
                    "target_tokens": worker.get("target_tokens"),
                },
                bytes={"adapter": adapter.stat().st_size},
                hashes={"adapter": sha256_file(adapter)},
                selection=measured_selection,
                study_role=values.get("study_role"),
                axis=values.get("axis"),
                safety_probe=probe,
                rendering_hashes=rendering_hashes,
                prompt_hash=prompt_hash,
                comparable_rendering=rendering_matches,
                controlled=controlled,
                winner_source_run=values.get("winner_source_run"),
            )
            write_status(contract, "completed", result=worker, validation=evaluation_payload)
            selection = {
                "status": "completed",
                "task_success": float(evaluation["task_success"]),
                "argument_value_f1": float(evaluation["argument_value_f1"]),
                "schema_validity": float(evaluation["schema_validity"]),
                "rss_gib": measured_rss,
                "learning_rate": float(
                    values.get("learning_rate", experiment.training.learning_rate)
                ),
                "study_role": values.get("study_role"),
                "model": model,
                "precision": precision,
                "run_id": run_id,
                "throughput_tokens_per_second": evaluation_worker.get(
                    "throughput_tokens_per_second"
                ),
                "prompt_hash": prompt_hash,
                "rendering_hashes": rendering_hashes,
                "resolved_fingerprint": resolved_fingerprint,
                "controlled": controlled,
                "validation": evaluation,
                "probe": probe,
                "recipe": recipe,
                "winner_source_run": values.get("winner_source_run"),
                "prompt_tokens": measured_prompt_tokens,
                "target_tokens": measured_target_tokens,
            }
            completed_selections.append(selection)
            if values.get("study_role") == "lr_pilot":
                pilot_results.setdefault(precision, []).append(selection)
            active_contract = None
            return {
                "status": "completed",
                "run_id": run_id,
                "artifact_dir": run_id,
                "selection": selection,
                "outcome": worker,
            }

        def execute_with_lr(one: SweepRun) -> dict[str, Any]:
            nonlocal active_contract
            try:
                role = one.values.get("study_role")
                if role == "lr_pilot":
                    return execute(one)
                precision = str(one.values.get("precision", "4bit"))
                pilot_precision = precision if pilot_results.get(precision) else "4bit"
                selected = select_best(pilot_results.get(pilot_precision, []))
                if selected is None:
                    raise RuntimeError(
                        f"no completed LR pilot is available for precision {pilot_precision}"
                    )
                resolved = {**one.values, "learning_rate": selected["learning_rate"]}
                if role == "winner_seed":
                    winner = select_best(
                        [
                            item
                            for item in completed_selections
                            if item.get("study_role") != "lr_pilot"
                        ]
                    )
                    if winner is None:
                        raise RuntimeError("no completed validation-selected winner is available")
                    recipe = winner.get("recipe")
                    if not isinstance(recipe, dict):
                        raise RuntimeError("winning run has no resolved recipe")
                    seed = one.values.get("seed")
                    resolved = {**resolved, **recipe, "seed": seed}
                    resolved["learning_rate"] = selected["learning_rate"]
                    resolved["winner_source_run"] = winner.get("run_id")
                one = SweepRun(one.index, resolved, one.fingerprint)
                return execute(one)
            except Exception as exc:
                if active_contract is not None:
                    write_status(
                        active_contract,
                        "failed",
                        error_category=type(exc).__name__,
                        error=str(exc),
                    )
                    active_contract = None
                raise

        results = run_sweep(
            config,
            output=_root() / "runs" / "sweep.jsonl",
            resume=resume,
            execute=execute_with_lr,
        )
        typer.echo(
            json.dumps({"count": len(results), "best": select_best(results)}, sort_keys=True)
        )
    except Exception as exc:
        _fail(exc)


def _sweep_experiment(plan: SweepConfig, values: dict[str, Any], *, root: Path) -> Any:
    """Resolve one explicit study row over the pinned base experiment."""
    if not plan.base_config:
        raise ValueError("sweep --run requires base_config")
    base_path = Path(plan.base_config)
    if not base_path.is_absolute():
        base_path = root / base_path
    experiment = load_config(base_path)
    raw = experiment.resolved()
    raw["model"] = values["model"]
    raw["model_revision"] = get_model_spec(str(values["model"])).revision
    raw.setdefault("model_info", {})["name"] = raw["model"]
    raw["model_info"]["revision"] = raw["model_revision"]
    raw["dataset"]["version"] = values.get("dataset_version", raw["dataset"]["version"])
    raw["training"]["seed"] = int(values.get("seed", raw["training"]["seed"]))
    raw["training"]["learning_rate"] = float(
        values.get("learning_rate", raw["training"]["learning_rate"])
    )
    raw["training"]["batch_size"] = int(values.get("batch_size", raw["training"]["batch_size"]))
    raw["training"]["grad_accumulation_steps"] = int(
        values.get("grad_accumulation_steps", raw["training"]["grad_accumulation_steps"])
    )
    effective_batch = raw["training"]["batch_size"] * raw["training"]["grad_accumulation_steps"]
    if "effective_batch" in values and int(values["effective_batch"]) != effective_batch:
        raise ValueError(
            "sweep effective_batch does not equal batch_size * grad_accumulation_steps"
        )
    updates = int(values.get("updates", raw["training"]["iters"]))
    raw["training"]["iters"] = updates * raw["training"]["grad_accumulation_steps"]
    raw["training"]["num_layers"] = int(values.get("adapter_layers", raw["training"]["num_layers"]))
    raw["training"]["lora_parameters"]["rank"] = int(
        values.get("adapter_rank", raw["training"]["lora_parameters"]["rank"])
    )
    raw["training"]["lora_parameters"]["keys"] = list(
        values.get("adapter_targets", raw["training"]["lora_parameters"]["keys"])
    )
    raw["training"]["lora_parameters"]["scale"] = float(values.get("lora_scale", 20.0))
    raw["training"]["max_seq_length"] = int(
        values.get("context", raw["training"]["max_seq_length"])
    )
    raw["rendering"]["max_seq_length"] = raw["training"]["max_seq_length"]
    return type(experiment).model_validate(raw)


def _sweep_dataset_paths(root: Path, experiment: Any, context: int) -> tuple[Path, Path]:
    dataset_dir = root / "data" / experiment.dataset.version
    if context == 512:
        return dataset_dir / "train.jsonl", dataset_dir / "validation.jsonl"
    context_dir = dataset_dir / "contexts" / str(context)
    train_file = context_dir / "train.jsonl"
    valid_file = context_dir / "validation.jsonl"
    if not train_file.is_file() or not valid_file.is_file():
        raise ValueError(f"context {context} requires prepared eligibility view: {context_dir}")
    return train_file, valid_file


def _run_sweep_probe(
    plan: SweepConfig, values: dict[str, Any], *, root: Path, spec: Any
) -> dict[str, Any]:
    experiment = _sweep_experiment(plan, values, root=root)
    train_file, valid_file = _sweep_dataset_paths(root, experiment, int(values.get("context", 512)))
    if not train_file.is_file() or not valid_file.is_file():
        raise ValueError(f"probe requires prepared dataset files for {experiment.dataset.version}")
    run_id = f"probe-{spec.precision}-{spec.model_id.split('/')[-1]}"
    identity_hash = sha256_records([{"model": spec.model_id, "revision": spec.revision}])
    probe_fingerprint = sha256_records(
        [{"model": spec.model_id, "revision": spec.revision, "precision": spec.precision}]
    )
    existing = root / "runs" / run_id
    if existing.exists():
        existing_manifest = json.loads((existing / "manifest.json").read_text(encoding="utf-8"))
        existing_status = json.loads((existing / "status.json").read_text(encoding="utf-8"))
        safe = existing_manifest.get("safety_probe", {})
        if (
            existing_status.get("status") == "completed"
            and safe.get("status") == "safe"
            and existing_manifest.get("hashes", {}).get("model") == identity_hash
            and safe.get("fingerprint") == probe_fingerprint
        ):
            return {**safe, "model_hash": identity_hash, "fingerprint": probe_fingerprint}
        raise RuntimeError(f"existing probe {run_id} is not a reusable safe matching probe")
    try:
        contract = create_run_contract(
            root,
            run_id,
            kind="benchmark",
            config=experiment.resolved(),
            model_hash=identity_hash,
            command=["ftlab", "sweep", "probe", spec.model_id, spec.precision],
        )
    except FileExistsError:
        if existing.is_dir():
            existing_manifest = json.loads((existing / "manifest.json").read_text(encoding="utf-8"))
            existing_safety = existing_manifest.get("safety_probe", {})
            if (
                existing_safety.get("status") == "safe"
                and existing_safety.get("model_hash") == identity_hash
                and existing_safety.get("fingerprint") == probe_fingerprint
            ):
                return {
                    **existing_safety,
                    "model_hash": identity_hash,
                    "fingerprint": probe_fingerprint,
                }
        raise RuntimeError(f"existing probe {run_id} is not reusable")
    result = run_isolated(
        contract,
        {
            "mode": "probe",
            "model": experiment.model,
            "model_revision": experiment.model_revision,
            "cache_root": str(cache_root(root)),
            "steps": 32,
            "config": experiment.training.model_dump(mode="json"),
            "train_path": str(train_file),
            "valid_path": str(valid_file),
            "adapter_path": str(contract.path("adapter")),
        },
    )
    import csv

    with contract.path("system_metrics.csv").open(newline="", encoding="utf-8") as handle:
        samples = [dict(row) for row in csv.DictReader(handle)]
    for sample in samples:
        for key in ("memory_available_gib", "swap_used_gib", "disk_free_gib", "rss_gib"):
            if key in sample:
                try:
                    sample[key] = float(sample[key])
                except ValueError:
                    pass
    safety = classify_probe_safety(
        samples, returncode=result.returncode, worker_result=result.result
    )
    safety = {**safety, "model_hash": identity_hash, "fingerprint": probe_fingerprint}
    update_manifest(contract, safety_probe=safety)
    write_status(
        contract,
        "completed"
        if result.status == "completed" and safety["status"] == "safe"
        else "infeasible",
        safety=safety,
    )
    if result.status != "completed" or safety["status"] != "safe":
        raise RuntimeError(result.error or "sweep safety probe is infeasible")
    return safety


@benchmark_app.command("probe")
def benchmark_probe(
    config: Path = typer.Option(..., "--config"),
    steps: int = typer.Option(32, "--steps", min=1),
) -> None:
    """Run a bounded child probe and classify resource safety."""
    contract = None
    try:
        experiment = load_config(config)
        root = _root()
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-probe"
        contract = create_run_contract(
            root,
            run_id,
            kind="benchmark",
            config=experiment.resolved(),
            seed=experiment.training.seed,
            command=["ftlab", "benchmark", "probe", "--config", str(config)],
        )
        request = {
            "mode": "probe",
            "model": experiment.model,
            "model_revision": experiment.model_revision,
            "cache_root": str(cache_root(root)),
            "steps": steps,
            "config": {
                **experiment.training.model_dump(mode="json"),
                "iters": steps,
            },
            "train_path": str(root / "data" / experiment.dataset.version / "train.jsonl"),
            "valid_path": str(root / "data" / experiment.dataset.version / "validation.jsonl"),
            "adapter_path": str(contract.run_dir / "adapter"),
        }
        result = run_isolated(contract, request)
        import csv

        with contract.path("system_metrics.csv").open(newline="", encoding="utf-8") as handle:
            samples = [dict(row) for row in csv.DictReader(handle)]
        for sample in samples:
            for key in ("memory_available_gib", "swap_used_gib", "disk_free_gib", "rss_gib"):
                if key in sample:
                    try:
                        sample[key] = float(sample[key])
                    except ValueError:
                        pass
        safety = classify_probe_safety(
            samples,
            returncode=result.returncode,
            worker_result=result.result,
        )
        infeasible = safety["status"] != "safe" or result.status != "completed"
        update_manifest(contract, safety_probe=safety)
        write_status(contract, "infeasible" if infeasible else "completed", safety=safety)
        typer.echo(
            json.dumps(
                {"status": "infeasible" if infeasible else "safe", "safety": safety}, sort_keys=True
            )
        )
    except Exception as exc:
        if contract is not None:
            write_status(contract, "failed", error_category=type(exc).__name__, error=str(exc))
        _fail(exc)


@bfcl_app.command("prepare")
def bfcl_prepare_command(
    source: Path | None = typer.Option(None),
    output: Path = typer.Option(Path("data/bfcl.manifest.json")),
) -> None:
    try:
        typer.echo(str(prepare_bfcl(source, output)))
    except Exception as exc:
        _fail(exc)


@bfcl_app.command("export")
def bfcl_export_command(
    run: Path = typer.Option(..., "--run"), output: Path | None = typer.Option(None)
) -> None:
    try:
        run_path = run
        if not run_path.exists():
            run_path = _root() / "runs" / run
        if not run_path.exists():
            raise ValueError(f"run does not exist: {run}")
        rows = [
            json.loads(line)
            for line in (run_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest_path = run_path / "bfcl.manifest.json"
        if not manifest_path.exists():
            manifest_path = _root() / "data" / "bfcl.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_ids = {str(row["id"]) for row in manifest.get("rows", [])}
        row_ids = [str(row.get("id")) for row in rows]
        if any(case_id not in selected_ids for case_id in row_ids):
            raise ValueError("predictions contain IDs outside selected BFCL cases")
        if len(row_ids) != len(set(row_ids)) or set(row_ids) != selected_ids:
            raise ValueError("predictions must contain each selected BFCL case exactly once")
        target = output or run_path / "bfcl.results.jsonl"
        typer.echo(str(export_bfcl_responses(rows, target, run_id=run_path.name)))
    except Exception as exc:
        _fail(exc)


@bfcl_app.command("evaluate")
def bfcl_evaluate_command(run: Path = typer.Option(..., "--run")) -> None:
    try:
        run_path = run if run.is_absolute() else _root() / "runs" / run
        if not run_path.exists():
            raise ValueError(f"run does not exist: {run}")
        prediction_run = run_path
        base_manifest_path = run_path / "manifest.json"
        base_kind = ""
        if base_manifest_path.exists():
            base_kind = str(
                json.loads(base_manifest_path.read_text(encoding="utf-8")).get("kind", "")
            )
        if not (run_path / "predictions.jsonl").exists() or base_kind != "bfcl":
            base_manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
            config = base_manifest.get("config", {})
            model = str(config.get("model", ""))
            revision = str(config.get("model_revision", ""))
            spec = get_model_spec(model, revision)
            selected_manifest = _root() / "data" / "bfcl.manifest.json"
            if not selected_manifest.exists():
                raise ValueError("BFCL selected-case manifest is required before generation")
            selected = json.loads(selected_manifest.read_text(encoding="utf-8"))
            records_path = selected_manifest.parent / str(selected.get("records_file", ""))
            if not records_path.exists():
                raise ValueError("BFCL selected records file is missing")
            subrun_id = f"{run_path.name}-bfcl"
            prediction_run = _root() / "runs" / subrun_id
            reuse_prediction_run = False
            if prediction_run.exists():
                status = json.loads((prediction_run / "status.json").read_text(encoding="utf-8"))
                if (
                    status.get("status") == "completed"
                    and (prediction_run / "predictions.jsonl").exists()
                ):
                    reuse_prediction_run = True
                else:
                    subrun_id += "-retry-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
                    prediction_run = _root() / "runs" / subrun_id
            if not reuse_prediction_run:
                contract = create_run_contract(
                    _root(),
                    subrun_id,
                    kind="bfcl",
                    config={
                        "model": spec.model_id,
                        "model_revision": spec.revision,
                        "parent_run": run_path.name,
                        "bfcl_manifest": str(selected_manifest),
                    },
                    model_hash=sha256_records(
                        [{"model": spec.model_id, "revision": spec.revision}]
                    ),
                    command=["ftlab", "bfcl", "evaluate", "--run", run_path.name],
                )
                adapter = run_path / "adapter"
                result = run_isolated(
                    contract,
                    {
                        "mode": "bfcl",
                        "model": spec.model_id,
                        "model_revision": spec.revision,
                        "adapter_path": str(adapter) if adapter.exists() else None,
                        "cache_root": str(cache_root(_root())),
                        "records_path": str(records_path),
                        "predictions_path": str(contract.path("predictions.jsonl")),
                    },
                )
                if result.status != "completed":
                    raise RuntimeError(result.error or "BFCL generation worker failed")
                write_json(contract.path("evaluation.json"), result.result or {})
                update_manifest(contract, controlled=False, comparable_rendering=False)
                write_status(contract, "completed", result=result.result or {})
        completed = evaluate_bfcl(prediction_run, project_root=_root())
        typer.echo("\n".join(item.stdout for item in completed))
    except Exception as exc:
        _fail(exc)


@storage_app.command("status")
def storage_status_command() -> None:
    """Show project-local storage state."""
    try:
        typer.echo(json.dumps(storage_status(_root()), indent=2, sort_keys=True))
    except Exception as exc:
        _fail(exc)


@app.command()
def clean(
    dry_run: bool = typer.Option(
        True, "--dry-run/--delete", help="List artifacts only unless explicitly disabled."
    ),
) -> None:
    """List local artifacts. Deletion is intentionally not implemented in version 1."""
    if not dry_run:
        _fail(ValueError("cleanup requires a later explicit approval; use --dry-run"))
    for path in cleanup_candidates(_root()):
        typer.echo(str(path))


if __name__ == "__main__":
    app()
