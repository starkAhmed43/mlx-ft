"""W&B privacy boundary. Offline remains the code default."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from typing import Any

from .exceptions import ExternalDependencyError, PrivacyError

ALLOWED_CONFIG = {
    "run_label",
    "seed",
    "model_revision",
    "dataset_revision",
    "split",
    "max_seq_length",
    "prompt_contract_version",
}
ALLOWED_ROW = {
    "row_hash",
    "prompt_tokens",
    "candidate_count",
    "parsed",
    "task_success",
    "error_category",
}
ALLOWED_METRICS = {
    "count",
    "tool_accuracy",
    "json_validity",
    "schema_validity",
    "argument_key_f1",
    "argument_value_f1",
    "exact_match",
    "task_success",
    "extra_argument_rate",
    "extra_argument_count",
    "bootstrap_low",
    "bootstrap_high",
    "sample_std",
    "microbatches",
    "optimizer_updates",
    "train_loss",
    "val_loss",
    "learning_rate",
    "iterations_per_second",
    "tokens_per_second",
    "trained_tokens",
    "peak_memory",
    "val_time",
    "exit_code",
}
FORBIDDEN_PARTS = {
    "query",
    "schema",
    "function",
    "function_name",
    "source_id",
    "arguments",
    "output",
    "log",
    "path",
    "dataset",
    "adapter",
    "checkpoint",
    "directory",
    "prompt",
    "completion",
}


def _forbidden(key: str) -> bool:
    if key in ALLOWED_CONFIG or key in ALLOWED_ROW:
        return False
    lowered = key.lower()
    return any(part in lowered for part in FORBIDDEN_PARTS)


def validate_upload_allowlist(payload: dict[str, Any], *, row: bool = False) -> None:
    allowed = ALLOWED_ROW if row else ALLOWED_CONFIG
    unknown = [key for key in payload if key not in allowed or _forbidden(key)]
    if unknown:
        raise PrivacyError(f"W&B upload contains disallowed fields: {sorted(unknown)}")
    for key, value in payload.items():
        if isinstance(value, (dict, list, tuple)):
            raise PrivacyError(f"W&B upload field {key!r} must be scalar")
        if isinstance(value, str) and len(value) > 256:
            raise PrivacyError(f"W&B upload field {key!r} is too long")


def validate_metric_allowlist(metrics: dict[str, Any]) -> None:
    unknown = set(metrics) - ALLOWED_METRICS
    if unknown:
        raise PrivacyError(f"W&B metric names are not allowlisted: {sorted(unknown)}")


def row_hash(run_id: str, source_id: str | int) -> str:
    return hashlib.sha256(f"{run_id}:{source_id}".encode()).hexdigest()[:24]


def sanitized_row(
    run_id: str,
    source_id: str | int,
    *,
    prompt_tokens: int,
    candidate_count: int,
    parsed: bool,
    task_success: bool,
    error_category: str | None = None,
) -> dict[str, Any]:
    row = {
        "row_hash": row_hash(run_id, source_id),
        "prompt_tokens": int(prompt_tokens),
        "candidate_count": int(candidate_count),
        "parsed": bool(parsed),
        "task_success": bool(task_success),
        "error_category": error_category or "",
    }
    validate_upload_allowlist(row, row=True)
    return row


def verify_private_project(api: Any, entity: str, project: str) -> None:
    """Verify W&B access through the SDK GraphQL service."""
    service = getattr(api, "_service_api", None)
    execute = getattr(service, "execute_graphql", None)
    if not callable(execute):
        raise PrivacyError("W&B SDK GraphQL service is unavailable")
    query = """
    query ProjectPrivacy($name: String!, $entity: String!) {
      project(name: $name, entityName: $entity) { id name entityName access }
    }
    """
    try:
        data = execute(query, {"name": project, "entity": entity})
    except Exception as exc:
        raise PrivacyError("could not verify W&B project privacy") from exc
    remote = data.get("project") if isinstance(data, dict) else None
    if not isinstance(remote, dict) or not remote.get("id") or remote.get("access") != "PRIVATE":
        raise PrivacyError(f"W&B project {entity}/{project} must be private")


def init_tracking(
    *,
    online: bool,
    entity: str | None,
    project: str = "mlx-ft",
    run_name: str | None = None,
    config: dict[str, Any] | None = None,
    directory: str | None = None,
) -> Any:
    """Initialize W&B only after privacy checks. Return None in offline mode."""
    if not online:
        return None
    if not entity or not project:
        raise PrivacyError("online W&B runs require explicit entity and project")
    safe_config = config or {}
    validate_upload_allowlist(safe_config)
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    try:
        import wandb
    except ImportError as exc:
        raise ExternalDependencyError("wandb is required for online tracking") from exc
    settings = wandb.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        save_code=False,
        x_disable_meta=True,
        x_disable_machine_info=True,
        x_disable_stats=True,
        disable_job_creation=True,
        x_save_requirements=False,
    )
    try:
        api = wandb.Api()
        viewer = api.viewer
        if viewer is None:
            raise PrivacyError("W&B authentication is required for online tracking")
        verify_private_project(api, entity, project)
    except PrivacyError:
        raise
    except Exception as exc:
        raise PrivacyError("could not verify W&B authentication and project privacy") from exc
    return wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=safe_config,
        save_code=False,
        settings=settings,
        dir=directory,
    )


def log_evaluation(
    run: Any, metrics: dict[str, float], rows: Iterable[dict[str, Any]] = ()
) -> None:
    """Log only allowlisted aggregate metrics and sanitized rows."""
    if run is None:
        return
    row_values = list(rows)
    for row in row_values:
        validate_upload_allowlist(row, row=True)
    validate_metric_allowlist(metrics)
    payload: dict[str, Any] = {key: float(value) for key, value in metrics.items()}
    if row_values:
        table = _make_table(row_values)
        if table is not None:
            payload["predictions"] = table
    run.log(payload)


def _make_table(rows: list[dict[str, Any]]) -> Any | None:
    """Create a W&B table without importing W&B in offline or fixture paths."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb.Table(
        columns=sorted(ALLOWED_ROW),
        data=[[row[column] for column in sorted(ALLOWED_ROW)] for row in rows],
    )


def finish_tracking(
    run: Any, *, metrics: dict[str, float] | None = None, rows: Iterable[dict[str, Any]] = ()
) -> None:
    if run is None:
        return
    row_values = list(rows)
    for row in row_values:
        validate_upload_allowlist(row, row=True)
    try:
        if metrics:
            validate_metric_allowlist(metrics)
        payload: dict[str, Any] = {key: float(value) for key, value in (metrics or {}).items()}
        if row_values:
            table = _make_table(row_values)
            if table is not None:
                payload["predictions"] = table
        if payload:
            run.log(payload)
    finally:
        run.finish()
