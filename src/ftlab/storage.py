"""Project-local artifact paths and disk safety checks."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .exceptions import FtlabError

LOCAL_DIRS = ("data", "models", "runs", "predictions", "adapters", "wandb", "reports", "artifacts")


def project_dirs(root: str | Path) -> dict[str, Path]:
    base = Path(root).resolve()
    return {name: base / name for name in LOCAL_DIRS}


def cache_root(root: str | Path) -> Path:
    path = Path(root).resolve() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def system_metrics() -> dict[str, float | str]:
    try:
        import psutil
    except ImportError:
        return {}
    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
    except OSError:
        # Sandboxed CI can deny swap counters. Keep the row usable.
        swap = type("Swap", (), {"used": 0, "percent": 0.0})()
    free = shutil.disk_usage(Path.cwd()).free / (1024**3)
    result: dict[str, float | str] = {
        "memory_available_gib": memory.available / (1024**3),
        "memory_used_percent": float(memory.percent),
        "cpu_percent": float(psutil.cpu_percent(interval=None)),
        "swap_used_gib": swap.used / (1024**3),
        "swap_percent": float(swap.percent),
        "disk_free_gib": free,
        "pressure_state": pressure_state(),
    }
    result.update(allocator_metrics())
    return result


def allocator_metrics() -> dict[str, float]:
    """Read MLX allocator counters when MLX is available in this process."""
    try:
        import mlx.core as mx
    except ImportError:
        return {}
    result: dict[str, float] = {}
    for key, name in (
        ("mlx_allocator_active_bytes", "get_active_memory"),
        ("mlx_allocator_peak_bytes", "get_peak_memory"),
    ):
        getter = getattr(mx, name, None)
        if callable(getter):
            try:
                result[key] = float(getter())
            except TypeError, ValueError:
                pass
    return result


def pressure_state() -> str:
    """Return a portable pressure label without requiring Apple-only APIs."""
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            check=False,
            capture_output=True,
            text=True,
        )
        level = result.stdout.strip()
        return {"1": "normal", "2": "warning", "4": "critical"}.get(level, "unknown")
    except OSError:
        return "unknown"


def process_metrics(
    pid: int | None = None, root: str | Path | None = None
) -> dict[str, float | int | str]:
    """Capture child RSS plus system safety values."""
    result: dict[str, float | int | str] = {
        "timestamp": time.time(),
        **system_metrics(),
    }
    if pid is not None:
        try:
            import psutil

            process = psutil.Process(pid)
            result["pid"] = pid
            result["rss_gib"] = process.memory_info().rss / (1024**3)
        except ImportError, psutil.Error:
            result["pid"] = pid
            result["rss_gib"] = 0.0
    if root is not None:
        result["disk_free_gib"] = shutil.disk_usage(Path(root).resolve()).free / (1024**3)
    return result


@contextmanager
def exclusive_project_lock(root: str | Path) -> Iterator[Path]:
    """Hold one cross-process lock for all model jobs."""
    lock_path = Path(root).resolve() / "runs" / ".ftlab.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FtlabError("another ftlab model job is already running") from exc
        yield lock_path
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError, OSError:
            pass
        handle.close()


def ensure_project_dirs(root: str | Path) -> dict[str, Path]:
    paths = project_dirs(root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def free_gib(path: str | Path) -> float:
    return shutil.disk_usage(Path(path).resolve()).free / (1024**3)


def require_free_space(
    root: str | Path, minimum_gib: float = 30.0, estimate_gib: float = 0.0
) -> None:
    available = free_gib(root)
    if available - estimate_gib < minimum_gib:
        raise FtlabError(
            f"free disk space {available:.2f} GiB is below required floor "
            f"{minimum_gib:.2f} GiB after estimated {estimate_gib:.2f} GiB"
        )


def storage_status(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    paths = project_dirs(base)
    return {
        "root": str(base),
        "free_gib": free_gib(base),
        "directories": {name: path.exists() for name, path in paths.items()},
    }


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise ValueError("JSON output contains a non-finite float") from exc
    target.write_text(encoded + "\n", encoding="utf-8")
    return target


def start_run(root: str | Path, run_id: str, config: dict[str, Any]) -> Path:
    run_dir = project_dirs(root)["runs"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.resolved.json", config)
    write_json(
        run_dir / "status.json", {"status": "started", "started_at": datetime.now(UTC).isoformat()}
    )
    return run_dir


def finish_run(run_dir: str | Path, status: str, **extra: Any) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("status must be completed or failed")
    payload = {"status": status, "finished_at": datetime.now(UTC).isoformat(), **extra}
    write_json(Path(run_dir) / "status.json", payload)


def cleanup_candidates(root: str | Path) -> list[Path]:
    """List local artifacts without deleting anything."""
    base = Path(root).resolve()
    return [path for path in (base / name for name in LOCAL_DIRS) if path.exists()]
