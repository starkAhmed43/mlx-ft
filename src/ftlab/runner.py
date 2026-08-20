"""Parent-side process isolation, locking, and one-second monitoring."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import RunContract, append_system_metric, update_manifest, write_status
from .storage import exclusive_project_lock, process_metrics


@dataclass(frozen=True)
class RunnerResult:
    returncode: int
    signal: int | None
    status: str
    result: dict[str, Any] | None
    error: str | None
    samples: int


def classify_probe_safety(
    samples: list[dict[str, Any]],
    *,
    returncode: int = 0,
    worker_result: dict[str, Any] | None = None,
    projected_free_disk_gib: float | None = None,
) -> dict[str, Any]:
    """Classify the one-second samples collected during a bounded probe."""

    def three_consecutive(predicate: Any) -> bool:
        streak = 0
        for sample in samples:
            if predicate(sample):
                streak += 1
                if streak >= 3:
                    return True
            else:
                streak = 0
        return False

    def finite(sample: dict[str, Any]) -> bool:
        for key in ("memory_available_gib", "swap_used_gib", "disk_free_gib", "rss_gib"):
            value = sample.get(key)
            if value is not None:
                try:
                    if not math.isfinite(float(value)):
                        return False
                except TypeError, ValueError:
                    return False
        return True

    swap = [float(item["swap_used_gib"]) for item in samples if "swap_used_gib" in item]
    disk_values = [float(item["disk_free_gib"]) for item in samples if "disk_free_gib" in item]
    projected = projected_free_disk_gib
    if projected is None and disk_values:
        projected = min(disk_values)
    safety = {
        "sample_count": len(samples),
        "available_memory": three_consecutive(
            lambda item: float(item.get("memory_available_gib", math.inf)) < 1.0
        ),
        "critical_pressure": three_consecutive(
            lambda item: item.get("pressure_state") == "critical"
        ),
        "swap_growth": bool(swap) and max(swap) - min(swap) > 2.0,
        "aborted_or_killed": returncode != 0,
        "nonfinite": any(not finite(item) for item in samples)
        or bool(worker_result and worker_result.get("nonfinite")),
        "projected_disk": projected is not None and projected < 30.0,
    }
    return {
        "status": "infeasible" if any(safety.values()) else "safe",
        "projected_free_disk_gib": projected,
        **safety,
    }


def _request_path(run: RunContract, request: dict[str, Any]) -> Path:
    path = run.path("worker.request.json")
    request = {**request, "result_path": str(run.path("worker.result.json"))}
    path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_isolated(
    run: RunContract,
    request: dict[str, Any],
    *,
    interval: float = 1.0,
    timeout: float | None = None,
) -> RunnerResult:
    """Run one worker under the project lock and always persist final status."""
    root = run.root
    request_path = _request_path(run, request)
    command = [sys.executable, "-m", "ftlab.worker", "--request", str(request_path)]
    update_manifest(run, command=command, worker_started_at=time.time())
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    samples = 0
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = 1
    error: str | None = None
    result_payload: dict[str, Any] | None = None
    try:
        with exclusive_project_lock(root):
            with run.path("train.log").open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while True:
                    returncode = process.poll()
                    row = process_metrics(process.pid, root)
                    append_system_metric(run, row)
                    samples += 1
                    if returncode is not None:
                        break
                    if timeout is not None and time.monotonic() - started >= timeout:
                        process.terminate()
                        error = "worker timed out"
                        returncode = process.wait()
                        break
                    time.sleep(max(0.0, interval))
        result_path = run.path("worker.result.json")
        if result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        if returncode == 0 and result_payload and result_payload.get("status") == "completed":
            status = "completed"
            update_manifest(run, worker=result_payload.get("result", {}))
        else:
            status = "failed"
            if result_payload:
                error = error or result_payload.get("error", {}).get("message")
            write_status(
                run,
                status,
                returncode=returncode,
                signal=-returncode if returncode is not None and returncode < 0 else None,
                error=error or "worker failed",
            )
        if status == "completed":
            write_status(
                run,
                status,
                returncode=returncode,
                signal=-returncode if returncode is not None and returncode < 0 else None,
            )
        update_manifest(run, worker_finished_at=time.time())
        final_code = returncode if returncode is not None else 1
        return RunnerResult(
            returncode=final_code,
            signal=-final_code if final_code < 0 else None,
            status=status,
            result=result_payload.get("result") if result_payload else None,
            error=error,
            samples=samples,
        )
    except BaseException as exc:
        error = str(exc)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        write_status(
            run,
            "failed",
            returncode=returncode,
            signal=-returncode if returncode is not None and returncode < 0 else None,
            error=error,
        )
        raise


def run_command_isolated(
    run: RunContract,
    command: list[str],
    *,
    interval: float = 1.0,
    timeout: float | None = None,
) -> RunnerResult:
    """Run a test or benchmark command under the same lock and monitor."""
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    samples = 0
    returncode: int | None = 1
    error: str | None = None
    try:
        update_manifest(run, command=command, worker_started_at=time.time())
        with exclusive_project_lock(run.root):
            with run.path("train.log").open("ab") as log:
                process = subprocess.Popen(
                    command, cwd=run.root, stdout=log, stderr=subprocess.STDOUT
                )
                while True:
                    returncode = process.poll()
                    append_system_metric(run, process_metrics(process.pid, run.root))
                    samples += 1
                    if returncode is not None:
                        break
                    if timeout is not None and time.monotonic() - started >= timeout:
                        process.kill()
                        error = "command timed out"
                        returncode = process.wait()
                        break
                    time.sleep(max(interval, 0.0))
        status = "completed" if returncode == 0 else "failed"
        write_status(
            run,
            status,
            returncode=returncode,
            signal=-returncode if returncode is not None and returncode < 0 else None,
            error=error,
        )
        update_manifest(run, worker_finished_at=time.time())
        final_code = returncode if returncode is not None else 1
        return RunnerResult(
            final_code, -final_code if final_code < 0 else None, status, None, error, samples
        )
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait()
        write_status(
            run,
            "failed",
            returncode=returncode,
            signal=-returncode if returncode is not None and returncode < 0 else None,
            error=str(exc),
        )
        raise
