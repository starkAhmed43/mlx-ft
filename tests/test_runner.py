from __future__ import annotations

import json

from ftlab.artifacts import create_run_contract
from ftlab.runner import _stop_process, classify_probe_safety, run_command_isolated


def test_runner_writes_samples_and_status(tmp_path) -> None:
    run = create_run_contract(tmp_path, "run", kind="benchmark", config={})
    result = run_command_isolated(run, ["python", "-c", "print('ok')"], interval=0.01)
    assert result.status == "completed"
    assert (run.run_dir / "system_metrics.csv").read_text()
    manifest = json.loads((run.run_dir / "manifest.json").read_text())
    assert manifest["command"] == ["ftlab"]
    assert manifest["worker_command"] == ["python", "-c", "print('ok')"]
    assert manifest["system_summary"]["samples"] >= 1


def test_stop_process_kills_a_stubborn_child_after_term_timeout() -> None:
    class StubbornProcess:
        def __init__(self) -> None:
            self.signals: list[int] = []
            self.killed = False
            self.waits = 0

        def send_signal(self, value: int) -> None:
            self.signals.append(value)

        def wait(self, *, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                from subprocess import TimeoutExpired

                raise TimeoutExpired("worker", timeout)
            return -9

        def kill(self) -> None:
            self.killed = True

    process = StubbornProcess()
    assert _stop_process(process) == -9  # type: ignore[arg-type]
    assert process.signals
    assert process.killed


def test_probe_classifier_requires_three_consecutive_samples() -> None:
    samples = [
        {
            "memory_available_gib": 0.5,
            "swap_used_gib": 1.0,
            "disk_free_gib": 40.0,
            "pressure_state": "critical",
        },
        {
            "memory_available_gib": 0.5,
            "swap_used_gib": 2.0,
            "disk_free_gib": 40.0,
            "pressure_state": "critical",
        },
        {
            "memory_available_gib": 2.0,
            "swap_used_gib": 4.0,
            "disk_free_gib": 40.0,
            "pressure_state": "normal",
        },
        {
            "memory_available_gib": 0.5,
            "swap_used_gib": 4.0,
            "disk_free_gib": 29.0,
            "pressure_state": "critical",
        },
    ]
    safety = classify_probe_safety(samples)
    assert safety["available_memory"] is False
    assert safety["critical_pressure"] is False
    assert safety["swap_growth"] is True
    assert safety["projected_disk"] is True
