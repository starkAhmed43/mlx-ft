from __future__ import annotations

from ftlab.artifacts import create_run_contract
from ftlab.runner import classify_probe_safety, run_command_isolated


def test_runner_writes_samples_and_status(tmp_path) -> None:
    run = create_run_contract(tmp_path, "run", kind="benchmark", config={})
    result = run_command_isolated(run, ["python", "-c", "print('ok')"], interval=0.01)
    assert result.status == "completed"
    assert (run.run_dir / "system_metrics.csv").read_text()


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
