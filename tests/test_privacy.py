from __future__ import annotations

import os
import sys

import pytest

import ftlab.tracking as tracking
from ftlab.exceptions import PrivacyError
from ftlab.tracking import (
    finish_tracking,
    init_tracking,
    sanitized_row,
    validate_upload_allowlist,
    verify_private_project,
)


def test_sanitized_row_is_allowlisted() -> None:
    row = sanitized_row(
        "run", "source", prompt_tokens=8, candidate_count=2, parsed=True, task_success=False
    )
    assert set(row) == {
        "row_hash",
        "prompt_tokens",
        "candidate_count",
        "parsed",
        "task_success",
        "error_category",
    }


def test_privacy_rejects_raw_query() -> None:
    with pytest.raises(PrivacyError):
        validate_upload_allowlist({"query": "secret"})


def test_private_project_uses_graphql_access() -> None:
    class Service:
        def execute_graphql(self, query, variables):
            assert "access" in query
            assert variables == {"name": "mlx-ft", "entity": "team"}
            return {"project": {"id": "p", "access": "PRIVATE"}}

    class Api:
        _service_api = Service()

    verify_private_project(Api(), "team", "mlx-ft")


def test_private_project_rejects_non_private_access() -> None:
    class Service:
        def execute_graphql(self, query, variables):
            return {"project": {"id": "p", "access": "READ_ONLY"}}

    class Api:
        _service_api = Service()

    with pytest.raises(PrivacyError):
        verify_private_project(Api(), "team", "mlx-ft")


def test_tracking_table_is_sanitized_and_run_finishes() -> None:
    class Run:
        def __init__(self) -> None:
            self.payloads = []
            self.finished = False

        def log(self, payload):
            self.payloads.append(payload)

        def finish(self):
            self.finished = True

    run = Run()
    row = sanitized_row(
        "run", "source", prompt_tokens=4, candidate_count=1, parsed=True, task_success=True
    )
    finish_tracking(run, metrics={"task_success": 1.0}, rows=[row])
    assert run.finished
    payload = run.payloads[0]
    assert set(payload) == {"task_success", "predictions"}
    assert all(value not in repr(payload) for value in ("source", "query", "schema", "arguments"))


def test_tracking_finishes_after_allowlist_failure() -> None:
    class Run:
        finished = False

        def log(self, payload):
            raise AssertionError("log must not run")

        def finish(self):
            self.finished = True

    run = Run()
    with pytest.raises(PrivacyError):
        finish_tracking(run, metrics={"raw_query": 1.0})
    assert run.finished


def _fake_wandb(events: list[tuple[str, object]], *, private: bool = True, fail_init: bool = False):
    class Settings:
        def __init__(self, **kwargs):
            events.append(("settings", kwargs))
            self.kwargs = kwargs

    class Service:
        def execute_graphql(self, query, variables):
            events.append(("verify", variables))
            access = "PRIVATE" if private else "READ_ONLY"
            return {"project": {"id": "project-id", "access": access}}

    class Api:
        _service_api = Service()

        def __init__(self):
            events.append(
                (
                    "api",
                    {
                        name: os.environ.get(name)
                        for name in ("WANDB_CONSOLE", "WANDB_DISABLE_GIT", "WANDB_DISABLE_CODE")
                    },
                )
            )
            self.viewer = object()

    settings_class = Settings
    api_class = Api

    class FakeWandb:
        Settings = settings_class
        Api = api_class

        @staticmethod
        def init(**kwargs):
            events.append(("init", kwargs))
            if fail_init:
                raise RuntimeError("W&B init failed")
            return "run"

    return FakeWandb


def test_online_tracking_applies_privacy_before_api_and_init(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(events))
    for name in ("WANDB_CONSOLE", "WANDB_DISABLE_GIT", "WANDB_DISABLE_CODE"):
        monkeypatch.delenv(name, raising=False)
    original_validate = tracking.validate_upload_allowlist

    def validate(payload, *, row=False):
        events.append(("allowlist", payload))
        return original_validate(payload, row=row)

    monkeypatch.setattr(tracking, "validate_upload_allowlist", validate)
    result = init_tracking(
        online=True,
        entity="team",
        project="mlx-ft",
        config={"run_label": "base", "seed": 42},
    )

    assert result == "run"
    assert [name for name, _ in events] == ["allowlist", "settings", "api", "verify", "init"]
    settings = events[1][1]
    assert settings == {
        "console": "off",
        "disable_code": True,
        "disable_git": True,
        "save_code": False,
        "x_disable_meta": True,
        "x_disable_machine_info": True,
        "x_disable_stats": True,
        "disable_job_creation": True,
        "x_save_requirements": False,
    }
    api_env = events[2][1]
    assert api_env == {
        "WANDB_CONSOLE": "off",
        "WANDB_DISABLE_GIT": "true",
        "WANDB_DISABLE_CODE": "true",
    }
    init_kwargs = events[-1][1]
    assert init_kwargs["settings"].kwargs == settings
    assert init_kwargs["save_code"] is False


def test_online_tracking_verification_failure_does_not_init(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(events, private=False))
    with pytest.raises(PrivacyError, match="must be private"):
        init_tracking(online=True, entity="team", project="mlx-ft", config={})
    assert not any(name == "init" for name, _ in events)


def test_online_tracking_init_failure_is_not_silenced(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(events, fail_init=True))
    with pytest.raises(RuntimeError, match="W&B init failed"):
        init_tracking(online=True, entity="team", project="mlx-ft", config={})
    assert [name for name, _ in events][-1] == "init"
