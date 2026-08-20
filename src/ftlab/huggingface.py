"""Pinned Hugging Face snapshot resolution with project-local caches."""

from __future__ import annotations

from pathlib import Path

from .exceptions import ExternalDependencyError


def resolve_snapshot(repo_id: str, revision: str, cache_root: str | Path) -> Path:
    """Resolve a commit-pinned repository before optional clients are imported."""
    local = Path(repo_id).expanduser()
    if local.exists():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ExternalDependencyError(
            "huggingface_hub is required for pinned model and tokenizer snapshots"
        ) from exc
    cache = Path(cache_root).resolve() / "huggingface" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(repo_id, revision=revision, cache_dir=str(cache))
    except Exception as exc:
        raise ExternalDependencyError(
            f"failed to resolve pinned Hugging Face snapshot {repo_id}@{revision}"
        ) from exc
    return Path(path).resolve()
