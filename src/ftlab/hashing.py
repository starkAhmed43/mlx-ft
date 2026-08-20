"""Stable hashes for reproducibility manifests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data import canonical_json


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_records(records: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(canonical_json(record) for record in records).encode("utf-8")
    return sha256_bytes(payload)
