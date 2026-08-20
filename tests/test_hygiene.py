from __future__ import annotations

from pathlib import Path


def test_text_files_have_lf_and_no_trailing_whitespace() -> None:
    root = Path(__file__).parents[1]
    owned_roots = (
        root / "src",
        root / "tests",
        root / ".github",
        root / "scripts",
        root / "cards",
        root / "configs",
        root / "docs",
    )
    explicit = {
        root / name
        for name in (
            ".gitattributes",
            ".gitignore",
            "LICENSE",
            "Makefile",
            "README.md",
            "environment.yml",
            "environment-bfcl.yml",
            "pyproject.toml",
        )
    }
    paths = [path for base in owned_roots for path in base.rglob("*") if path.is_file()]
    paths.extend(path for path in explicit if path.is_file())
    for path in paths:
        if path.suffix not in {
            ".py",
            ".md",
            ".yml",
            ".yaml",
            ".toml",
            ".json",
            ".ipynb",
            ".gitattributes",
            ".gitignore",
        }:
            continue
        data = path.read_bytes()
        assert b"\r" not in data, path
        for line in data.splitlines():
            assert line == line.rstrip(), path
