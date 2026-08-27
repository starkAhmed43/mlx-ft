from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "docs" / "tool-calling-lab.ipynb"


def _read_notebook() -> tuple[dict[str, object], str]:
    text = NOTEBOOK.read_text(encoding="utf-8")
    return json.loads(text), text


def test_notebook_metadata_outputs_and_links() -> None:
    notebook, text = _read_notebook()
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    metadata = notebook["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["kernelspec"] == {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    assert metadata["language_info"]["name"] == "python"

    absolute_path = re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home|private|tmp|var|workspace)/")
    credential = re.compile(
        r"(?:api[_-]?key|client[_-]?secret|password|bearer\s+[A-Za-z0-9._-]{8,}|hf_[A-Za-z0-9]{8,})",
        re.IGNORECASE,
    )
    assert not absolute_path.search(text)
    assert not credential.search(text)
    assert "Salesforce/xlam-function-calling-60k" not in text
    assert "write_text" not in text
    assert "to_csv" not in text
    assert "savefig" not in text

    cells = notebook["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        assert cell["cell_type"] in {"code", "markdown"}
        assert all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell["outputs"] == []

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tutorial = (ROOT / "docs" / "tutorial.md").read_text(encoding="utf-8")
    assert "(docs/tutorial.md)" in readme
    assert "(docs/tool-calling-lab.ipynb)" in readme
    assert "tool-calling-lab.ipynb" in tutorial


def test_tutorial_preserves_pinned_pipeline_facts() -> None:
    tutorial = (ROOT / "docs" / "tutorial.md").read_text(encoding="utf-8")

    for revision in (
        "173234aa840d113125e9f2271100ddbaf16c9620",
        "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
    ):
        assert revision in tutorial
    assert "No prepared dataset is included." in tutorial
    assert "only the final 20k endpoint" in tutorial

    assert "starkahmed43/mlx-ft" in tutorial
    assert re.search(r"accepts the\s+supplied entity and project", tutorial)
    assert re.search(r"verifies that project is private before\s+initialization", tutorial)
    assert "results remain preliminary" in tutorial
    assert "legacy validation run is excluded" in tutorial
    assert "schema failure can still be JSON-valid" in tutorial
    assert re.search(
        r"when absent, pinned model and tokenizer assets\s+download into the project-local models cache",
        tutorial,
        re.IGNORECASE,
    )
    assert re.search(
        r"zero evaluations still succeeds with\s+an empty aggregate and a blank plot",
        tutorial,
        re.IGNORECASE,
    )

    commands = (
        "conda env create -f environment.yml",
        "conda activate mlx-ft",
        'UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --extra apple --extra data --extra tracking --extra reporting --group dev',
        "jupyter notebook docs/tool-calling-lab.ipynb",
        "ftlab preflight",
        'UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run pytest tests',
        "ftlab storage status",
        "ftlab clean --dry-run",
        "ftlab data prepare --version smoke",
        "ftlab data validate --version smoke",
        "ftlab preflight --real-model",
        "ftlab evaluate --model Qwen/Qwen3-0.6B-MLX-4bit --split test --wandb-online --wandb-entity starkahmed43 --wandb-project mlx-ft",
        "ftlab train --config configs/experiments/smoke-06b.yaml --wandb-online --wandb-entity starkahmed43 --wandb-project mlx-ft",
        "ftlab evaluate --model Qwen/Qwen3-0.6B-MLX-4bit --split test --adapter-path RUNS_ADAPTER_PATH --wandb-online --wandb-entity starkahmed43 --wandb-project mlx-ft",
        "ftlab report build",
    )
    for command in commands:
        assert command in tutorial


def test_notebook_safe_cells_compile_and_execute_sequentially() -> None:
    notebook, _ = _read_notebook()
    safe_imports = {"json", "pathlib", "yaml"}
    forbidden_names = {
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "wandb",
        "mlx",
        "transformers",
        "load_dataset",
        "from_pretrained",
        "load_model",
        "generate_predictions",
    }
    forbidden_calls = {"train", "fit", "save", "download", "execute_tool"}
    namespace: dict[str, object] = {"__name__": "__main__"}

    import os

    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tree = ast.parse(source, filename=f"tool-calling-lab-cell-{index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] in safe_imports for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] in safe_imports or (
                        node.module or ""
                    ).startswith("ftlab")
                elif isinstance(node, ast.Name):
                    assert node.id not in forbidden_names
                elif isinstance(node, ast.Call):
                    function = node.func
                    call_name = function.id if isinstance(function, ast.Name) else ""
                    assert call_name not in forbidden_calls
            exec(compile(tree, f"tool-calling-lab-cell-{index}", "exec"), namespace)
    finally:
        os.chdir(old_cwd)

    record = namespace["weather_record"]
    assert record.query == "What is the weather in Paris in Celsius?"
    assert record.function_name == "weather.get"
    assert record.arguments == {"city": "Paris", "units": "celsius"}
    assert namespace["normalized_schema"]["required"] == ["city"]
    assert namespace["gold_call"] == {
        "name": "weather.get",
        "arguments": {"city": "Paris", "units": "celsius"},
    }

    rendered = namespace["rendered"]
    assert rendered.full == rendered.prompt + rendered.completion
    labels = namespace["masked_labels"]
    prompt_size = len(namespace["teaching_prompt_tokens"])
    assert labels[:prompt_size] == [-100] * prompt_size
    assert labels[prompt_size:] == namespace["teaching_completion_tokens"]

    assert namespace["parse_categories"] == {
        "thinking_block": "thinking_block",
        "partial_boundary": "partial_boundary",
        "extra_text": "extra_output",
        "invalid_json": "invalid_json",
        "nonobject_payload": "json_fragment",
        "nonobject_arguments": "non_object_arguments",
        "extra_call_field": "invalid_call",
        "unknown_tool": "unknown_tool",
        "schema_failure": "schema_invalid",
        "multiple_calls": "multiple_calls",
    }

    evaluation = namespace["evaluation"]
    assert evaluation.count == 4
    for attribute, expected in {
        "tool_accuracy": 0.75,
        "json_validity": 0.75,
        "schema_validity": 0.75,
        "argument_key_f1": 10 / 13,
        "argument_value_f1": 6 / 13,
        "exact_match": 0.25,
        "task_success": 0.5,
    }.items():
        assert math.isclose(getattr(evaluation, attribute), expected)
    assert evaluation.parser_errors == {"thinking_block": 1}
    assert namespace["no_call_score"] == 0.5

    assert namespace["lora_counts"] == {
        "q_output": 2048,
        "v_output": 1024,
        "q_per_layer": 24576,
        "v_per_layer": 16384,
        "modules": 16,
        "total": 327680,
    }
    assert namespace["optimizer_updates"] == 16
    config = namespace["config_values"]
    assert config["model_revision"] == "173234aa840d113125e9f2271100ddbaf16c9620"
    assert config["dataset_revision"] == "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
    assert config["max_seq_length"] == 512
    assert namespace["manifest_summary"]["exists"] in {True, False}
    if not namespace["manifest_summary"]["exists"]:
        assert "message" in namespace["manifest_summary"]
