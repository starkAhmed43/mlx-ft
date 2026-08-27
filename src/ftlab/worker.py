"""Child-process entry points for MLX model work."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any


def bfcl_row_inputs(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Convert one supported official BFCL row to a Qwen prompt input."""
    from .data import normalize_schema

    question: Any = row.get("query", row.get("question"))
    query: Any
    if isinstance(question, str):
        query = question
    else:
        turns = question
        if isinstance(turns, list) and len(turns) == 1 and isinstance(turns[0], list):
            turns = turns[0]
        if not isinstance(turns, list) or len(turns) != 1:
            raise ValueError("BFCL question must contain exactly one user turn")
        turn = turns[0]
        if not isinstance(turn, dict) or turn.get("role") not in {None, "user"}:
            raise ValueError("BFCL question must contain one user turn")
        query = turn.get("content")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("BFCL user turn content must be a non-empty string")

    raw_tools = row.get("tools", row.get("functions", row.get("function")))
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("BFCL row must contain a non-empty function list")
    tools: list[dict[str, Any]] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            raise ValueError("BFCL function descriptor must be an object")
        descriptor = raw.get("function", raw)
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("name"), str):
            raise ValueError("BFCL function descriptor must contain a name")
        parameters = descriptor.get("parameters", descriptor.get("schema"))
        if not isinstance(parameters, dict):
            raise ValueError("BFCL function descriptor must contain an object schema")
        normalized = normalize_schema(parameters)
        function = {
            key: value for key, value in descriptor.items() if key in {"name", "description"}
        }
        function["parameters"] = normalized
        tools.append({"type": "function", "function": function})
    return query, tools


def _allocator_peak() -> int | None:
    try:
        import mlx.core as mx

        for name in ("get_peak_memory", "get_active_memory"):
            value = getattr(mx, name, None)
            if callable(value):
                return int(value())
    except ImportError, TypeError, ValueError:
        return None
    return None


def _write_result(request: dict[str, Any], result: dict[str, Any]) -> None:
    path = Path(request["result_path"])
    # Result files are persisted run artifacts. Do not leak request paths or
    # exception text that can contain local paths or credentials.
    from .artifacts import sanitize_persisted_value

    root = path.parent.parent.parent
    safe_result = sanitize_persisted_value(result, root)
    path.write_text(
        json.dumps(safe_result, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_training_request(request: dict[str, Any]) -> dict[str, Any]:
    from .config import TrainingConfig
    from .modeling import run_training

    config = TrainingConfig.model_validate(request["config"])
    result = run_training(
        config,
        model_path=request["model_path"],
        model_revision=request["model_revision"],
        cache_root=request.get("cache_root"),
        train_path=request["train_path"],
        valid_path=request["valid_path"],
        adapter_path=request["adapter_path"],
    )
    peak = _allocator_peak()
    if peak is not None:
        result["mlx_allocator_peak_bytes"] = peak
    return result


def run_evaluation_request(request: dict[str, Any]) -> dict[str, Any]:
    from .data import normalize_records
    from .metrics import evaluate_predictions
    from .modeling import generate_predictions, generate_predictions_measured, load_model
    from .rendering import TOOL_END, TOOL_START, render_record, tokenizer_fingerprint

    payload = json.loads(Path(request["manifest"]).read_text(encoding="utf-8"))
    records, rejected = normalize_records(payload["splits"][request.get("split", "test")])
    if rejected:
        raise ValueError(f"manifest contains invalid records: {rejected[0].reason}")
    load_started = time.perf_counter()
    model, tokenizer = load_model(
        request["model"],
        revision=request["model_revision"],
        adapter_path=request.get("adapter_path"),
        cache_root=request.get("cache_root"),
    )
    load_time = time.perf_counter() - load_started
    try:
        tokenizer_identity = tokenizer_fingerprint(
            tokenizer, getattr(tokenizer, "_ftlab_snapshot", None)
        )
    except Exception:
        tokenizer_identity = {}
    decoding = request.get("decoding", {})
    if not isinstance(decoding, dict):
        decoding = {}
    max_tokens = int(decoding.get("max_tokens", request.get("max_tokens", 128)))
    render_started = time.perf_counter()
    rendered = [render_record(record, tokenizer) for record in records]
    render_time = time.perf_counter() - render_started
    warmup_prompts = (
        [rendered[index % len(rendered)].prompt for index in range(8)] if rendered else []
    )
    for prompt in warmup_prompts:
        generate_predictions(model, tokenizer, [prompt], max_tokens=max_tokens)
    latencies: list[float] = []
    completion_tokens: list[int] = []
    prompt_tokens: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions: list[str] = []
    for index, item in enumerate(rendered):
        measured = generate_predictions_measured(
            model, tokenizer, [item.prompt], max_tokens=max_tokens
        )[0]
        value = measured.prediction
        latencies.append(measured.latency_seconds)
        predictions.append(value)
        prompt_count = measured.prompt_tokens
        generated_count = measured.output_tokens
        truncated = (
            generated_count >= max_tokens
            or (TOOL_START in value and TOOL_END not in value)
            or (TOOL_END in value and TOOL_START not in value)
        )
        prompt_tokens.append(prompt_count)
        completion_tokens.append(generated_count)
        diagnostics.append(
            {
                "generated_tokens": generated_count,
                "max_generation_tokens": max_tokens,
                "truncated": truncated,
            }
        )
        gold_record = records[index].as_dict()
        # The query is the rendered prompt's source text. Metrics require the
        # tool schema and gold answer, not the prompt itself.
        gold_record.pop("query", None)
        gold_hash = hashlib.sha256(
            json.dumps(gold_record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prediction_rows.append(
            {
                "source_id": records[index].source_id,
                "record": gold_record,
                "prediction": value,
                "prompt_hash": hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
                "record_hash": gold_hash,
                "hashes": {
                    "prompt": hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(),
                    "record": gold_hash,
                },
                "decoding": decoding or {"max_tokens": max_tokens},
                "prompt_tokens": prompt_count,
                "completion_tokens": generated_count,
                "latency_seconds": latencies[-1],
                "prompt_processing_tokens_per_second": measured.prompt_processing_tokens_per_second,
                "generation_tokens_per_second": measured.generation_tokens_per_second,
                "allocator_before_bytes": measured.allocator_before_bytes,
                "allocator_peak_bytes": measured.allocator_peak_bytes,
                "allocator_after_bytes": measured.allocator_after_bytes,
                "generated_tokens": generated_count,
                "max_generation_tokens": max_tokens,
                "generation_truncated": truncated,
                "diagnostics": diagnostics[-1],
            }
        )
    evaluation = evaluate_predictions(
        records, predictions, prediction_diagnostics=diagnostics
    ).as_dict()
    Path(request["predictions_path"]).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in prediction_rows)
        + "\n",
        encoding="utf-8",
    )
    Path(request["evaluation_path"]).write_text(
        json.dumps(evaluation, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    peak = _allocator_peak()
    ordered = sorted(latencies)

    def percentile(value: float) -> float:
        if not ordered:
            return 0.0
        return ordered[min(len(ordered) - 1, int(value * (len(ordered) - 1)))]

    rates = [
        tokens / latency for tokens, latency in zip(completion_tokens, latencies) if latency > 0
    ]
    quarter = max(1, len(rates) // 4)
    throughput = {
        "first_quartile": statistics.fmean(rates[:quarter]) if rates else 0.0,
        "last_quartile": statistics.fmean(rates[-quarter:]) if rates else 0.0,
    }
    generation_throughput = sum(completion_tokens) / sum(latencies) if sum(latencies) else 0.0
    prompt_rates = [
        row["prompt_processing_tokens_per_second"]
        for row in prediction_rows
        if row["prompt_processing_tokens_per_second"] is not None
    ]
    generation_rates = [
        row["generation_tokens_per_second"]
        for row in prediction_rows
        if row["generation_tokens_per_second"] is not None
    ]
    allocator_values = {
        key: [row[key] for row in prediction_rows if row[key] is not None]
        for key in ("allocator_before_bytes", "allocator_peak_bytes", "allocator_after_bytes")
    }
    return {
        "evaluation": evaluation,
        "count": len(predictions),
        "mlx_allocator_peak_bytes": peak,
        "load_time_seconds": load_time,
        "cold_load_seconds": load_time,
        "warmup_steps_excluded": 8,
        "latency_p50": percentile(0.50),
        "latency_p95": percentile(0.95),
        "latency_p99": percentile(0.99),
        "prompt_tokens": sum(prompt_tokens),
        "completion_tokens": sum(completion_tokens),
        "throughput_tokens_per_second": generation_throughput,
        "generation_throughput_tokens_per_second": generation_throughput,
        "prompt_processing_tokens_per_second": statistics.fmean(prompt_rates)
        if prompt_rates
        else None,
        "generation_tokens_per_second": statistics.fmean(generation_rates)
        if generation_rates
        else None,
        "prompt_rendering_seconds": render_time,
        "allocator_memory_bytes": {
            "before": allocator_values["allocator_before_bytes"][0]
            if allocator_values["allocator_before_bytes"]
            else None,
            "peak": max(allocator_values["allocator_peak_bytes"])
            if allocator_values["allocator_peak_bytes"]
            else None,
            "after": allocator_values["allocator_after_bytes"][-1]
            if allocator_values["allocator_after_bytes"]
            else None,
        },
        "throughput_quartiles": throughput,
        "tokenizer_identity": tokenizer_identity,
    }


def run_probe_request(request: dict[str, Any]) -> dict[str, Any]:
    """Run a bounded 32-microbatch training probe."""
    from .config import TrainingConfig
    from .modeling import run_training

    raw = dict(request["config"])
    raw["iters"] = int(request.get("steps", 32))
    raw["steps_per_report"] = max(1, int(request.get("steps", 32)))
    raw["steps_per_eval"] = max(1, int(request.get("steps", 32)))
    config = TrainingConfig.model_validate(raw)
    result = run_training(
        config,
        model_path=request["model"],
        model_revision=request["model_revision"],
        cache_root=request.get("cache_root"),
        train_path=request["train_path"],
        valid_path=request["valid_path"],
        adapter_path=request["adapter_path"],
    )
    result["mlx_allocator_peak_bytes"] = _allocator_peak()
    return result


def run_robustness_request(request: dict[str, Any]) -> dict[str, Any]:
    """Generate the fixed 100-case robustness suite in this child process."""
    from .modeling import generate_predictions, load_model
    from .rendering import render_tools_prompt, tokenizer_fingerprint
    from .robustness import all_robustness_cases, evaluate_robustness

    model, tokenizer = load_model(
        request["model"],
        revision=request["model_revision"],
        adapter_path=request.get("adapter_path"),
        cache_root=request.get("cache_root"),
    )
    cases = all_robustness_cases()
    values: list[str] = []
    rows: list[dict[str, Any]] = []
    decoding = request.get("decoding", {})
    if not isinstance(decoding, dict):
        decoding = {}
    max_tokens = int(decoding.get("max_tokens", request.get("max_tokens", 128)))
    for case in cases:
        tools = list(case.tools) if hasattr(case, "tools") else []
        prompt = render_tools_prompt(tools, case.query, tokenizer)
        value = generate_predictions(model, tokenizer, [prompt], max_tokens=max_tokens)[0]
        values.append(value)
        rows.append(
            {
                "case_id": case.case_id,
                "prediction": value,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "decoding": decoding or {"max_tokens": max_tokens},
            }
        )
    Path(request["predictions_path"]).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    result = evaluate_robustness(values)
    Path(request["evaluation_path"]).write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    peak = _allocator_peak()
    return {
        "evaluation": result,
        "count": len(values),
        "mlx_allocator_peak_bytes": peak,
        "tokenizer_identity": tokenizer_fingerprint(tokenizer),
        "prompt_hash": hashlib.sha256(
            "\n".join(row["prompt_hash"] for row in rows).encode("utf-8")
        ).hexdigest(),
        "decoding_hash": hashlib.sha256(
            json.dumps(decoding or {"max_tokens": max_tokens}, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def run_bfcl_request(request: dict[str, Any]) -> dict[str, Any]:
    """Generate selected BFCL rows in a separate child artifact."""
    from .modeling import generate_predictions, load_model
    from .rendering import render_tools_prompt

    model, tokenizer = load_model(
        request["model"],
        revision=request["model_revision"],
        adapter_path=request.get("adapter_path"),
        cache_root=request.get("cache_root"),
    )
    rows = [
        json.loads(line)
        for line in Path(request["records_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: list[dict[str, Any]] = []
    for row in rows:
        query, tools = bfcl_row_inputs(row)
        prompt = render_tools_prompt(tools, query, tokenizer)
        started = time.perf_counter()
        prediction = generate_predictions(
            model, tokenizer, [prompt], max_tokens=int(request.get("max_tokens", 128))
        )[0]
        latency = time.perf_counter() - started
        output.append(
            {
                "id": row.get("id"),
                "prediction": prediction,
                "tokens": len(tokenizer.encode(prediction, add_special_tokens=False)),
                "latency": latency,
            }
        )
    Path(request["predictions_path"]).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    return {"count": len(output), "mlx_allocator_peak_bytes": _allocator_peak()}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode == "train":
        return run_training_request(request)
    if mode == "evaluate":
        return run_evaluation_request(request)
    if mode == "probe":
        return run_probe_request(request)
    if mode == "robustness":
        return run_robustness_request(request)
    if mode == "bfcl":
        return run_bfcl_request(request)
    raise ValueError(f"unknown worker mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        result = dispatch(request)
        _write_result(request, {"status": "completed", "result": result})
        return 0
    except BaseException as exc:
        message = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        _write_result(request, {"status": "failed", "error": message})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
