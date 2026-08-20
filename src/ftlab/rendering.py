"""Qwen tools rendering and the prompt/completion training contract."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .data import AcceptedRecord, canonical_json
from .exceptions import ExternalDependencyError
from .huggingface import resolve_snapshot

TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"
PROMPT_CONTRACT_VERSION = 1
LEGACY_PROMPT_CONTRACT_VERSION = 0
LEGACY_CONTRACT_RUN_IDS = frozenset(
    {
        "20260809T104310.301493Z-base-test",
        "20260809T104939.096339Z-base-test",
        "20260809T105219.723552Z-smoke-06b",
        "20260809T105438.796044Z-tuned-test",
    }
)


class TokenizerLike(Protocol):
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class RenderedRecord:
    prompt: str
    completion: str
    full: str
    prompt_tokens: tuple[int, ...] = ()
    full_tokens: tuple[int, ...] = ()

    @property
    def target_start(self) -> int:
        return len(self.prompt_tokens)


def _fallback_tools(tools: list[dict[str, Any]], query: str) -> tuple[str, str]:
    tools_json = canonical_json(tools)
    answer = tools[0]["name"] if tools else ""
    completion = f"{TOOL_START}\n{canonical_json({'name': answer, 'arguments': {}})}\n{TOOL_END}"
    prompt = f"<tools>\n{tools_json}\n</tools>\n<|user|>\n{query}\n<|assistant|>\n"
    return prompt, completion


def _tool_completion(record: AcceptedRecord) -> str:
    call = {"name": record.function_name, "arguments": record.arguments}
    return f"{TOOL_START}\n{canonical_json(call)}\n{TOOL_END}"


def _template_render(record: AcceptedRecord, tokenizer: TokenizerLike) -> tuple[str, str]:
    tools = list(record.tools)
    user = [{"role": "user", "content": record.query}]
    assistant = [
        {"role": "user", "content": record.query},
        {"role": "assistant", "content": _tool_completion(record)},
    ]
    try:
        prompt_value = tokenizer.apply_chat_template(
            user,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        full_value = tokenizer.apply_chat_template(
            assistant,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExternalDependencyError(f"pinned tokenizer cannot render tools: {exc}") from exc
    if not isinstance(prompt_value, str) or not isinstance(full_value, str):
        raise ExternalDependencyError("tokenizer must return text with tokenize=False")
    if not full_value.startswith(prompt_value):
        raise ExternalDependencyError("tokenizer prompt and completion are not a stable prefix")
    suffix = full_value[len(prompt_value) :]
    target_offset = suffix.find(TOOL_START)
    if target_offset < 0:
        raise ExternalDependencyError("rendered completion does not contain a tool call")
    # Tokens emitted by the model before the tool boundary remain masked prompt
    # context. This keeps the target boundary stable without deleting template text.
    prompt_value += suffix[:target_offset]
    return prompt_value, suffix[target_offset:]


def render_record(record: AcceptedRecord, tokenizer: TokenizerLike | None = None) -> RenderedRecord:
    """Render one record with thinking disabled for both template passes."""
    if tokenizer is None:
        prompt, completion = _fallback_tools(list(record.tools), record.query)
        completion = _tool_completion(record)
    else:
        prompt, completion = _template_render(record, tokenizer)
    full = prompt + completion
    prompt_tokens: tuple[int, ...] = ()
    full_tokens: tuple[int, ...] = ()
    if tokenizer is not None:
        try:
            prompt_tokens = tuple(tokenizer.encode(prompt, add_special_tokens=False))
            full_tokens = tuple(tokenizer.encode(full, add_special_tokens=False))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ExternalDependencyError(
                f"tokenizer cannot encode rendered record: {exc}"
            ) from exc
        if full_tokens[: len(prompt_tokens)] != prompt_tokens:
            raise ExternalDependencyError(
                "tokenizer token prefix differs between prompt and full record"
            )
    return RenderedRecord(prompt, completion, full, prompt_tokens, full_tokens)


def render_tools_prompt(
    tools: list[dict[str, Any]], query: str, tokenizer: TokenizerLike | None = None
) -> str:
    """Render one tools prompt for the installation gate."""
    if tokenizer is None:
        return _fallback_tools(tools, query)[0]
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    if not isinstance(value, str):
        raise ExternalDependencyError("tokenizer did not return a text tools prompt")
    # Mirror the full training pass so template-owned assistant markers are
    # included in both the inference prompt and masked training context.
    name = tools[0].get("name", "") if tools else ""
    probe = f"{TOOL_START}\n{canonical_json({'name': name, 'arguments': {}})}\n{TOOL_END}"
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": probe},
        ],
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
        enable_thinking=False,
    )
    if isinstance(full, str) and full.startswith(value):
        suffix = full[len(value) :]
        target_offset = suffix.find(TOOL_START)
        if target_offset >= 0:
            return value + suffix[:target_offset]
    return value


def to_completion_record(
    record: AcceptedRecord, tokenizer: TokenizerLike | None = None
) -> dict[str, str]:
    rendered = render_record(record, tokenizer)
    return {"prompt": rendered.prompt, "completion": rendered.completion}


def load_tokenizer(
    model: str, *, revision: str, cache_root: str | Path | None = None
) -> TokenizerLike:
    """Load a pinned tokenizer only when an integration command needs it."""
    try:
        if cache_root is not None:
            cache = Path(cache_root).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ.update(
                {
                    "HF_HOME": str(cache),
                    "HF_HUB_CACHE": str(cache / "hub"),
                    "HF_DATASETS_CACHE": str(cache / "datasets"),
                    "TRANSFORMERS_CACHE": str(cache / "transformers"),
                }
            )
            source = resolve_snapshot(model, revision, cache)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(source), local_files_only=True, trust_remote_code=False
            )
            setattr(tokenizer, "_ftlab_snapshot", source)
            setattr(tokenizer, "_ftlab_revision", revision)
            return cast(TokenizerLike, tokenizer)
        raise ExternalDependencyError("a project cache root is required for a remote tokenizer")
    except ImportError as exc:
        raise ExternalDependencyError(
            "transformers is required to load the pinned tokenizer"
        ) from exc
    except Exception as exc:
        raise ExternalDependencyError(f"failed to load tokenizer {model}@{revision}") from exc


def render_json_call(name: str, arguments: dict[str, Any]) -> str:
    """Render the strict single-call boundary used by fixtures and evaluation."""
    return f"{TOOL_START}\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False, sort_keys=True)}\n{TOOL_END}"


def prompt_hash(
    tools: list[dict[str, Any]], query: str, tokenizer: TokenizerLike | None = None
) -> str:
    """Hash the exact inference prompt for a tools/query pair."""
    prompt = render_tools_prompt(tools, unicodedata.normalize("NFC", query).strip(), tokenizer)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def rendering_metadata(
    record: AcceptedRecord, tokenizer: TokenizerLike | None = None
) -> dict[str, Any]:
    """Return reproducibility metadata for the versioned rendering contract."""
    rendered = render_record(record, tokenizer)
    return {
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_hash": hashlib.sha256(rendered.prompt.encode("utf-8")).hexdigest(),
        "tools_query_prompt_hash": prompt_hash(list(record.tools), record.query, tokenizer),
        "target_prefix": TOOL_START,
    }


def prompt_contract_metadata(version: int = PROMPT_CONTRACT_VERSION) -> dict[str, Any]:
    """Describe a run's prompt contract, including preserved legacy runs."""
    if version not in {LEGACY_PROMPT_CONTRACT_VERSION, PROMPT_CONTRACT_VERSION}:
        raise ValueError(f"unsupported prompt contract version: {version}")
    return {"prompt_contract_version": version, "legacy": version == LEGACY_PROMPT_CONTRACT_VERSION}


def prompt_contract_for_run(run_id: str) -> dict[str, Any]:
    """Return reusable contract metadata without rewriting historical runs."""
    version = (
        LEGACY_PROMPT_CONTRACT_VERSION
        if run_id in LEGACY_CONTRACT_RUN_IDS
        else PROMPT_CONTRACT_VERSION
    )
    return prompt_contract_metadata(version)


def tokenizer_fingerprint(
    tokenizer: TokenizerLike, snapshot_root: str | Path | None = None
) -> dict[str, str]:
    """Hash tokenizer files, chat template, and one golden token stream."""
    files: list[Path] = []
    snapshot = snapshot_root or getattr(tokenizer, "_ftlab_snapshot", None)
    snapshot_path = Path(snapshot).resolve() if snapshot is not None else None
    if snapshot_path is not None and snapshot_path.is_dir():
        tokenizer_names = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "spiece.model",
            "sentencepiece.bpe.model",
        }
        files = [
            path
            for path in snapshot_path.rglob("*")
            if path.is_file() and (path.name in tokenizer_names or "tokenizer" in path.name)
        ]
    init = getattr(tokenizer, "init_kwargs", {})
    for value in init.values() if isinstance(init, dict) else ():
        if isinstance(value, str) and Path(value).is_file():
            files.append(Path(value))
    vocab = getattr(tokenizer, "vocab_file", None)
    if isinstance(vocab, str) and Path(vocab).is_file():
        files.append(Path(vocab))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative: str
        if snapshot_path is not None:
            try:
                relative = str(path.resolve().relative_to(snapshot_path))
            except ValueError:
                relative = path.name
        else:
            relative = path.name
        digest.update(str(relative).encode())
        digest.update(path.read_bytes())
    if not files:
        raise ExternalDependencyError("pinned tokenizer snapshot has no files to hash")
    template = str(getattr(tokenizer, "chat_template", ""))
    golden = render_tools_prompt(
        [{"name": "fixture.echo", "parameters": {"type": "object", "properties": {}}}],
        "Say hello.",
        tokenizer,
    )
    tokens = tuple(tokenizer.encode(golden, add_special_tokens=False))
    token_digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    return {
        "tokenizer_files": digest.hexdigest(),
        "chat_template": hashlib.sha256(template.encode()).hexdigest(),
        "golden_tokens": token_digest,
    }
