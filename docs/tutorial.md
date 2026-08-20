# A beginner’s tool-calling lab

This lesson introduces the first mlx-ft milestone: a small, pinned
Qwen3 0.6B 4-bit function-calling pipeline on Apple Silicon. The goal is
pipeline validation, not a quality claim.

By the end, you can:

- describe what an LLM, a tool call, and an agent do;
- read a tool JSON Schema and a strict tool-call boundary;
- see how records are normalized, rendered, parsed, and scored;
- follow the LoRA parameter and optimizer-update arithmetic; and
- run the pinned smoke workflow while keeping raw artifacts local.

The companion [safe fixture notebook](tool-calling-lab.ipynb) runs the
normalization, rendering, parser, metric, masking, configuration, and
arithmetic examples without downloading, training, executing tools, or
initializing W&B.

Current study runs use immutable nested training pools (1k, 5k, 10k, and
20k), shared locked validation/test sets, and a strict schema-OOD test outside
the full training and validation API/schema components. Use
`ftlab data audit --version smoke --samples 100` to create the local ignored
100-record inspection queue and tracked aggregate findings.

The four legacy prompt-contract-0 runs are excluded from reports because
training and evaluation used different prompt contracts. Current results remain preliminary until completed controlled
runs include commit and provenance metadata. W&B private verification and
GitHub authentication are manual external gates.

## Start with the mental model

An LLM is a model that predicts the next token from text. A tool call is
structured text that asks a host application to call a named function. An
agent is a larger loop: it asks the model what to do, executes an approved
tool, gives the result back to the model, and repeats as needed.

This lab does only the first part. It generates and evaluates tool requests.
It never executes a tool and never implements an agent loop. The host
application owns those safety decisions.

The same weather example appears throughout this lesson:

User request:

~~~text
What is the weather in Paris in Celsius?
~~~

Available tool and JSON Schema:

~~~json
{
  "name": "weather.get",
  "description": "Return current weather for a city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"],
    "additionalProperties": false
  }
}
~~~

Expected strict call:

~~~text
<tool_call>
{"name":"weather.get","arguments":{"city":"Paris","units":"celsius"}}
</tool_call>
~~~

The parser checks one complete boundary, valid JSON, the tool name, an
object-valued arguments field, and the schema. It returns a parsed call or
a named error. Evaluation then compares the parsed call with the gold
answer. A metric result might say task_success: 1.0 for an exact call.
Neither the parser nor this lab contacts a weather service.

~~~mermaid
flowchart LR
    U[User request] --> M[Language model]
    M --> C[Strict tool call]
    C --> P[Parser and schema validation]
    P --> H[Host application]
~~~

The training and evaluation workflow has a separate shape:

~~~mermaid
flowchart LR
    D[Dataset] --> R[Deterministic rendering]
    R --> B[Base evaluation]
    R --> T[LoRA training]
    T --> E[Tuned evaluation]
    B --> Q[Validation report]
    E --> Q
~~~

### Words used in the lab

Text is split into tokens. A token may be a whole word, part of a word,
punctuation, or a special marker. A prompt is the input context. A completion
is the generated continuation. A schema describes valid JSON data: its type,
properties, required keys, and allowed values.

A gold answer is the trusted target call paired with a record. Prompt masking
marks prompt positions with an ignored loss label, so training learns from
the completion positions. Thinking blocks are hidden reasoning text such as
the string <think>...</think>; this strict contract rejects them. The train
split is used for updates, validation checks the run while developing, and
test stays held out for the final comparison.

The notebook uses characters as teaching tokens. Real Qwen tokenization uses
its pinned tokenizer and special tokens, so the notebook’s mask positions
explain the idea rather than reproduce model token IDs.

### A short transformer picture

Token embeddings turn token IDs into vectors. Attention lets each position
use information from other positions. With query, key, and value matrices,
one common attention expression is:

~~~text
softmax(QK^T / sqrt(d)) V
~~~

The softmax turns attention scores into weights. Feed-forward layers apply
learned nonlinear transformations to each position. Residual connections add
a layer’s input back to its output, which helps information and gradients
move through many layers. During next-token prediction, the model scores
possible next tokens and learns from the target token.

### Model size, quantization, and adapters

0.6B means about 0.6 billion learned parameters. 4-bit means the base
weights use a four-bit quantized representation. A raw lower-bound estimate
is 0.6 billion × 4 / 8 = 300,000,000 bytes: about 0.3 GB decimal
or 0.28 GiB. Runtime memory is higher because it also needs quantization
metadata, activations, attention caches, adapters, the tokenizer, and the
runtime itself.

LoRA freezes the base matrix and learns two small matrices:

~~~text
W' = W + scale × B A
~~~

MLX-LM applies the configured scale directly. This lab does not describe
that setting as an alpha/r conversion.

For this architecture, the evidence used by the experiment is:

- hidden size 1024;
- 16 attention heads;
- 8 key/value heads;
- head dimension 128;
- q output 16 × 128 = 2048;
- v output 8 × 128 = 1024.

For rank 8, a q adapter has
8 × (1024 + 2048) = 24,576 parameters. A v adapter has
8 × (1024 + 1024) = 16,384. Together they have 40,960 parameters per
layer. The final 8 layers use q and v, so there are 8 × 2 = 16 modules and
8 × 40,960 = 327,680 trainable parameters.

The smoke run reads 128 microbatches with batch size 1. Accumulation of 8
combines eight gradients into one optimizer update:
128 / 8 = 16 optimizer updates.

### What the metrics mean

ftlab.metrics.Evaluation reports these fields:

- count: number of records scored.
- tool_accuracy: fraction with the gold tool name after strict parsing.
- json_validity: fraction with valid JSON after the tool boundary is parsed.
- schema_validity: fraction whose parsed arguments pass the tool schema. A
  schema failure can still be JSON-valid and can still name the supplied tool.
- argument_key_f1: F1 for predicted versus gold argument leaf paths.
- argument_value_f1: F1 for matching values at shared leaf paths.
- exact_match: fraction with the exact tool and recursively exact arguments.
- task_success: version 1’s task result; it equals exact tool-and-argument matching.
- parser_errors: counts by strict parser error category.

Parser failures have zero validity and correctness for that record. Key and
value F1 use nested leaf paths such as location.city and location.country, not only
top-level keys. The implementation calculates each example’s F1 first, then
averages those scores. Strings are trimmed and normalized to Unicode NFC.
Integers and floats can match by numeric value only when the schema says
number; booleans never match numbers. Exact matching still checks the
complete argument structure.

no_call_accuracy is separate robustness evidence. It measures whether
synthetic prompts that should not call a tool avoid both tool boundaries. It
is not a tool-call quality metric.

The strict parser rejects:

- thinking blocks;
- incomplete or multiple boundaries;
- extra text outside the boundary;
- invalid JSON, duplicate object keys, or non-finite numbers;
- a non-object JSON payload or non-object arguments;
- extra call fields;
- a missing or unknown tool.

Schema-invalid arguments are reported separately after valid JSON and tool
parsing. They do not erase the JSON or tool validity measurements.

## Data scope and frozen smoke counts

The pinned source is Salesforce/xlam-function-calling-60k at revision
26d14ebfe18b1f7b524bd39b404b50af5dc97866. The frozen smoke preparation
produced this exact source-to-accepted flow:

| Stage | Count |
| --- | ---: |
| Source | 60,000 |
| Accepted | 21,718 |
| Rejected | 38,282 |

| Rejection category | Count |
| --- | ---: |
| Exactly one answer | 29,059 |
| Rendered length | 2,295 |
| Bare list | 2,644 |
| Union | 1,145 |
| Tuple | 1,069 |
| Callable | 510 |
| Dict without properties | 397 |
| Candidate names not unique | 260 |
| Set | 213 |
| Duplicate gold conflict | 121 |
| Gold schema failure | 550 |
| Unsupported native schema keyword | 19 |

This table is strict scope filtering, not a claim that all source records
are bad. A record is rejected when it has inconsistent or ambiguous source
data, an unsupported schema shape, or does not fit the rendered prompt limit
of 512 tokens. All unsupported cases remain rejected. Duplicate groups keep
conflicting gold answers out of a split.

## Setup

Create the dedicated environment. Expected result: a new mlx-ft Conda
environment. Common failure: Conda is not installed, or an environment with
that name already exists; use your normal Conda environment-management command
before continuing.

~~~sh
conda env create -f environment.yml
~~~

Activate it. Expected result: the shell prompt shows (mlx-ft). Common
failure: Conda shell integration is not initialized for this shell.

~~~sh
conda activate mlx-ft
~~~

Install the locked dependencies and the optional Apple, data, tracking,
reporting, and development groups. Expected result: the lock resolves without
changes. Common failure: the Python version or platform does not meet the
lock; use Apple Silicon and Python 3.14 as declared by the project.

~~~sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --extra apple --extra data --extra tracking --extra reporting --group dev
~~~

Open the safe notebook with an existing Jupyter or VS Code installation.
Expected result: the notebook opens; it adds no notebook dependency. Common
failure: the command is not installed, so open the file from an existing IDE.

~~~sh
jupyter notebook docs/tool-calling-lab.ipynb
# Or open docs/tool-calling-lab.ipynb in an existing VS Code installation.
~~~

## Fixture checks before any model work

Run the import and configuration gate. Expected output: package versions and
model=skipped. It does not download a model. Common failure: a missing core
dependency means the locked sync did not complete.

~~~sh
ftlab preflight
~~~

Run the project tests. Expected output: all core tests pass. Artifact: no
model or dataset is required. Common failure: a failed test names the
contract that needs investigation.

~~~sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run pytest tests
~~~

Inspect local artifact directories. Expected output: JSON with free space and
directory presence. Artifact: none. Common failure: a full disk; keep the
30 GiB safety floor for model work.

~~~sh
ftlab storage status
~~~

List cleanup candidates without deleting anything. Expected output: paths
already present under the project. Artifact: none. Common failure: none in
version 1; deletion is intentionally not implemented.

~~~sh
ftlab clean --dry-run
~~~

## Pinned data preparation

Prepare the smoke split from the pinned source. This command requires accepted
source terms and an authenticated Hugging Face account. Expected artifact:
data/smoke/manifest.json, train.jsonl, validation.jsonl, and test.jsonl
(plus hashes). Common failure: access, authentication, revision, or
tokenizer-rendering failure. The command fails rather than substituting
another source.

~~~sh
ftlab data prepare --version smoke
~~~

Validate the manifest without changing it. Expected output: valid: true and
split sizes. Artifact: no new data. Common failure: a missing sidecar hash,
changed lockfile, or invalid normalized record.

~~~sh
ftlab data validate --version smoke
~~~

## Base, training, and tuned runs

The model revision is pinned to
173234aa840d113125e9f2271100ddbaf16c9620. Real model commands require
Apple Silicon, the Apple extras, enough disk space, accepted source terms,
and Hugging Face/model access. When absent, pinned model and tokenizer assets
download into the project-local models cache.

Run the real model preflight. Expected output: optional package versions and a
short generated response. Artifact: model files in the local models cache.
Common failure: unsupported hardware, missing optional packages, network
failure, or unavailable Hugging Face/model access.

~~~sh
ftlab preflight --real-model
~~~

Evaluate the base model on the held-out smoke test. Expected output: the
evaluation JSON. Artifacts under runs/<run-id>/: evaluation.json,
predictions.raw.json, input hashes, and sanitized tracking data. Common
failure: missing manifest, network or Hugging Face/model access, tokenizer
rendering, or the 30 GiB disk floor.

~~~sh
ftlab evaluate --model Qwen/Qwen3-0.6B-MLX-4bit --split test --wandb-online --wandb-entity starkahmed43 --wandb-project mlx-ft
~~~

Train the configured LoRA adapter. Expected output: JSON containing the
adapter directory, 128 microbatches, and 16 optimizer updates. Artifacts:
runs/<run-id>/adapter/adapters.safetensors, training.json, and local
training metrics. Common failure: missing prepared data, Apple/MLX
incompatibility, memory or disk limits, or model-cache access.

~~~sh
ftlab train --config configs/experiments/smoke-06b.yaml --wandb-online --wandb-entity starkahmed43 --wandb-project mlx-ft
~~~

Copy the adapter path printed by training into the next command. Evaluate the
tuned model with the same held-out split. Expected output: another evaluation
JSON; artifact: another local runs/<run-id>/ directory. Common failure:
the adapter path does not contain adapters.safetensors.

~~~sh
ftlab evaluate --model Qwen/Qwen3-0.6B-MLX-4bit --split test --adapter-path RUNS_ADAPTER_PATH --wandb-online --wandb-entity starkahmed43 --wandb-project mlx-ft
~~~

Build the local aggregate report. Expected output: paths to the aggregate
table, robustness JSON, and seven stable figure files. Artifacts:
reports/aggregate.csv, aggregate.json, robustness.json, and the seven planned
PNG files. Common failure:
the reporting extras are not installed. Zero evaluations still succeeds with
an empty aggregate and a blank plot.

~~~sh
ftlab report build
~~~

### Tracking and privacy

W&B is offline by default. --wandb-online is an explicit choice. For online
runs, the tutorial commands select starkahmed43/mlx-ft. The client accepts the
supplied entity and project, then verifies that project is private before
initialization. Raw predictions, prompts, source records, and adapters remain
local. Only allowlisted configuration, metrics, and sanitized rows upload.
Rows contain opaque hashes, lengths, counts, booleans, and error categories.
Confirm the project privacy and your organization’s policy before using online
tracking.

## What the milestone showed

The legacy validation run is excluded from current reports because its prompt
contract was invalid. Current results remain preliminary until controlled runs
have complete artifacts, commit provenance, and final measurements.

## Glossary

- Agent: a host-controlled loop that can ask a model, execute approved tools,
  and continue from results.
- Adapter: a small trainable parameter set applied to a frozen base model.
- Completion: generated text after a prompt.
- Gold answer: the trusted tool name and arguments for one record.
- JSON Schema: a machine-readable description of valid JSON values.
- LoRA: low-rank adapters represented here by scale × B A.
- Masking: ignoring selected token positions in the training loss.
- Prompt: the text and context given to a model.
- Schema validity: whether arguments pass the selected tool’s schema.
- Tool call: structured model output naming a function and its arguments.
- Token: one unit in a model’s text vocabulary.

## Suggested hands-on exercises

1. Open the notebook and change units from celsius to fahrenheit. Predict
   which metrics change, then run the cell.
2. Remove the closing tool boundary and identify the parser category.
3. Add a valid optional argument and compare key F1 with exact match.
4. Change one value to the other allowed unit and explain schema validity
   versus exact match.
5. Change the LoRA rank from 8 to 4 on paper. Recalculate every count.
6. Run the fixture checks, prepare the smoke manifest, and inspect only its
   structural fields before deciding whether real model work is appropriate.
