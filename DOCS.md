# Repository Docs

## Overview

This repository provides a small set of utilities for working with LLM inference pipelines, with a particular focus on reusable local and server-backed chat generation. The code is now organized entirely around a reusable Python package in `llm_utils/`, which contains both the lower-level request helpers and the higher-level pipeline/CLI modules.

## Repository Structure

### Top Level

- `DOCS.md`
  This file. It documents the repository structure and shows basic usage examples.

- `main.py`
  Top-level command-line entrypoint for running `VLLMPipeline` over a CSV
  column of prompts. It reads the CSV path and prompt-column name from the
  command line, batches prompts into one pipeline call, and prints a JSON list
  of generations.

- `test/`
  Test directory for the concrete pipeline implementations. It includes a
  readable `README.md`, unit-style fake-backed coverage for the local
  `TransformersPipeline` and `VLLMPipeline`, and live integration tests for
  `SGLangPipeline` against a running local server.

### Package: `llm_utils/`

- `llm_utils/__init__.py`
  Re-exports the main package helpers so callers can import key utilities directly from `llm_utils`, including both the request helpers and the pipeline abstractions.

- `llm_utils/_compat.py`
  Small compatibility shim for older Python versions. It provides fallback support for `dataclass` and `Literal` when the runtime does not provide them natively.

- `llm_utils/llm_configs.py`
  Defines `RequestConfig`, which stores runtime settings for SGLang chat requests, and `args_to_request_config`, which converts parsed CLI arguments into a validated config object.

- `llm_utils/request_sglang.py`
  Implements the client-side request helpers for sending chat completions to an SGLang server through its OpenAI-compatible API. It also includes basic logging setup and assistant-response parsing.

- `llm_utils/request_vllm.py`
  Implements reusable helper functions for local `vllm` chat generation. It can initialize a local vLLM engine, render structured chat messages into prompts, and run either single-request or batch-request completions that return the standard assistant-message dict shape used by the pipelines.

- `llm_utils/deploy_sglang.py`
  Implements a command-line deployment helper for launching an SGLang model server. It validates inputs, optionally checks the model with `transformers`, constructs the server command, and launches the process.

- `llm_utils/sglang_offline_batch_inference.py`
  Provides a small offline batch-inference script built around SGLang's
  `sgl.Engine(...).generate(...)` flow. It reads a CSV file with `pandas`,
  pulls prompts from one user-specified column, wraps each prompt as a user
  chat message, renders prompt strings with a tokenizer chat template when
  available, and runs the batch with a model passed explicitly through the
  script's `--model` flag.

- `llm_utils/pipelines.py`
  Defines the main pipeline abstraction used by the repo. It includes:
  - `PipelineConfig` for pipeline settings
  - `LLMPipeline` as the base class
  - `SGLangPipeline` for OpenAI-compatible SGLang chat requests
  - `TransformersPipeline` for local Hugging Face `transformers` chat generation with the same high-level `.generate(...)` interface used by the other pipelines
  - `VLLMPipeline` for local `vllm` chat generation with the same `.generate(...)` interface and batching semantics
  - `MockPipeline` for lightweight testing without a GPU
  - helper functions for building pipelines from config or CLI-style args

- `llm_utils/cli.py`
  Command-line entrypoint for running a single SGLang pipeline request. It parses runtime arguments with `argparse`, reads a prompt file containing system and user prompts, initializes `SGLangPipeline`, and prints the response.

## How The Pieces Fit Together

The usual flow is:

1. Start an SGLang server with `llm_utils/deploy_sglang.py`.
2. Either run `llm_utils/cli.py` with a prompt file or create a `PipelineConfig` in `llm_utils/pipelines.py`.
3. Build an `SGLangPipeline`, `TransformersPipeline`, `VLLMPipeline`, or `MockPipeline` from that config.
4. Send prompts through the pipeline and receive assistant messages.

For local vLLM batch generation from a CSV file, use the top-level `main.py`
script instead of the SGLang-specific CLI.

The `llm_utils/` package contains both the lower-level building blocks and the higher-level pipeline interfaces.

## Example: Deploy An SGLang Model

```python
import subprocess
import sys

cmd = [
    sys.executable,
    "llm_utils/deploy_sglang.py",
    "--model",
    "meta-llama/Llama-3.1-8B-Instruct",
    "--host",
    "0.0.0.0",
    "--port",
    "30000",
    "--tp-size",
    "1",
    "--skip-transformers-check",
]

subprocess.run(cmd, check=True)
```

## Example: Set Up An SGLang Pipeline

```python
from llm_utils import PipelineConfig, pipeline_from_config

cfg = PipelineConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    pipeline_type="sglang",
    host="127.0.0.1",
    port=30000,
    temperature=0.7,
    max_new_tokens=256,
    timeout=60,
)

pipeline = pipeline_from_config(cfg)

response = pipeline.generate("Explain what this repository does.")
print(response)
```

## Example: Offline Batch Inference With SGLang

Create a CSV file such as:

```csv
prompt,topic
What does offline batch inference mean?,definition
Name one benefit of batching prompts.,benefits
```

Then run:

```bash
python3 llm_utils/sglang_offline_batch_inference.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --csv-file prompts.csv \
  --prompt-column prompt \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 128
```

The script prints a JSON list where each item includes the original prompt
string, the wrapped chat messages, the rendered prompt string, and the
generated text.

## Example: Set Up A vLLM Pipeline

```python
from llm_utils import PipelineConfig, pipeline_from_config

cfg = PipelineConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    pipeline_type="vllm",
    temperature=0.2,
    max_new_tokens=128,
    top_p=0.9,
    top_k=40,
)

pipeline = pipeline_from_config(cfg)

response = pipeline.generate(
    [
        {"role": "system", "content": "Reply in one short paragraph."},
        {"role": "user", "content": "What is vLLM in this repository?"},
    ]
)
print(response)
```

The vLLM pipeline loads the model locally, renders chat prompts through a chat
template when one is available, and returns assistant-shaped response dicts for
either one request or a batch of parallel requests.

## Example: Run `main.py` On A CSV Prompt Column

Create a CSV file such as:

```csv
id,prompt
1,Explain batching in one sentence.
2,Name one advantage of vLLM.
```

Then run:

```bash
python3 main.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --csv-file prompts.csv \
  --prompt-column prompt \
  --max-new-tokens 128
```

The script loads the CSV, reads all non-blank values from the selected prompt
column, sends them through `VLLMPipeline` as one batch, and prints a JSON list
with each input prompt, the raw assistant response dict, and the generated
text.

## Testing

The repository test suite lives in `test/`.

- `test/test_transformers_pipeline.py`
  Verifies `TransformersPipeline` behavior without requiring local
  `transformers` or `torch` installs by using small fake tokenizer/model
  objects.

- `test/test_sglang_pipeline.py`
  Exercises `SGLangPipeline` against a live SGLang server at `0.0.0.0:30000`
  using the `Qwen/Qwen3.5-35B-A3B-FP8` model and short deterministic prompts.

- `test/test_vllm_pipeline.py`
  Verifies `VLLMPipeline` behavior with fake local vLLM engine and tokenizer
  objects so the pipeline contract is covered without requiring a real GPU
  runtime in tests.

- `test/test_sglang_offline_batch_inference.py`
  Covers the CSV prompt loader and prompt-rendering helpers used by the offline
  batch-inference script without requiring a live SGLang runtime.

Run the full suite with:

```bash
python3 -m pytest test
```
