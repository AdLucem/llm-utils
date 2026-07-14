# Repository Docs

## Overview

This repository provides a small set of utilities for working with LLM inference pipelines, with a particular focus on SGLang. The code is now organized entirely around a reusable Python package in `llm_utils/`, which contains both the lower-level request helpers and the higher-level pipeline/CLI modules.

## Repository Structure

### Top Level

- `DOCS.md`
  This file. It documents the repository structure and shows basic usage examples.

- `test/`
  Test directory for the two concrete pipeline implementations. It includes a
  readable `README.md`, unit-style fake-backed coverage for
  `TransformersPipeline` and `MinimaxPipeline`, plus live integration tests for
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

- `llm_utils/request_minimax.py`
  Implements client-side request helpers for sending chat completions to a MiniMax model through the OpenAI SDK. It loads `MINIMAX_API_KEY` and `MINIMAX_BASE_URL` from a local `.env` file or the process environment and exposes both single-request and batch-request helpers.

- `llm_utils/deploy_sglang.py`
  Implements a command-line deployment helper for launching an SGLang model server. It validates inputs, optionally checks the model with `transformers`, constructs the server command, and launches the process.

- `llm_utils/pipelines.py`
  Defines the main pipeline abstraction used by the repo. It includes:
  - `PipelineConfig` for pipeline settings
  - `LLMPipeline` as the base class
  - `SGLangPipeline` for OpenAI-compatible SGLang chat requests
  - `MinimaxPipeline` for MiniMax chat requests through the shared `.generate(...)` pipeline interface
  - `TransformersPipeline` for local Hugging Face `transformers` chat generation with the same high-level `.generate(...)` interface used by the other pipelines
  - `MockPipeline` for lightweight testing without a GPU
  - helper functions for building pipelines from config or CLI-style args

- `llm_utils/cli.py`
  Command-line entrypoint for running a single pipeline request. It parses runtime arguments with `argparse`, reads a prompt file containing system and user prompts, initializes either `SGLangPipeline` or `MinimaxPipeline` based on `--pipeline-type`, and prints the response. The CLI supports both `python -m llm_utils.cli ...` and `python llm_utils/cli.py ...` invocation styles.

## How The Pieces Fit Together

The usual flow is:

1. Start an SGLang server with `llm_utils/deploy_sglang.py`.
2. Either run `llm_utils/cli.py` with a prompt file and a selected `--pipeline-type`, or create a `PipelineConfig` in `llm_utils/pipelines.py`.
3. Build an `SGLangPipeline`, `MinimaxPipeline`, `TransformersPipeline`, or `MockPipeline` from that config.
4. Send prompts through the pipeline and receive assistant messages.

The `llm_utils/` package contains both the lower-level building blocks and the higher-level pipeline interfaces.

For the MiniMax request helper, create a `.env` file in the repository root or export environment variables with:

```text
MINIMAX_API_KEY=your_api_key
MINIMAX_BASE_URL=your_openai_compatible_minimax_base_url
```

## Prompt File Format

`llm_utils/cli.py` accepts either of these prompt-file formats:

- JSON:
  `{"system": "You are a helpful assistant.", "user": "Summarize this repo."}`

- Plain text:

```text
[SYSTEM]
You are a helpful assistant.

[USER]
Summarize this repo.
```

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

## Example: Run The CLI With SGLang

```bash
python3 -m llm_utils.cli \
  --pipeline-type sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-file prompts.txt \
  --host 127.0.0.1 \
  --port 30000
```

## Example: Set Up A MiniMax Pipeline

```python
from llm_utils import PipelineConfig, pipeline_from_config

cfg = PipelineConfig(
    model="MiniMax-M1",
    pipeline_type="minimax",
    temperature=0.7,
    max_new_tokens=256,
    timeout=60,
)

pipeline = pipeline_from_config(cfg)

response = pipeline.generate("Explain what this repository does.")
print(response)
```

## Example: Run The CLI With MiniMax

```bash
python3 -m llm_utils.cli \
  --pipeline-type minimax \
  --model MiniMax-M1 \
  --prompt-file prompts.txt
```

This works equivalently with the file-style entrypoint:

```bash
python3 llm_utils/cli.py \
  --pipeline-type minimax \
  --model MiniMax-M1 \
  --prompt-file prompts.txt
```

## Testing

The repository test suite lives in `test/`.

- `test/test_transformers_pipeline.py`
  Verifies `TransformersPipeline` behavior without requiring local
  `transformers` or `torch` installs by using small fake tokenizer/model
  objects.

- `test/test_minimax_pipeline.py`
  Verifies `MinimaxPipeline` input handling and request dispatch without
  requiring live MiniMax credentials or network access.

- `test/test_sglang_pipeline.py`
  Exercises `SGLangPipeline` against a live SGLang server at `0.0.0.0:30000`
  using the `Qwen/Qwen3.5-35B-A3B-FP8` model and short deterministic prompts.

Run the full suite with:

```bash
python3 -m pytest test
```
