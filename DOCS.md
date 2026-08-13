# Repository Docs

## Overview

This repository is organized as an installable Python package named `llm-utils`.
The package lives in `llm_utils/` and provides:

- reusable request helpers for SGLang, MiniMax, Anthropic-compatible endpoints, and local vLLM
- a shared pipeline abstraction in `llm_utils.pipelines`
- command-line entry points for running prompts, deploying SGLang, and offline batch inference

## Repository Structure

### Top Level

- `AGENTS.md`
  Repository-specific instructions for contributors and coding agents.

- `DOCS.md`
  This file.

- `pyproject.toml`
  Modern Python build-system declaration for the package.

- `setup.py`
  Setuptools packaging entry point that keeps the repository installable with
  older `pip` versions as well as modern tooling.

- `requirements.txt`
  Environment-specific dependency snapshot used by this workspace. It is not the
  canonical package metadata.

- `sample-prompt.txt`
  Example prompt file for the package CLI.

- `setenv`
  Shell helper for environment setup in this workspace.

- `vllm_setenv`
  Shell helper for vLLM-oriented environment setup in this workspace.

- `test/`
  Focused tests and supporting fixtures for the package.

### Package: `llm_utils/`

- `llm_utils/__init__.py`
  Public package exports for the main request helpers and pipeline classes.

- `llm_utils/__main__.py`
  Module entry point so the package can be run with `python -m llm_utils`.

- `llm_utils/_compat.py`
  Compatibility helpers for `dataclass` and `Literal`.

- `llm_utils/cli.py`
  Main CLI entry point for sending a prompt file through a configured pipeline.

- `llm_utils/deploy_sglang.py`
  CLI helper for launching an SGLang server process.

- `llm_utils/llm_configs.py`
  Shared config object and argparse-to-config conversion helpers for request code.

- `llm_utils/pipelines.py`
  Shared pipeline abstraction and concrete implementations for SGLang,
  MiniMax, Anthropic-compatible endpoints, local `transformers`, local `vllm`,
  and mock testing.

- `llm_utils/request_anthropic_api.py`
  Anthropic Messages API-compatible request helpers.

- `llm_utils/request_minimax.py`
  MiniMax request helpers built on the OpenAI SDK.

- `llm_utils/request_sglang.py`
  SGLang OpenAI-compatible request helpers.

- `llm_utils/request_vllm.py`
  Local vLLM request helpers and prompt rendering.

- `llm_utils/sglang_offline_batch_inference.py`
  Offline batch inference utility that reads prompts from a CSV column and runs
  them through SGLang.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

The base install includes the shared package and the SGLang HTTP request helper.
Backend-specific integrations are exposed through extras:

- `pip install -e ".[minimax]"` for MiniMax support
- `pip install -e ".[anthropic]"` for Anthropic-compatible endpoints
- `pip install -e ".[transformers]"` for local Hugging Face generation
- `pip install -e ".[vllm]"` for local vLLM generation
- `pip install -e ".[offline-batch]"` for CSV-driven SGLang batch inference
- `pip install -e ".[all]"` for the combined optional stack
- `pip install -e ".[dev]"` for test tooling

Some optional extras depend on newer Python versions than the package base
itself. The core package and SGLang HTTP helper remain installable with older
Python environments that already satisfy the repository code.

## Command-Line Usage

After installation, the package exposes these console scripts:

- `llm-utils`
  Runs the main prompt-file pipeline CLI.

- `llm-utils-deploy-sglang`
  Launches an SGLang model server.

- `llm-utils-sglang-offline-batch`
  Runs offline batch inference from a CSV file.

You can also invoke the main CLI as a module:

```bash
python -m llm_utils --help
python -m llm_utils.cli --help
```

## Prompt File Format

`llm-utils` accepts either of these prompt-file formats:

- JSON:
  `{"system": "You are a helpful assistant.", "user": "Summarize this repo."}`

- Plain text:

```text
[SYSTEM]
You are a helpful assistant.

[USER]
Summarize this repo.
```

## Example: Run The Main CLI

```bash
llm-utils \
  --pipeline-type sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-file sample-prompt.txt \
  --host 127.0.0.1 \
  --port 30000
```

## Example: Use The Package In Python

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

## Example: Deploy An SGLang Server

```bash
llm-utils-deploy-sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --skip-transformers-check
```

## Example: Offline Batch Inference

```bash
llm-utils-sglang-offline-batch \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --csv-file test/generated_test.csv \
  --prompt-column prompt \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 128
```

## Testing

Run the focused test suite from the repository root with:

```bash
pytest
```
