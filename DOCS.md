# Repository Docs

## Overview

This repository is organized as an installable Python package named `llm-utils`.
The package lives in `llm_utils/` and provides:

- reusable request helpers for SGLang, MiniMax, Anthropic-compatible endpoints, and local vLLM
- a shared pipeline abstraction in `llm_utils.pipelines`
- image generation for OpenAI and MiniMax behind one interface in `llm_utils.images`
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

- `llm_utils/_env.py`
  Shared `.env` reading for every provider helper: `read_dotenv(path)` for one
  file, `dotenv_values()` for the merge of every `.env` from the working
  directory up to the filesystem root (nearest wins), `clean_env_value` for
  stripping quotes, and `env_value(name)` for "environment first, then `.env`".

- `llm_utils/cli.py`
  Main CLI entry point for sending a prompt file through a configured pipeline.

- `llm_utils/deploy_sglang.py`
  CLI helper for launching an SGLang server process.

- `llm_utils/images.py`
  Provider-neutral image generation: `ImagePipelineConfig`, `ImageResult`, the
  `ImagePipeline` base class with `OpenAIImagePipeline`, `MinimaxImagePipeline`
  and `MockImagePipeline`, the `image_pipeline_from_config` factory, the
  `ImageRequestRejected` / `ImageProviderError` split, and the dependency-free
  header readers `png_dimensions` and `jpeg_dimensions`. `MockImagePipeline`
  writes a solid-colour PNG with `zlib` and `struct`, so nothing here needs
  Pillow.

- `llm_utils/llm_configs.py`
  Shared config object and argparse-to-config conversion helpers for request code.

- `llm_utils/pipelines.py`
  Shared pipeline abstraction and concrete implementations for SGLang,
  MiniMax, Anthropic-compatible endpoints, OpenAI, local `transformers`,
  local `vllm`, and mock testing.

- `llm_utils/request_anthropic_api.py`
  Anthropic Messages API-compatible request helpers.

- `llm_utils/request_minimax.py`
  MiniMax request helpers built on the Anthropic SDK (MiniMax is served
  through an Anthropic-compatible endpoint). `minimax_chat_completion_stream`
  is the streaming counterpart to `minimax_chat_completion`: it opens
  `client.messages.stream(...)` and yields delta/thinking-delta events as
  they arrive, joining block texts with `"\n"` on each `content_block_stop`
  to match `minimax_chat_completion`'s non-streaming accumulation exactly,
  then yields a final `{"type": "done", "message": {...}}` event.

  Both functions construct their `anthropic.Anthropic` client with
  `max_retries=4` (double the SDK default of 2) and wrap the actual API call
  in `try/except anthropic.AnthropicError`, translating any escaping
  exception via `_raise_as_backend_error` into `MinimaxBackendError` — a
  clean, human-readable message (never the raw Anthropic error dump, which
  includes a JSON blob and request id) plus `retryable: bool` and
  `status_code: int | None`. `anthropic.OverloadedError` (529) and
  `RateLimitError` get their own friendly messages; any other
  `APIStatusError` is retryable exactly when its status is in
  `{408, 409, 429, 500, 502, 503, 504, 529}`; `APIConnectionError` (including
  timeouts) is always retryable. This is what a MiniMax overload/rate-limit
  surfaces as everywhere upstream (`MinimaxPipeline`, `Actor.generate`,
  `centaurus/src/api.py`) instead of a raw SDK exception — none of those
  layers needed to change to get a clean message, since they already just
  propagate `str(exc)`. `centaurus/src/api.py` also reads `retryable`: it
  answers `503` when the error is retryable and `502` when it is not.

- `llm_utils/request_minimax_images.py`
  MiniMax image generation over plain HTTP (`minimax_image_generation`).
  Requires `MINIMAX_API_KEY`; `MINIMAX_IMAGE_BASE_URL` is optional and defaults
  to `https://api.minimax.io/v1`. Unlike the chat helper it does **not** need
  `MINIMAX_BASE_URL`. Always requests base64 because returned URLs expire after
  24 hours, and checks `base_resp.status_code` because a failure can arrive
  inside an HTTP 200.

- `llm_utils/request_openai.py`
  OpenAI Chat Completions request helpers built on the `openai` SDK
  (`openai_chat_completion`, `openai_chat_completion_batch`,
  `openai_chat_completion_stream`). Credentials come from `OPENAI_API_KEY`
  (required) and `OPENAI_BASE_URL` (optional, defaults to the public API);
  both are read from the environment or from the nearest `.env` found by
  walking up from the working directory. Reasoning models (`gpt-5*`, `o1*`,
  `o3*`, `o4*`) are sent without `temperature`.

- `llm_utils/request_openai_images.py`
  OpenAI Images API helper (`openai_image_generation`). Reuses the chat
  helper's credentials, returns bytes rather than a URL, and converts an
  HTTP 400 into `ImageRequestRejected` so a refused prompt is not retried.
  `dall-e-*` models get `response_format="b64_json"` instead of
  `output_format`/`quality`.

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
- `pip install -e ".[images]"` for OpenAI and MiniMax image generation
- `pip install -e ".[anthropic]"` for Anthropic-compatible endpoints
- `pip install -e ".[transformers]"` for local Hugging Face generation
- `pip install -e ".[vllm]"` for local vLLM generation
- `pip install -e ".[offline-batch]"` for CSV-driven SGLang batch inference
- `pip install -e ".[all]"` for the combined optional stack
- `pip install -e ".[dev]"` for test tooling

Some optional extras depend on newer Python versions than the package base
itself. The core package and SGLang HTTP helper remain installable with older
Python environments that already satisfy the repository code.

When installing into a virtualenv, use `python3 -m pip install -e .` rather
than a bare `pip install -e .` if a plain `pip` might resolve to a different
interpreter's launcher (e.g. a user-level `~/.local/bin/pip` shadowing the
venv's own `pip` on `PATH` even after activation) — otherwise the edit lands
in the wrong environment and the venv silently keeps whatever `llm_utils`
build it already had. Confirm with `python3 -c "import llm_utils; print(llm_utils.__file__)"`
from outside this directory; if it resolves anywhere other than this
repository's `llm_utils/`, the venv has a stale, non-editable copy (for
example centaurus's `requirements.txt` pip-installs from
`git+https://github.com/AdLucem/llm-utils.git` directly, independently of
this submodule checkout) and edits here will not take effect until it is
reinstalled from this path.

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

To run a local vLLM model through the same CLI:

```bash
llm-utils \
  --pipeline-type vllm \
  --model meta-llama/Llama-3.2-1B \
  --prompt-file sample-prompt.txt
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

## Example: Stream A Response

Every pipeline also has `generate_stream(inputs)`, a generator of events:

- `{"type": "delta", "text": str}` for each chunk of assistant text
- `{"type": "thinking_delta", "text": str}` for reasoning tokens, where the
  provider emits them
- exactly one final `{"type": "done", "message": {...}}`, whose `message` is
  identical to what `generate(inputs)` would have returned

Concatenating every `delta` text reproduces the final message content, so a
caller can render tokens as they arrive and then replace them with the final
message without duplication.

```python
for event in pipeline.generate_stream("Explain what this repository does."):
    if event["type"] == "delta":
        print(event["text"], end="", flush=True)
    elif event["type"] == "done":
        final_message = event["message"]
```

`MinimaxPipeline`, `AnthropicAPIPipeline`, `OpenAIPipeline`, and `MockPipeline`
stream token by token. Every other pipeline inherits the `LLMPipeline` default, which calls
`generate` and yields only the final `done` event, so the interface is safe to
call on any pipeline. Batched (list-of-lists) inputs always fall back to the
single `done` event.

## Example: Generate An Image

`image_pipeline_from_config` mirrors `pipeline_from_config` one level down: pick
a provider, ask for one image, get bytes back.

```python
from llm_utils import (
    ImagePipelineConfig,
    ImageProviderError,
    ImageRequestRejected,
    image_pipeline_from_config,
)

cfg = ImagePipelineConfig(
    model="gpt-image-2.5-flare",   # or "image-01" for MiniMax, "mock" offline
    pipeline_type="openai",        # "openai" | "minimax" | "mock"
    aspect_ratio="landscape",      # "square" | "landscape" | "portrait"
    quality="medium",              # OpenAI only
    output_format="png",           # OpenAI only; MiniMax always returns JPEG
    timeout=120,
)

try:
    result = image_pipeline_from_config(cfg).generate_image(
        "a labeled diagram of the water cycle"
    )
except ImageRequestRejected as exc:
    print(f"The provider refused this prompt: {exc}")   # do not retry
except ImageProviderError as exc:
    print(f"The provider failed: {exc}")                # retrying may work
else:
    print(result.mime_type, result.width, result.height, len(result.data))
    open("water-cycle.png", "wb").write(result.data)
```

`ImageResult` carries `data`, `mime_type`, `width`, `height`, `model`,
`provider`, and `revised_prompt` (OpenAI only, `None` elsewhere). Width and
height are read out of the file's own header, so they are `None` for a
container these helpers cannot parse rather than a guess.

The two exception types are the whole error contract:
`ImageRequestRejected` means the provider said no to this prompt (content
policy, an over-long prompt, invalid parameters) and the caller should show the
reason; `ImageProviderError` means the provider itself failed and a retry may
succeed. Everything else -- authentication, an org not yet enabled for a model,
rate limits, timeouts -- propagates as the SDK's or `requests`' own exception.

`MockImagePipeline` needs no credentials and no network: it returns a
deterministic solid-colour PNG whose colour comes from the prompt. A prompt
containing `[reject]` raises `ImageRequestRejected` and one containing `[fail]`
raises `ImageProviderError`, which is how callers test both failure paths.

Configuration read from the environment or the nearest `.env`:

- `OPENAI_API_KEY` (required for `openai`) and `OPENAI_BASE_URL` (optional)
- `MINIMAX_API_KEY` (required for `minimax`)
- `MINIMAX_IMAGE_BASE_URL` (optional, defaults to `https://api.minimax.io/v1`;
  set it to `https://api.minimaxi.com/v1` for the China region)

Aspect ratios map to `1024x1024` / `1536x1024` / `1024x1536` on OpenAI and to
`1:1` / `3:2` / `2:3` on MiniMax.

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

The image pipelines have their own file, which runs without credentials or a
network:

```bash
pytest test/test_image_pipelines.py -q
```
