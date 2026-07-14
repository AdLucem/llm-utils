# Tests

This directory contains focused tests for the two production pipeline
implementations in `llm_utils.pipelines`.

## Files

- `test/test_transformers_pipeline.py`
  Unit-style tests for `TransformersPipeline`. These tests do not require
  `torch` or `transformers` to be installed. Instead, they use small fake
  tokenizer/model objects so we can verify:
  - the pipeline accepts the same input shapes as `SGLangPipeline`
  - single-message generation returns one assistant message dictionary
  - parallel list-of-message-list generation returns one assistant response per
    request
  - greedy mode is configured correctly when `temperature=0`

- `test/test_sglang_pipeline.py`
  Integration tests for `SGLangPipeline`. These tests assume a live SGLang
  server is reachable at `0.0.0.0:30000` and serving
  `Qwen/Qwen3.5-35B-A3B-FP8`.
  They verify:
  - a single request returns an assistant response dictionary
  - batched requests return one assistant response per prompt
  - the response format matches the contract used elsewhere in this repo

## Running The Tests

Run the whole suite:

```bash
python3 -m pytest test
```

Run only the fast unit tests:

```bash
python3 -m pytest test/test_transformers_pipeline.py
```

Run only the live SGLang integration tests:

```bash
python3 -m pytest test/test_sglang_pipeline.py
```

## Reading Notes

The Transformers tests intentionally use simple fake objects instead of mocking
every internal method away. That keeps the tests readable while still exercising
the pipeline's batching, prompt formatting, generation, and decoding flow.

The SGLang tests use short deterministic prompts and low token counts to keep
them fast and reduce output variability.
