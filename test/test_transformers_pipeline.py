"""Tests for the local Hugging Face pipeline implementation.

These tests use lightweight fake tokenizer/model objects so we can verify the
pipeline contract without requiring heavyweight runtime dependencies in the test
environment.
"""

import sys
import types

import pytest

from llm_utils.pipelines import PipelineConfig, TransformersPipeline


class _FakeVector(list):
    """Small list wrapper that mimics the bits of a tensor API we need."""

    def tolist(self):
        return list(self)


class _FakeMatrix(list):
    """List-of-lists wrapper that supports `.to()` and row-wise `.sum()`."""

    def to(self, _device):
        return self

    def sum(self, dim):
        if dim != 1:
            raise AssertionError(f"Unexpected dim for fake sum: {dim}")
        return _FakeVector([sum(row) for row in self])


class _FakeTokenizer:
    """Tokenizer fake that records prompts and decodes fixed token ids."""

    chat_template = "fake-template"
    pad_token_id = 0
    eos_token_id = 99
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.rendered_prompts = []
        self.decode_map = {
            (101,): "alpha",
            (102,): "beta",
            (103,): "gamma",
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        rendered = " || ".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        self.rendered_prompts.append(rendered)
        return rendered

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True):
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        input_ids = _FakeMatrix([[10, 11] for _ in prompts])
        attention_mask = _FakeMatrix([[1, 1] for _ in prompts])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return self.decode_map[tuple(token_ids)]


class _FakeModel:
    """Minimal fake CausalLM that captures generate kwargs."""

    def __init__(self):
        self.device = types.SimpleNamespace(type="cpu")
        self.last_generate_kwargs = None

    def eval(self):
        return self

    def to(self, _device):
        return self

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        batch_size = len(kwargs["input_ids"])
        outputs = []
        generated_tokens = [101, 102, 103]
        for idx in range(batch_size):
            outputs.append([10, 11, generated_tokens[idx]])
        return outputs


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_torch(monkeypatch):
    """Provide the tiny subset of the torch API used by the pipeline."""

    torch_module = types.SimpleNamespace(
        no_grad=lambda: _FakeNoGrad(),
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    return torch_module


@pytest.fixture
def transformers_pipeline(monkeypatch, fake_torch):
    """Create a pipeline instance wired to fake model/tokenizer objects."""

    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    def fake_init_transformers(self, cfg):
        assert cfg.model == "fake-model"
        return tokenizer, model

    monkeypatch.setattr(
        TransformersPipeline,
        "_init_transformers",
        fake_init_transformers,
    )

    cfg = PipelineConfig(
        model="fake-model",
        pipeline_type="transformers",
        temperature=0.6,
        max_new_tokens=16,
        top_p=0.8,
        top_k=12,
        log_level="DEBUG",
    )
    pipeline = TransformersPipeline(cfg)
    return pipeline, tokenizer, model


def test_transformers_pipeline_single_request_returns_assistant_message(
    transformers_pipeline,
):
    """A single message list should return one assistant-shaped response dict."""

    pipeline, tokenizer, model = transformers_pipeline

    response = pipeline.generate(
        [
            {"role": "system", "content": "Reply with a single token."},
            {"role": "user", "content": "alpha"},
        ]
    )

    assert response == {"role": "assistant", "content": "alpha"}
    assert tokenizer.rendered_prompts == [
        "system:Reply with a single token. || user:alpha"
    ]
    assert model.last_generate_kwargs["max_new_tokens"] == 16
    assert model.last_generate_kwargs["do_sample"] is True


def test_transformers_pipeline_parallel_requests_return_one_response_each(
    transformers_pipeline,
):
    """A list of message batches should produce one assistant response per batch."""

    pipeline, tokenizer, _model = transformers_pipeline

    response = pipeline.generate(
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]
    )

    assert response == [
        {"role": "assistant", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
    ]
    assert tokenizer.rendered_prompts == [
        "user:first",
        "user:second",
    ]


def test_transformers_pipeline_temperature_zero_switches_to_greedy_generation(
    monkeypatch,
    fake_torch,
):
    """Greedy generation should disable sampling-only kwargs."""

    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    monkeypatch.setattr(
        TransformersPipeline,
        "_init_transformers",
        lambda self, cfg: (tokenizer, model),
    )

    pipeline = TransformersPipeline(
        PipelineConfig(
            model="fake-model",
            pipeline_type="transformers",
            temperature=0,
            max_new_tokens=8,
            top_p=0.9,
            top_k=7,
        )
    )

    response = pipeline.generate([{"role": "user", "content": "greedy"}])

    assert response == {"role": "assistant", "content": "alpha"}
    assert model.last_generate_kwargs["do_sample"] is False
    assert model.last_generate_kwargs["temperature"] is None
    assert model.last_generate_kwargs["top_p"] is None
    assert "top_k" not in model.last_generate_kwargs
