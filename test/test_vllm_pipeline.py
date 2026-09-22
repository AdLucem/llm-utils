
"""Integration test for the live local vLLM request helpers."""

import pytest

from llm_utils.pipelines import PipelineConfig, VLLMPipeline
from llm_utils.request_vllm import (
    init_vllm,
    vllm_chat_completion,
    vllm_chat_completion_batch,
)


VLLM_CFG = PipelineConfig(
    model="meta-llama/Llama-3.2-1B",
    pipeline_type="vllm",
    temperature=0.0,
    max_new_tokens=16,
    top_p=1.0,
    top_k=1,
    log_level="INFO",
)


def test_vllm_request_helpers_load_model_and_generate_hello_world_response():
    """vLLM should load the configured model and return assistant-shaped output."""

    pytest.importorskip("vllm")
    messages = [{"role": "user", "content": "Hello world"}]

    llm, tokenizer = init_vllm(VLLM_CFG)
    single_response = vllm_chat_completion(
        VLLM_CFG,
        messages,
        llm=llm,
        tokenizer=tokenizer,
    )
    batch_response = vllm_chat_completion_batch(
        VLLM_CFG,
        [messages],
        llm=llm,
        tokenizer=tokenizer,
    )

    assert tokenizer is not None
    assert single_response["role"] == "assistant"
    assert isinstance(single_response["content"], str)
    assert single_response["content"].strip() != ""
    assert len(batch_response) == 1
    assert batch_response[0]["role"] == "assistant"
    assert isinstance(batch_response[0]["content"], str)
    assert batch_response[0]["content"].strip() != ""


def test_vllm_pipeline_loads_model_and_generates_hello_world_response():
    """The live VLLMPipeline should return the standard assistant dict shape."""

    pytest.importorskip("vllm")

    pipeline = VLLMPipeline(VLLM_CFG)
    response = pipeline.generate([{"role": "user", "content": "Hello world"}])

    assert response["role"] == "assistant"
    assert isinstance(response["content"], str)
    assert response["content"].strip() != ""
