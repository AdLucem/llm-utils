"""Integration tests for the live SGLang-backed pipeline.

These tests intentionally hit a real SGLang server so we verify the actual HTTP
request/response path used by production code.
"""

from llm_utils.pipelines import PipelineConfig, SGLangPipeline


QWEN_SGLANG_CFG = PipelineConfig(
    model="Qwen/Qwen3.5-35B-A3B-FP8",
    pipeline_type="sglang",
    host="0.0.0.0",
    port=30000,
    temperature=0.0,
    max_new_tokens=32,
    timeout=60,
    log_level="INFO",
)


def test_sglang_pipeline_single_request_returns_assistant_message():
    """The live SGLang pipeline should return the standard assistant dict shape."""

    pipeline = SGLangPipeline(QWEN_SGLANG_CFG)
    response = pipeline.generate(
        [
            {"role": "system", "content": "Answer with exactly the word PINEAPPLE."},
            {"role": "user", "content": "Respond now."},
        ]
    )

    assert response["role"] == "assistant"
    assert isinstance(response["content"], str)
    assert response["content"].strip()
    assert "pineapple" in response["content"].lower()


def test_sglang_pipeline_batch_requests_return_one_response_per_prompt():
    """Parallel message batches should map to one assistant response each."""

    pipeline = SGLangPipeline(QWEN_SGLANG_CFG)
    response = pipeline.generate(
        [
            [
                {
                    "role": "system",
                    "content": "Answer with exactly the word SATURN.",
                },
                {"role": "user", "content": "Respond now."},
            ],
            [
                {
                    "role": "system",
                    "content": "Answer with exactly the word NEPTUNE.",
                },
                {"role": "user", "content": "Respond now."},
            ],
        ]
    )

    assert len(response) == 2
    assert response[0]["role"] == "assistant"
    assert response[1]["role"] == "assistant"
    assert "saturn" in response[0]["content"].lower()
    assert "neptune" in response[1]["content"].lower()
