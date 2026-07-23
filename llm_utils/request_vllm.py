"""Helpers for local chat generation with vLLM."""

import logging
from typing import Dict, List, Optional, Tuple

try:
    from .request_sglang import configure_logging, parse_assistant_message
except ImportError:  # pragma: no cover - direct script fallback
    try:
        from llm_utils.request_sglang import configure_logging, parse_assistant_message
    except ImportError:
        from request_sglang import configure_logging, parse_assistant_message


def _build_sampling_params(cfg):
    """Translate pipeline/request config into vLLM sampling params."""

    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("vllm is required to use VLLMPipeline.") from exc

    sampling_kwargs = {
        "max_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
    }

    if getattr(cfg, "top_p", None) is not None:
        sampling_kwargs["top_p"] = cfg.top_p
    if getattr(cfg, "top_k", None) is not None and cfg.top_k > 0:
        sampling_kwargs["top_k"] = cfg.top_k

    return SamplingParams(**sampling_kwargs)


def _messages_to_prompt(messages, tokenizer) -> str:
    """Render a chat prompt string from structured messages."""

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered_messages = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        if content:
            rendered_messages.append(f"{role}: {content}")
    rendered_messages.append("ASSISTANT:")
    return "\n\n".join(rendered_messages)


def init_vllm(cfg) -> Tuple[object, object]:
    """Create a reusable vLLM engine and tokenizer pair."""

    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("vllm is required to use VLLMPipeline.") from exc

    llm_kwargs = {"model": cfg.model}
    if getattr(cfg, "dtype", "auto") != "auto":
        llm_kwargs["dtype"] = cfg.dtype

    llm = LLM(**llm_kwargs)

    if not hasattr(llm, "get_tokenizer"):
        raise AttributeError("The loaded vLLM engine does not expose get_tokenizer().")

    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def vllm_chat_completion(
    cfg,
    messages: List[Dict[str, str]],
    llm: Optional[object] = None,
    tokenizer: Optional[object] = None,
) -> Dict[str, str]:
    """Send one chat completion request through a local vLLM engine."""

    responses = vllm_chat_completion_batch(
        cfg=cfg,
        requests_messages=[messages],
        llm=llm,
        tokenizer=tokenizer,
    )
    return responses[0]


def vllm_chat_completion_batch(
    cfg,
    requests_messages: List[List[Dict[str, str]]],
    llm: Optional[object] = None,
    tokenizer: Optional[object] = None,
) -> List[Dict[str, str]]:
    """Send multiple chat completion requests through a local vLLM engine."""

    if llm is None or tokenizer is None:
        llm, tokenizer = init_vllm(cfg)

    logging.debug(f"REQUESTS MESSAGES: {requests_messages}")
    prompts = [_messages_to_prompt(messages, tokenizer) for messages in requests_messages]
    sampling_params = _build_sampling_params(cfg)

    outputs = llm.generate(prompts, sampling_params)
    responses = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        responses.append(parse_assistant_message(text))

    logging.debug("vLLM rendered %d prompt(s).", len(prompts))
    return responses


__all__ = [
    "configure_logging",
    "init_vllm",
    "vllm_chat_completion",
    "vllm_chat_completion_batch",
]
