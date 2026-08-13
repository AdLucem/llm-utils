"""Send chat completion requests to an Anthropic Messages API-compatible endpoint."""

import logging
from typing import Dict, List, Optional

try:
    from .llm_configs import RequestConfig
except ImportError:  # pragma: no cover - direct script fallback
    try:
        from llm_utils.llm_configs import RequestConfig
    except ImportError:
        from llm_configs import RequestConfig


def configure_logging(level: str) -> None:
    """Configure global logger."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _extract_system_prompt(messages: List[Dict[str, str]]) -> Optional[str]:
    """Collect system messages into the Anthropic `system` parameter."""
    system_parts: List[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content", "")
        if content:
            system_parts.append(str(content).strip())

    if not system_parts:
        return None
    return "\n\n".join(system_parts)


def _to_anthropic_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Translate repository messages into Anthropic Messages API format."""
    anthropic_messages: List[Dict[str, str]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue

        content = message.get("content", "")
        if content is None:
            content = ""

        anthropic_messages.append(
            {
                "role": role,
                "content": str(content),
            }
        )

    return anthropic_messages


def _extract_text_from_response(response) -> str:
    """Normalize Anthropic response content into a plain assistant string."""
    content = getattr(response, "content", None)
    if content is None:
        raise ValueError(f"Unexpected API response shape: {response}")

    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
        raise ValueError(f"Unexpected empty assistant content in API response: {response}")

    text_parts: List[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            block_text = getattr(block, "text", "")
            if block_text:
                text_parts.append(str(block_text))
            continue

        if isinstance(block, dict) and block.get("type") == "text":
            block_text = block.get("text", "")
            if block_text:
                text_parts.append(str(block_text))

    text = "".join(text_parts).strip()
    if not text:
        raise ValueError(f"Unexpected assistant content in API response: {response}")
    return text


def anthropic_messages_completion(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """Send one Messages API request and return the assistant message."""

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic is required to use Anthropic helpers. Install `llm-utils[anthropic]`."
        ) from exc

    client = Anthropic(
        api_key=cfg.token,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
    )

    response = client.messages.create(
        model=cfg.model,
        system=_extract_system_prompt(messages),
        messages=_to_anthropic_messages(messages),
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
    )

    content = _extract_text_from_response(response)
    return {"role": "assistant", "content": content}


def anthropic_messages_completion_batch(
    cfg: RequestConfig,
    requests_messages: List[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Send multiple Messages API requests and return assistant messages."""
    outputs: List[Dict[str, str]] = []
    for messages in requests_messages:
        outputs.append(anthropic_messages_completion(cfg, messages))
    return outputs
