"""Send chat completion requests to a MiniMax model via the OpenAI SDK."""

import os
from pathlib import Path
from typing import Dict, List
import anthropic 

try:
    from ._env import clean_env_value, dotenv_values, read_dotenv
    from .llm_configs import RequestConfig
except ImportError:  # pragma: no cover - direct script fallback
    try:
        from llm_utils._env import clean_env_value, dotenv_values, read_dotenv
        from llm_utils.llm_configs import RequestConfig
    except ImportError:
        from _env import clean_env_value, dotenv_values, read_dotenv
        from llm_configs import RequestConfig


DEFAULT_DOTENV_PATH = Path.cwd() / ".env"

#: Kept as private aliases so existing callers keep working; the implementations
#: now live in `_env`, which also fixes this module only ever reading the
#: working directory's `.env` and missing keys kept one level up.
_read_dotenv = read_dotenv
_clean_env_value = clean_env_value


def _get_minimax_credentials() -> Dict[str, str]:
    """Load MiniMax credentials from .env and environment variables."""

    values = dotenv_values()

    api_key = clean_env_value(os.environ.get("MINIMAX_API_KEY") or values.get("MINIMAX_API_KEY"))
    base_url = clean_env_value(os.environ.get("MINIMAX_BASE_URL") or values.get("MINIMAX_BASE_URL"))

    missing = []
    if not api_key:
        missing.append("MINIMAX_API_KEY")
    if not base_url:
        missing.append("MINIMAX_BASE_URL")

    if missing:
        raise ValueError(
            "Missing MiniMax configuration in .env or environment: "
            + ", ".join(missing)
        )

    if not base_url.startswith(("http://", "https://")):
        raise ValueError("MINIMAX_BASE_URL must start with http:// or https://.")

    return {"api_key": api_key, "base_url": base_url}


def minimax_chat_completion(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """Send one chat completion request to MiniMax and return the assistant message."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required to use MiniMax helpers. Install `llm-utils[minimax]`."
        ) from exc

    credentials = _get_minimax_credentials()
    client = anthropic.Anthropic(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=cfg.timeout,
    )

    response = client.messages.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_new_tokens,
    )

    thinking, text = "", ""
    for block in response.content:
        if block.type == "thinking":
            thinking += f"{block.thinking}\n"
        elif block.type == "text":
            text += f"{block.text}\n"
    return {"role": "assistant", 
            "thinking": thinking.strip(),
            "content": text.strip()}


def minimax_chat_completion_stream(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
):
    """Stream one MiniMax chat completion.

    Yields `{"type": "delta"|"thinking_delta", "text": str}` as tokens arrive and
    finishes with `{"type": "done", "message": {...}}`, whose message has the same
    shape `minimax_chat_completion` returns.
    """

    credentials = _get_minimax_credentials()
    client = anthropic.Anthropic(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=cfg.timeout,
    )

    thinking, text = "", ""
    with client.messages.stream(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_new_tokens,
    ) as stream:
        for event in stream:
            if getattr(event, "type", None) != "content_block_delta":
                continue
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                text += delta.text
                yield {"type": "delta", "text": delta.text}
            elif delta_type == "thinking_delta":
                thinking += delta.thinking
                yield {"type": "thinking_delta", "text": delta.thinking}

    yield {
        "type": "done",
        "message": {
            "role": "assistant",
            "thinking": thinking.strip(),
            "content": text.strip(),
        },
    }


def minimax_chat_completion_batch(
    cfg: RequestConfig,
    requests_messages: List[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Send multiple MiniMax chat completion requests and return assistant messages."""

    outputs: List[Dict[str, str]] = []
    for messages in requests_messages:
        outputs.append(minimax_chat_completion(cfg, messages))
    return outputs


if __name__ == "__main__":

    creds = _get_minimax_credentials()
    print(creds)

    client = anthropic.Anthropic(
        api_key=creds["api_key"],
        base_url=creds["base_url"]
    )
    msg = [
        {
            "role": "system",
            "content": "You are an evil wizard."
        },
        {
            "role": "user",
            "content": "Hi, how are you?"
        }
    ]
    cfg = RequestConfig(
        model="Minimax-M2.7",
        host="",
        port="",
        temperature=0.7,
        max_new_tokens=1024,
        timeout=1000,
        log_level="DEBUG"
    )

    
    rsp = minimax_chat_completion_batch(cfg=cfg, 
                                        requests_messages=[msg, msg])
    print(rsp)
