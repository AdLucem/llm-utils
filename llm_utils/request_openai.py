"""Send chat completion requests to an OpenAI model via the OpenAI SDK."""

import os
from pathlib import Path
from typing import Dict, List, Optional

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

#: Kept as private aliases so existing callers and tests that reach for the
#: old module-level names keep working; the implementations now live in `_env`.
_read_dotenv = read_dotenv
_dotenv_values = dotenv_values
_clean_env_value = clean_env_value


#: Model families that only accept the API's default sampling settings and
#: spell the output cap `max_completion_tokens`.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _get_openai_credentials() -> Dict[str, str]:
    """Load OpenAI credentials from .env and environment variables.

    Only `OPENAI_API_KEY` is required; `OPENAI_BASE_URL` defaults to the public
    API and is there for Azure or gateway deployments.
    """

    values = dotenv_values()

    api_key = clean_env_value(
        os.environ.get("OPENAI_API_KEY") or values.get("OPENAI_API_KEY")
    )
    base_url = clean_env_value(
        os.environ.get("OPENAI_BASE_URL") or values.get("OPENAI_BASE_URL")
    ) or DEFAULT_BASE_URL

    if not api_key:
        raise ValueError(
            "Missing OpenAI configuration in .env or environment: OPENAI_API_KEY"
        )

    if not base_url.startswith(("http://", "https://")):
        raise ValueError("OPENAI_BASE_URL must start with http:// or https://.")

    return {"api_key": api_key, "base_url": base_url}


def _client(cfg: RequestConfig):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required to use the OpenAI helpers. Install `openai`."
        ) from exc

    credentials = _get_openai_credentials()
    return OpenAI(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=cfg.timeout,
    )


def _completion_kwargs(cfg: RequestConfig, messages: List[Dict[str, str]]) -> dict:
    """Build the request body, omitting parameters the model would reject."""

    kwargs = {
        "model": cfg.model,
        "messages": messages,
        "max_completion_tokens": cfg.max_new_tokens,
    }

    if not cfg.model.lower().startswith(REASONING_MODEL_PREFIXES):
        kwargs["temperature"] = cfg.temperature

    return kwargs


def _message_from_choice(choice) -> Dict[str, str]:
    message = choice.message
    return {
        "role": "assistant",
        "thinking": (getattr(message, "reasoning_content", None) or "").strip(),
        "content": (message.content or "").strip(),
    }


def openai_chat_completion(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """Send one chat completion request to OpenAI and return the assistant message."""

    response = _client(cfg).chat.completions.create(**_completion_kwargs(cfg, messages))
    return _message_from_choice(response.choices[0])


def openai_chat_completion_stream(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
):
    """Stream one OpenAI chat completion.

    Yields `{"type": "delta"|"thinking_delta", "text": str}` as tokens arrive and
    finishes with `{"type": "done", "message": {...}}`, whose message has the same
    shape `openai_chat_completion` returns.
    """

    thinking, text = "", ""
    stream = _client(cfg).chat.completions.create(
        stream=True,
        **_completion_kwargs(cfg, messages),
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            thinking += reasoning
            yield {"type": "thinking_delta", "text": reasoning}
        if delta.content:
            text += delta.content
            yield {"type": "delta", "text": delta.content}

    yield {
        "type": "done",
        "message": {
            "role": "assistant",
            "thinking": thinking.strip(),
            "content": text.strip(),
        },
    }


def openai_chat_completion_batch(
    cfg: RequestConfig,
    requests_messages: List[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Send multiple OpenAI chat completion requests and return assistant messages."""

    outputs: List[Dict[str, str]] = []
    for messages in requests_messages:
        outputs.append(openai_chat_completion(cfg, messages))
    return outputs
