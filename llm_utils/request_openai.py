"""Send chat completion requests to an OpenAI model via the OpenAI SDK."""

import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .llm_configs import RequestConfig
except ImportError:  # pragma: no cover - direct script fallback
    try:
        from llm_utils.llm_configs import RequestConfig
    except ImportError:
        from llm_configs import RequestConfig


#: Model families that only accept the API's default sampling settings and
#: spell the output cap `max_completion_tokens`.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _read_dotenv(path: Path) -> Dict[str, str]:
    """Read simple KEY=VALUE pairs from a .env file when present."""

    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        values[key] = value

    return values


def _dotenv_values() -> Dict[str, str]:
    """Merge every `.env` found from the working directory up to the filesystem root.

    The nearest `.env` wins. Walking upward matters because the API server is
    launched from `centaurus/` while the checkout keeps its `.env` at the repo
    root, one level above.
    """

    values: Dict[str, str] = {}
    for directory in (Path.cwd(), *Path.cwd().parents):
        for key, value in _read_dotenv(directory / ".env").items():
            values.setdefault(key, value)
    return values


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    """Strip whitespace and optional quote characters from environment values."""
    if value is None:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _get_openai_credentials() -> Dict[str, str]:
    """Load OpenAI credentials from .env and environment variables.

    Only `OPENAI_API_KEY` is required; `OPENAI_BASE_URL` defaults to the public
    API and is there for Azure or gateway deployments.
    """

    dotenv_values = _dotenv_values()

    api_key = _clean_env_value(
        os.environ.get("OPENAI_API_KEY") or dotenv_values.get("OPENAI_API_KEY")
    )
    base_url = _clean_env_value(
        os.environ.get("OPENAI_BASE_URL") or dotenv_values.get("OPENAI_BASE_URL")
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
