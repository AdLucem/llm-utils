"""Send chat completion requests to an OpenAI-compatible endpoint."""

import logging
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


DEFAULT_DOTENV_PATH = Path.cwd() / ".env"


def _read_dotenv(path: Path = DEFAULT_DOTENV_PATH) -> Dict[str, str]:
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


def configure_logging(level: str) -> None:
    """Configure global logger."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _openai_api_key(cfg: RequestConfig) -> str:
    """Resolve an API key from config, environment vars, or .env fallback."""
    token = _clean_optional(getattr(cfg, "token", None))
    if token:
        return token

    dotenv_values = _read_dotenv()

    token = _clean_optional(os.environ.get("OPENAI_API_KEY") or dotenv_values.get("OPENAI_API_KEY"))
    if token:
        return token

    token = _clean_optional(os.environ.get("OPENROUTER_API_KEY") or dotenv_values.get("OPENROUTER_API_KEY"))
    if token:
        return token

    raise ValueError(
        "Missing OpenAI-compatible API token. Set cfg.token, OPENAI_API_KEY, "
        "or OPENROUTER_API_KEY (as an environment variable or in .env)."
    )


def _openai_base_url(cfg: RequestConfig) -> Optional[str]:
    """Resolve an optional OpenAI-compatible base URL."""
    base_url = _clean_optional(getattr(cfg, "base_url", None))

    if not base_url:
        dotenv_values = _read_dotenv()
        base_url = _clean_optional(os.environ.get("OPENAI_BASE_URL") or dotenv_values.get("OPENAI_BASE_URL"))
        if not base_url:
            base_url = _clean_optional(os.environ.get("OPENROUTER_BASE_URL") or dotenv_values.get("OPENROUTER_BASE_URL"))

    if base_url and not base_url.startswith(("http://", "https://")):
        raise ValueError("OpenAI-compatible base_url must start with http:// or https://.")
    return base_url


def _openrouter_headers() -> Dict[str, str]:
    """Return optional OpenRouter attribution headers from the environment."""
    headers: Dict[str, str] = {}

    referer = _clean_optional(os.environ.get("OPENROUTER_HTTP_REFERER"))
    if referer:
        headers["HTTP-Referer"] = referer

    title = _clean_optional(
        os.environ.get("OPENROUTER_TITLE")
        or os.environ.get("X_OPENROUTER_TITLE")
    )
    if title:
        headers["X-OpenRouter-Title"] = title

    return headers


def _extract_content_from_message(message) -> str:
    """Normalize an OpenAI chat message into a plain assistant string."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
                continue

            block_text = getattr(block, "text", None)
            if isinstance(block_text, str):
                parts.append(block_text)

        text = "".join(parts).strip()
        if text:
            return text

    raise ValueError(f"Unexpected assistant message content: {message}")


def openai_chat_completion(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """Send one OpenAI-compatible chat completion request and return assistant message."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required to use OpenAI-compatible helpers. "
            "Install `llm-utils[openai]`."
        ) from exc

    client_kwargs = {
        "api_key": _openai_api_key(cfg),
        "timeout": cfg.timeout,
    }

    base_url = _openai_base_url(cfg)
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
        extra_headers=_openrouter_headers() or None,
    )

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected API response shape: {response}") from exc

    content = _extract_content_from_message(message)
    return {"role": "assistant", "content": content}


def openai_chat_completion_batch(
    cfg: RequestConfig,
    requests_messages: List[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Send multiple OpenAI-compatible chat completion requests."""
    outputs: List[Dict[str, str]] = []
    for messages in requests_messages:
        outputs.append(openai_chat_completion(cfg, messages))
    return outputs
