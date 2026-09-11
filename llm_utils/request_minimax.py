"""Send chat completion requests to a MiniMax model via the OpenAI SDK."""

import os
from pathlib import Path
from typing import Dict, Iterator, List
import anthropic

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


def _get_minimax_credentials() -> Dict[str, str]:
    """Load MiniMax credentials from .env and environment variables."""

    dotenv_values = _read_dotenv()

    api_key = _clean_env_value(os.environ.get("MINIMAX_API_KEY") or dotenv_values.get("MINIMAX_API_KEY"))
    base_url = _clean_env_value(os.environ.get("MINIMAX_BASE_URL") or dotenv_values.get("MINIMAX_BASE_URL"))

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


def _clean_env_value(value: str | None) -> str | None:
    """Strip whitespace and optional quote characters from environment values."""
    if value is None:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


# Status codes worth retrying / telling the caller "try again": rate
# limiting, overload, and generic server-side/gateway trouble. Anything else
# (bad request, auth, not found, ...) will not clear up on its own.
_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}

# A couple of extra attempts on top of the Anthropic SDK's own built-in
# retry (2, by default) for the transient overload/rate-limit case the
# MiniMax backend hits occasionally — cheap insurance against a brief blip
# without the request hanging for anywhere near the "1-5 minutes" the
# backend's own overloaded-error message warns about.
_ANTHROPIC_CLIENT_MAX_RETRIES = 4


class MinimaxBackendError(Exception):
    """Raised when the MiniMax/Anthropic-compatible backend fails or errors.

    Carries a short, human-readable `message` (via `str(exc)`) instead of
    the raw Anthropic SDK error dump, plus:

    - `retryable`: True for a condition expected to clear up on its own
      (overloaded, rate-limited, a transient server/connection error) —
      false for one that will keep failing until something changes (bad
      request, bad credentials, etc.).
    - `status_code`: the HTTP status from the backend, when there is one.
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def _raise_as_backend_error(exc: Exception) -> None:
    """Re-raise `exc` (must be the exception currently being handled) as a
    `MinimaxBackendError` with a clean message, for the Anthropic SDK error
    types worth translating; re-raises `exc` unchanged otherwise.
    """
    if isinstance(exc, anthropic.APIStatusError):
        status_code = exc.status_code
        retryable = status_code in _RETRYABLE_STATUS_CODES
        if isinstance(exc, anthropic.OverloadedError):
            message = (
                "The MiniMax backend is temporarily overloaded (HTTP 529). "
                "This usually clears up within a few minutes — please try again shortly."
            )
        elif isinstance(exc, anthropic.RateLimitError):
            message = (
                "The MiniMax backend is rate-limiting requests right now. "
                "Please wait a moment and try again."
            )
        elif retryable:
            message = (
                f"The MiniMax backend returned a temporary error (HTTP {status_code}). "
                "Please try again shortly."
            )
        else:
            message = f"The MiniMax backend rejected the request (HTTP {status_code}): {exc.message}"
        raise MinimaxBackendError(message, retryable=retryable, status_code=status_code) from exc

    if isinstance(exc, anthropic.APIConnectionError):
        raise MinimaxBackendError(
            "Could not reach the MiniMax backend. Please check connectivity and try again.",
            retryable=True,
        ) from exc

    raise


def minimax_chat_completion(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """Send one chat completion request to MiniMax and return the assistant message.

    Raises `MinimaxBackendError` (see above) instead of a raw Anthropic SDK
    exception when the backend errors out or is unreachable.
    """

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
        max_retries=_ANTHROPIC_CLIENT_MAX_RETRIES,
    )

    try:
        response = client.messages.create(
            model=cfg.model,
            messages=messages,
            max_tokens=cfg.max_new_tokens,
        )
    except anthropic.AnthropicError as exc:
        _raise_as_backend_error(exc)

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
) -> Iterator[Dict[str, str]]:
    """Stream one chat completion request to MiniMax.

    Yields ``{"type": "delta" | "thinking_delta", "text": str}`` as tokens
    arrive, then a final ``{"type": "done", "message": {...}}`` carrying the
    same shape `minimax_chat_completion` returns, so a caller that only wants
    the finished message can drain the generator and use its last event.

    Raises `MinimaxBackendError` (see above) instead of a raw Anthropic SDK
    exception when the backend errors out or is unreachable, whether that
    happens before the stream opens or partway through it.
    """

    credentials = _get_minimax_credentials()
    client = anthropic.Anthropic(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=cfg.timeout,
        max_retries=_ANTHROPIC_CLIENT_MAX_RETRIES,
    )

    thinking, text = "", ""
    block_types: Dict[int, str] = {}
    try:
        with client.messages.stream(
            model=cfg.model,
            messages=messages,
            max_tokens=cfg.max_new_tokens,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block_types[event.index] = event.content_block.type
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        thinking += event.delta.thinking
                        yield {"type": "thinking_delta", "text": event.delta.thinking}
                    elif event.delta.type == "text_delta":
                        text += event.delta.text
                        yield {"type": "delta", "text": event.delta.text}
                elif event.type == "content_block_stop":
                    # Mirrors `minimax_chat_completion`, which joins block texts with "\n".
                    block_type = block_types.get(event.index)
                    if block_type == "thinking":
                        thinking += "\n"
                    elif block_type == "text":
                        text += "\n"
    except anthropic.AnthropicError as exc:
        _raise_as_backend_error(exc)

    yield {
        "type": "done",
        "message": {"role": "assistant", "thinking": thinking.strip(), "content": text.strip()},
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
