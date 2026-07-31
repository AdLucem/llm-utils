"""Send chat completion requests to a MiniMax model via the OpenAI SDK."""

import os
from pathlib import Path
from typing import Dict, List

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

    api_key = os.environ.get("MINIMAX_API_KEY") or dotenv_values.get("MINIMAX_API_KEY")
    base_url = os.environ.get("MINIMAX_BASE_URL") or dotenv_values.get("MINIMAX_BASE_URL")

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

    return {"api_key": api_key, "base_url": base_url}


def minimax_chat_completion(
    cfg: RequestConfig,
    messages: List[Dict[str, str]],
) -> Dict[str, str]:
    """Send one chat completion request to MiniMax and return the assistant message."""

    credentials = _get_minimax_credentials()
    client = anthropic.Anthropic(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=cfg.timeout,
    )

    response = client.messages.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
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
