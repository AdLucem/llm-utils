"""Shared `.env` reading for the provider helpers.

Every helper needs the same three things: read a `KEY=VALUE` file, merge the
`.env` files found from the working directory upwards, and strip the quotes a
shell leaves behind. They used to be duplicated per provider, and the copies
had drifted -- `request_minimax` only ever looked at `Path.cwd()/.env`, so a
MiniMax key in the repository root was invisible whenever the server was
started from a subdirectory, while OpenAI's copy walked upwards and found it.
"""

import os
from pathlib import Path
from typing import Dict, Optional


def read_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """Read simple KEY=VALUE pairs from a .env file when present.

    Defaults to `.env` in the current working directory, which is what the
    per-provider copies of this function used to do.
    """

    path = (Path.cwd() / ".env") if path is None else Path(path)

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


def dotenv_values() -> Dict[str, str]:
    """Merge every `.env` found from the working directory up to the filesystem root.

    The nearest `.env` wins. Walking upward matters because the API server is
    launched from `centaurus/` while the checkout keeps its `.env` at the repo
    root, one level above.
    """

    values: Dict[str, str] = {}
    for directory in (Path.cwd(), *Path.cwd().parents):
        for key, value in read_dotenv(directory / ".env").items():
            values.setdefault(key, value)
    return values


def clean_env_value(value: Optional[str]) -> Optional[str]:
    """Strip whitespace and optional quote characters from environment values."""

    if value is None:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def env_value(name: str, values: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Read `name` from the process environment, falling back to the `.env` files."""

    values = dotenv_values() if values is None else values
    return clean_env_value(os.environ.get(name) or values.get(name))


__all__ = ["clean_env_value", "dotenv_values", "env_value", "read_dotenv"]
