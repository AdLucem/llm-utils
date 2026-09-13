"""Generate one image with MiniMax's image API, over plain HTTP.

MiniMax ships no SDK for this endpoint, so `requests` (already a dependency) is
used directly. Two things about the API are easy to get wrong and are handled
here: a failure can arrive inside a 200 response as a non-zero
`base_resp.status_code`, and returned URLs expire after 24 hours, so base64 is
always requested.

The chat helper in `request_minimax.py` requires `MINIMAX_BASE_URL` (the
Anthropic-compatible chat endpoint). Image generation deliberately does not:
it has its own optional `MINIMAX_IMAGE_BASE_URL`, so images work for anyone who
has only an API key.
"""

import base64
import os
from typing import Optional

import requests

try:
    from ._env import clean_env_value, dotenv_values
    from .images import (
        ImagePipelineConfig,
        ImageProviderError,
        ImageRequestRejected,
        ImageResult,
        MINIMAX_ASPECT_RATIOS,
        jpeg_dimensions,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from llm_utils._env import clean_env_value, dotenv_values
    from llm_utils.images import (
        ImagePipelineConfig,
        ImageProviderError,
        ImageRequestRejected,
        ImageResult,
        MINIMAX_ASPECT_RATIOS,
        jpeg_dimensions,
    )


DEFAULT_IMAGE_BASE_URL = "https://api.minimax.io/v1"

#: MiniMax truncates or refuses anything longer.
MAX_PROMPT_CHARS = 1500

#: `base_resp.status_code` values that describe the request rather than the
#: service: sensitive content and invalid parameters.
REJECTED_STATUS_CODES = frozenset({1026, 2013})

#: Substrings in `status_msg` that mean the same thing for codes not listed above.
REJECTED_STATUS_MARKERS = ("sensitive", "invalid")


def _get_minimax_image_credentials() -> dict:
    """Load the image API key and base URL from .env and the environment."""

    values = dotenv_values()

    api_key = clean_env_value(
        os.environ.get("MINIMAX_API_KEY") or values.get("MINIMAX_API_KEY")
    )
    base_url = clean_env_value(
        os.environ.get("MINIMAX_IMAGE_BASE_URL") or values.get("MINIMAX_IMAGE_BASE_URL")
    ) or DEFAULT_IMAGE_BASE_URL

    if not api_key:
        raise ValueError(
            "Missing MiniMax configuration in .env or environment: MINIMAX_API_KEY"
        )

    if not base_url.startswith(("http://", "https://")):
        raise ValueError("MINIMAX_IMAGE_BASE_URL must start with http:// or https://.")

    return {"api_key": api_key, "base_url": base_url.rstrip("/")}


def _response_detail(response) -> str:
    """A short explanation from an error response, whatever shape it arrived in."""

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - an error body is not always JSON
        return (getattr(response, "text", "") or "").strip()[:300]

    if isinstance(payload, dict):
        base_resp = payload.get("base_resp")
        if isinstance(base_resp, dict) and base_resp.get("status_msg"):
            return str(base_resp["status_msg"])
        for key in ("message", "error", "detail"):
            if payload.get(key):
                return str(payload[key])
    return str(payload)[:300]


def minimax_image_generation(cfg: ImagePipelineConfig, prompt: str) -> ImageResult:
    """Generate one image and return its bytes. Output is always JPEG."""

    text = prompt or ""
    if len(text) > MAX_PROMPT_CHARS:
        # Truncating silently would quietly drop the end of the teacher's
        # description; say so instead and let them shorten it.
        raise ImageRequestRejected(
            f"MiniMax accepts prompts up to {MAX_PROMPT_CHARS} characters; "
            f"this one is {len(text)}."
        )

    credentials = _get_minimax_image_credentials()
    body = {
        "model": cfg.model,
        "prompt": text,
        "aspect_ratio": MINIMAX_ASPECT_RATIOS.get(
            cfg.aspect_ratio, MINIMAX_ASPECT_RATIOS["landscape"]
        ),
        "response_format": "base64",
        "n": 1,
        "prompt_optimizer": True,
    }

    # requests.Timeout and requests.ConnectionError propagate: callers classify
    # them by name as upstream failures worth retrying.
    response = requests.post(
        f"{credentials['base_url']}/image_generation",
        headers={
            "Authorization": f"Bearer {credentials['api_key']}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=cfg.timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = getattr(response, "status_code", 0)
        detail = _response_detail(response)
        if status in (401, 403):
            raise ImageProviderError(
                f"MiniMax rejected the API key (HTTP {status}): {detail}"
            ) from exc
        if status == 429:
            raise ImageProviderError(
                f"MiniMax rate limit reached (HTTP 429): {detail}"
            ) from exc
        if 400 <= status < 500:
            raise ImageRequestRejected(
                f"MiniMax refused the request (HTTP {status}): {detail}"
            ) from exc
        raise ImageProviderError(f"MiniMax failed (HTTP {status}): {detail}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ImageProviderError("MiniMax returned a response that is not JSON.") from exc

    if not isinstance(payload, dict):
        raise ImageProviderError("MiniMax returned a response that is not an object.")

    # A failure can arrive inside an HTTP 200, so this check is not optional.
    base_resp = payload.get("base_resp") or {}
    status_code = base_resp.get("status_code", 0)
    status_msg = str(base_resp.get("status_msg") or "").strip()
    if status_code:
        lowered = status_msg.lower()
        rejected = status_code in REJECTED_STATUS_CODES or any(
            marker in lowered for marker in REJECTED_STATUS_MARKERS
        )
        if rejected:
            raise ImageRequestRejected(
                status_msg or f"MiniMax rejected the prompt (code {status_code})."
            )
        raise ImageProviderError(f"MiniMax error {status_code}: {status_msg}")

    data = payload.get("data") or {}
    encoded = (data.get("image_base64") or []) if isinstance(data, dict) else []
    if not encoded:
        raise ImageProviderError("MiniMax returned no image data.")

    raw = base64.b64decode(encoded[0])
    dimensions = jpeg_dimensions(raw)
    width: Optional[int] = dimensions[0] if dimensions else None
    height: Optional[int] = dimensions[1] if dimensions else None

    return ImageResult(
        data=raw,
        mime_type="image/jpeg",
        width=width,
        height=height,
        model=cfg.model,
        provider="minimax",
    )
