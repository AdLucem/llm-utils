"""Generate one image with the OpenAI Images API.

`client.images.generate` always answers with base64 for the `gpt-image-*`
models -- there is no `response_format` for them -- so the only compatibility
branch here is for the older `dall-e-*` models, which do need it and reject
`output_format`/`quality`.
"""

import base64
import logging
from typing import Optional

try:
    from .images import (
        ImagePipelineConfig,
        ImageProviderError,
        ImageRequestRejected,
        ImageResult,
        OPENAI_SIZES,
        image_dimensions,
        mime_type_for,
    )
    from .request_openai import _get_openai_credentials
except ImportError:  # pragma: no cover - direct script fallback
    from llm_utils.images import (
        ImagePipelineConfig,
        ImageProviderError,
        ImageRequestRejected,
        ImageResult,
        OPENAI_SIZES,
        image_dimensions,
        mime_type_for,
    )
    from llm_utils.request_openai import _get_openai_credentials


#: Qualities the API accepts. `auto` is sent as-is; anything else is dropped
#: with a warning rather than failing the request over a typo in a config file.
QUALITIES = ("low", "medium", "high", "xhigh", "max", "auto")

MAX_ERROR_CHARS = 300


def _openai_module():
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "openai is required to use the OpenAI image helpers. Install `openai`."
        ) from exc
    return openai


def _client(cfg: ImagePipelineConfig):
    openai = _openai_module()
    credentials = _get_openai_credentials()
    return openai.OpenAI(
        api_key=credentials["api_key"],
        base_url=credentials["base_url"],
        timeout=cfg.timeout,
    )


def _short(exc: BaseException) -> str:
    """The provider's own explanation, without the SDK's wrapper noise."""

    message = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
        message = message or body.get("message")

    text = str(message or exc).strip()
    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS - 1].rstrip() + "…"
    return text


def _generation_kwargs(cfg: ImagePipelineConfig, prompt: str) -> dict:
    """Build the request body, omitting parameters the model would reject."""

    size = OPENAI_SIZES.get(cfg.aspect_ratio, OPENAI_SIZES["landscape"])
    kwargs = {"model": cfg.model, "prompt": prompt, "n": 1, "size": size}

    if cfg.model.lower().startswith("dall-e"):
        # The legacy models answer with URLs unless asked for base64, and know
        # neither `output_format` nor the gpt-image quality names.
        kwargs["response_format"] = "b64_json"
        return kwargs

    kwargs["output_format"] = (cfg.output_format or "png").lower()
    quality = (cfg.quality or "").lower()
    if quality in QUALITIES:
        kwargs["quality"] = quality
    elif quality:
        logging.warning("Ignoring unknown image quality %r", cfg.quality)
    return kwargs


def _requested_dimensions(size: str):
    try:
        width, height = size.lower().split("x", 1)
        return int(width), int(height)
    except (AttributeError, ValueError):
        return None, None


def openai_image_generation(cfg: ImagePipelineConfig, prompt: str) -> ImageResult:
    """Generate one image and return it as bytes, never as a URL."""

    openai = _openai_module()
    kwargs = _generation_kwargs(cfg, prompt)

    try:
        response = _client(cfg).images.generate(**kwargs)
    except openai.BadRequestError as exc:
        # A 400 is the model refusing this prompt: content policy, an over-long
        # prompt, an unsupported parameter. Retrying is pointless, so it must not
        # reach the caller as one of the retryable provider errors below.
        raise ImageRequestRejected(_short(exc)) from exc
    # AuthenticationError, PermissionDeniedError (an org not yet enabled for the
    # model), RateLimitError, APIConnectionError, APITimeoutError and
    # InternalServerError all propagate: callers already classify those by name
    # as upstream failures worth retrying.

    data = getattr(response, "data", None) or []
    if not data:
        raise ImageProviderError(f"OpenAI returned no image for model {cfg.model!r}.")

    first = data[0]
    encoded: Optional[str] = getattr(first, "b64_json", None)
    if not encoded:
        raise ImageProviderError(
            f"OpenAI returned an image without base64 data for model {cfg.model!r}."
        )

    raw = base64.b64decode(encoded)
    output_format = kwargs.get("output_format", "png")
    width, height = image_dimensions(raw) or _requested_dimensions(kwargs["size"])

    return ImageResult(
        data=raw,
        mime_type=mime_type_for(output_format),
        width=width,
        height=height,
        model=cfg.model,
        provider="openai",
        revised_prompt=getattr(first, "revised_prompt", None),
    )
