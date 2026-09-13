"""Provider-neutral image generation.

Mirrors `pipelines.py` one level down: a config dataclass, a base pipeline with
one `generate_image` method, one subclass per provider, and a factory. The
provider helpers live in `request_openai_images.py` and
`request_minimax_images.py` and are imported lazily inside `generate_image` so
those modules can import this one's types without a circular import.

Two failure kinds are distinguished because callers treat them differently:
`ImageRequestRejected` is the provider saying no to *this prompt* (content
policy, too long, malformed) and must be reported to the person who typed it;
`ImageProviderError` is the provider itself failing and is worth retrying.
"""

import hashlib
import logging
import struct
import zlib
from typing import Optional, Tuple

from ._compat import Literal, dataclass
from .request_sglang import configure_logging

IMAGE_PIPELINE_TYPES = Literal["openai", "minimax", "mock"]
ASPECT_RATIOS = Literal["square", "landscape", "portrait"]

#: Default model per provider, used when nothing is configured.
DEFAULT_MODELS = {
    "openai": "gpt-image-2.5-flare",
    "minimax": "image-01",
    "mock": "mock",
}

#: `aspect_ratio` -> the `size` string the OpenAI Images API expects.
OPENAI_SIZES = {
    "square": "1024x1024",
    "landscape": "1536x1024",
    "portrait": "1024x1536",
}

#: `aspect_ratio` -> the `aspect_ratio` string MiniMax expects.
MINIMAX_ASPECT_RATIOS = {
    "square": "1:1",
    "landscape": "3:2",
    "portrait": "2:3",
}

#: Small stand-in sizes for the mock pipeline, chosen to be obviously not real.
MOCK_SIZES = {
    "square": (200, 200),
    "landscape": (256, 160),
    "portrait": (160, 256),
}

#: Prompt tokens the mock pipeline treats as instructions to fail, so tests can
#: exercise both failure paths without a provider.
MOCK_REJECT_TOKEN = "[reject]"
MOCK_FAIL_TOKEN = "[fail]"

MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


@dataclass
class ImagePipelineConfig:

    model: str
    pipeline_type: IMAGE_PIPELINE_TYPES = "openai"
    aspect_ratio: ASPECT_RATIOS = "landscape"

    #: OpenAI only: low | medium | high | xhigh | max | auto. MiniMax has no
    #: quality control and the mock pipeline ignores it.
    quality: Optional[str] = "medium"
    #: OpenAI only: png | jpeg | webp. MiniMax always returns JPEG.
    output_format: str = "png"

    timeout: int = 120
    log_level: str = "INFO"


@dataclass
class ImageResult:
    """One generated image and what is known about it.

    `width`/`height` are optional because they are read back out of the file's
    own header: a provider returning a container this module cannot parse still
    yields a usable image, just without dimensions.
    """

    data: bytes
    mime_type: str
    width: Optional[int]
    height: Optional[int]
    model: str
    provider: str
    revised_prompt: Optional[str] = None


class ImageRequestRejected(Exception):
    """The provider refused this prompt (content policy, too long, malformed).

    Not retryable: sending the same prompt again gets the same answer, so the
    caller should show the reason rather than retry.
    """


class ImageProviderError(Exception):
    """The provider failed (5xx, non-zero MiniMax base_resp, auth, empty body).

    Retryable from the client's point of view.
    """


def png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read `(width, height)` out of a PNG's IHDR chunk, or None if it is not one."""

    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Read `(width, height)` from a JPEG's first start-of-frame marker."""

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None

    index = 2
    end = len(data)
    while index + 9 < end:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        # Standalone markers carry no length field.
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        segment_length = struct.unpack(">H", data[index + 2:index + 4])[0]
        # SOF0..SOF15, excluding the four that are not frame headers.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return int(width), int(height)
        index += 2 + segment_length
    return None


def image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Dimensions of a PNG or JPEG, whichever it is, or None."""

    return png_dimensions(data) or jpeg_dimensions(data)


def mime_type_for(output_format: str) -> str:
    """Map an `output_format` to its media type, defaulting to PNG."""

    return MIME_TYPES.get((output_format or "png").lower().lstrip("."), "image/png")


def solid_png(width: int, height: int, colour: Tuple[int, int, int]) -> bytes:
    """Build a single-colour truecolour PNG with the standard library only.

    Pillow is deliberately not a dependency of this package, and a solid
    rectangle needs nothing more than `zlib` and `struct`: one IHDR, one
    zlib-compressed IDAT of filter-0 scanlines, one IEND.
    """

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes(colour) * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * height, 9))
        + chunk(b"IEND", b"")
    )


class ImagePipeline:
    """Base image pipeline. Subclasses implement `generate_image`."""

    def __init__(self, cfg: ImagePipelineConfig):
        self.cfg = cfg
        self.model_name = cfg.model
        self.provider = cfg.pipeline_type
        configure_logging(cfg.log_level)

    def generate_image(self, prompt: str) -> ImageResult:
        raise NotImplementedError(
            "It seems that you have accidentally used the base ImagePipeline class, "
            "which does not have a `generate_image` implementation."
        )

    def _log_request(self, prompt: str) -> None:
        # The prompt is the teacher's own words; keep it out of INFO logs.
        logging.debug("Image request to %s/%s: %s", self.provider, self.model_name, prompt)

    def _log_result(self, result: ImageResult) -> ImageResult:
        logging.info(
            "Image generated by %s/%s: %s, %sx%s, %d bytes",
            result.provider,
            result.model,
            result.mime_type,
            result.width,
            result.height,
            len(result.data),
        )
        return result


class OpenAIImagePipeline(ImagePipeline):

    def generate_image(self, prompt: str) -> ImageResult:
        # Imported here, not at module scope: the helper imports this module's
        # types, and the OpenAI SDK should stay optional for mock-only users.
        from .request_openai_images import openai_image_generation

        self._log_request(prompt)
        return self._log_result(openai_image_generation(self.cfg, prompt))


class MinimaxImagePipeline(ImagePipeline):

    def generate_image(self, prompt: str) -> ImageResult:
        from .request_minimax_images import minimax_image_generation

        self._log_request(prompt)
        return self._log_result(minimax_image_generation(self.cfg, prompt))


class MockImagePipeline(ImagePipeline):
    """Deterministic offline pipeline: a solid rectangle keyed to the prompt.

    The colour comes from the prompt's SHA-1, so the same prompt always yields
    the same picture and two different prompts almost always differ -- enough to
    see in a console that a new image really was generated. A prompt containing
    `[reject]` or `[fail]` raises, which is how the failure paths are tested
    without a provider.
    """

    def generate_image(self, prompt: str) -> ImageResult:
        self._log_request(prompt)

        text = prompt or ""
        if MOCK_REJECT_TOKEN in text:
            raise ImageRequestRejected(
                "The mock image pipeline rejects any prompt containing "
                f"{MOCK_REJECT_TOKEN!r}."
            )
        if MOCK_FAIL_TOKEN in text:
            raise ImageProviderError(
                "The mock image pipeline fails for any prompt containing "
                f"{MOCK_FAIL_TOKEN!r}."
            )

        width, height = MOCK_SIZES.get(self.cfg.aspect_ratio, MOCK_SIZES["landscape"])
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        colour = (digest[0], digest[1], digest[2])

        return self._log_result(
            ImageResult(
                data=solid_png(width, height, colour),
                mime_type="image/png",
                width=width,
                height=height,
                model="mock",
                provider="mock",
            )
        )


def image_pipeline_from_config(cfg: ImagePipelineConfig) -> ImagePipeline:

    if cfg.pipeline_type == "openai":
        image_pipeline = OpenAIImagePipeline(cfg)
    elif cfg.pipeline_type == "minimax":
        image_pipeline = MinimaxImagePipeline(cfg)
    elif cfg.pipeline_type == "mock":
        image_pipeline = MockImagePipeline(cfg)
    else:
        raise Exception(f"Incorrect image pipeline type {cfg.pipeline_type}")
    return image_pipeline
