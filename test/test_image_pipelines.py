"""Tests for the image pipelines. No network, no credentials, no Pillow."""

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_utils import _env  # noqa: E402
from llm_utils import request_minimax_images, request_openai_images  # noqa: E402
from llm_utils.images import (  # noqa: E402
    ImagePipelineConfig,
    ImageProviderError,
    ImageRequestRejected,
    MockImagePipeline,
    image_pipeline_from_config,
    jpeg_dimensions,
    png_dimensions,
    solid_png,
)


MOCK_PNG = solid_png(8, 6, (10, 20, 30))
MOCK_PNG_B64 = base64.b64encode(MOCK_PNG).decode("ascii")

#: The smallest thing `jpeg_dimensions` should be able to read: SOI, then a
#: baseline SOF0 declaring 40x24, then EOI.
TINY_JPEG = (
    b"\xff\xd8"
    + b"\xff\xc0\x00\x11\x08\x00\x18\x00\x28\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    + b"\xff\xd9"
)
TINY_JPEG_B64 = base64.b64encode(TINY_JPEG).decode("ascii")


def mock_config(**overrides) -> ImagePipelineConfig:
    fields = {"model": "mock", "pipeline_type": "mock"}
    fields.update(overrides)
    return ImagePipelineConfig(**fields)


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------

def test_png_dimensions_reads_the_header():
    assert png_dimensions(solid_png(37, 11, (0, 0, 0))) == (37, 11)


def test_png_dimensions_rejects_non_png():
    assert png_dimensions(b"not an image at all") is None
    assert png_dimensions(TINY_JPEG) is None


def test_jpeg_dimensions_reads_the_first_frame_header():
    assert jpeg_dimensions(TINY_JPEG) == (40, 24)


def test_jpeg_dimensions_rejects_non_jpeg():
    assert jpeg_dimensions(MOCK_PNG) is None


# --------------------------------------------------------------------------
# Mock pipeline
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "aspect,expected",
    [("landscape", (256, 160)), ("portrait", (160, 256)), ("square", (200, 200))],
)
def test_mock_pipeline_sizes_follow_the_aspect_ratio(aspect, expected):
    pipeline = image_pipeline_from_config(mock_config(aspect_ratio=aspect))
    result = pipeline.generate_image("a volcano")

    assert isinstance(pipeline, MockImagePipeline)
    assert result.data.startswith(b"\x89PNG")
    assert png_dimensions(result.data) == expected
    assert (result.width, result.height) == expected
    assert result.mime_type == "image/png"
    assert (result.model, result.provider) == ("mock", "mock")


def test_mock_pipeline_is_deterministic_and_prompt_dependent():
    pipeline = image_pipeline_from_config(mock_config())
    assert pipeline.generate_image("a volcano").data == pipeline.generate_image("a volcano").data
    assert pipeline.generate_image("a volcano").data != pipeline.generate_image("a river").data


def test_mock_pipeline_reject_token_raises_rejected():
    pipeline = image_pipeline_from_config(mock_config())
    with pytest.raises(ImageRequestRejected):
        pipeline.generate_image("draw [reject] this")


def test_mock_pipeline_fail_token_raises_provider_error():
    pipeline = image_pipeline_from_config(mock_config())
    with pytest.raises(ImageProviderError):
        pipeline.generate_image("draw [fail] this")


def test_unknown_pipeline_type_is_rejected():
    with pytest.raises(Exception):
        image_pipeline_from_config(mock_config(pipeline_type="crayon"))


# --------------------------------------------------------------------------
# OpenAI helper
# --------------------------------------------------------------------------

class StubImages:
    def __init__(self, response=None, error=None):
        self.kwargs = None
        self._response = response
        self._error = error

    def generate(self, **kwargs):
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class StubOpenAIClient:
    def __init__(self, images):
        self.images = images


def install_openai_stub(monkeypatch, images):
    monkeypatch.setattr(request_openai_images, "_client", lambda cfg: StubOpenAIClient(images))
    return images


def openai_response(b64=MOCK_PNG_B64, revised_prompt=None):
    return SimpleNamespace(data=[SimpleNamespace(b64_json=b64, revised_prompt=revised_prompt)])


def openai_config(**overrides) -> ImagePipelineConfig:
    fields = {"model": "gpt-image-2.5-flare", "pipeline_type": "openai"}
    fields.update(overrides)
    return ImagePipelineConfig(**fields)


@pytest.mark.parametrize(
    "aspect,size",
    [("square", "1024x1024"), ("landscape", "1536x1024"), ("portrait", "1024x1536")],
)
def test_openai_maps_the_aspect_ratio_to_a_size(monkeypatch, aspect, size):
    images = install_openai_stub(monkeypatch, StubImages(openai_response()))
    request_openai_images.openai_image_generation(openai_config(aspect_ratio=aspect), "a volcano")
    assert images.kwargs["size"] == size


def test_openai_sends_one_image_and_no_response_format(monkeypatch):
    images = install_openai_stub(monkeypatch, StubImages(openai_response()))
    result = request_openai_images.openai_image_generation(
        openai_config(output_format="webp", quality="high"), "a volcano"
    )

    assert images.kwargs["n"] == 1
    assert images.kwargs["model"] == "gpt-image-2.5-flare"
    assert images.kwargs["prompt"] == "a volcano"
    assert images.kwargs["output_format"] == "webp"
    assert images.kwargs["quality"] == "high"
    assert "response_format" not in images.kwargs
    assert result.mime_type == "image/webp"
    assert result.provider == "openai"
    assert result.data == MOCK_PNG


def test_openai_unknown_quality_is_dropped(monkeypatch):
    images = install_openai_stub(monkeypatch, StubImages(openai_response()))
    request_openai_images.openai_image_generation(openai_config(quality="ultra"), "a volcano")
    assert "quality" not in images.kwargs


def test_openai_dall_e_gets_the_legacy_parameters(monkeypatch):
    images = install_openai_stub(monkeypatch, StubImages(openai_response()))
    request_openai_images.openai_image_generation(openai_config(model="dall-e-3"), "a volcano")

    assert images.kwargs["response_format"] == "b64_json"
    assert "output_format" not in images.kwargs
    assert "quality" not in images.kwargs


def test_openai_reads_dimensions_and_revised_prompt(monkeypatch):
    install_openai_stub(
        monkeypatch, StubImages(openai_response(revised_prompt="a very detailed volcano"))
    )
    result = request_openai_images.openai_image_generation(openai_config(), "a volcano")

    assert (result.width, result.height) == (8, 6)
    assert result.revised_prompt == "a very detailed volcano"


def test_openai_falls_back_to_the_requested_size(monkeypatch):
    install_openai_stub(
        monkeypatch, StubImages(openai_response(b64=base64.b64encode(b"garbage").decode()))
    )
    result = request_openai_images.openai_image_generation(openai_config(), "a volcano")
    assert (result.width, result.height) == (1536, 1024)


def test_openai_bad_request_becomes_a_rejection(monkeypatch):
    import httpx
    import openai

    error = openai.BadRequestError(
        "rejected",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com")),
        body={"error": {"message": "Your prompt was rejected by the safety system."}},
    )
    install_openai_stub(monkeypatch, StubImages(error=error))

    with pytest.raises(ImageRequestRejected) as caught:
        request_openai_images.openai_image_generation(openai_config(), "something disallowed")
    assert "safety system" in str(caught.value)


def test_openai_permission_denied_propagates(monkeypatch):
    import httpx
    import openai

    error = openai.PermissionDeniedError(
        "forbidden",
        response=httpx.Response(403, request=httpx.Request("POST", "https://api.openai.com")),
        body=None,
    )
    install_openai_stub(monkeypatch, StubImages(error=error))

    # An org that is not enabled for the model is an upstream problem the API
    # layer classifies by name, not a rejection of this prompt.
    with pytest.raises(openai.PermissionDeniedError):
        request_openai_images.openai_image_generation(openai_config(), "a volcano")


def test_openai_empty_response_is_a_provider_error(monkeypatch):
    install_openai_stub(monkeypatch, StubImages(SimpleNamespace(data=[])))
    with pytest.raises(ImageProviderError):
        request_openai_images.openai_image_generation(openai_config(), "a volcano")


# --------------------------------------------------------------------------
# MiniMax helper
# --------------------------------------------------------------------------

class StubResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def install_minimax_stub(monkeypatch, response, *, key="test-key"):
    monkeypatch.setenv("MINIMAX_API_KEY", key)
    monkeypatch.delenv("MINIMAX_IMAGE_BASE_URL", raising=False)
    calls = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.update(url=url, headers=headers, body=json, timeout=timeout)
        return response

    monkeypatch.setattr(request_minimax_images.requests, "post", fake_post)
    return calls


def minimax_config(**overrides) -> ImagePipelineConfig:
    fields = {"model": "image-01", "pipeline_type": "minimax"}
    fields.update(overrides)
    return ImagePipelineConfig(**fields)


def minimax_payload(images=(TINY_JPEG_B64,), status_code=0, status_msg="success"):
    return {
        "id": "img-1",
        "data": {"image_base64": list(images)},
        "metadata": {"success_count": len(images), "failed_count": 0},
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


def test_minimax_success_returns_a_jpeg(monkeypatch):
    calls = install_minimax_stub(monkeypatch, StubResponse(payload=minimax_payload()))
    result = request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")

    assert calls["url"] == "https://api.minimax.io/v1/image_generation"
    assert calls["headers"]["Authorization"] == "Bearer test-key"
    assert calls["body"]["response_format"] == "base64"
    assert calls["body"]["aspect_ratio"] == "3:2"
    assert calls["body"]["n"] == 1
    assert result.mime_type == "image/jpeg"
    assert result.provider == "minimax"
    assert (result.width, result.height) == (40, 24)


def test_minimax_base_url_can_be_overridden(monkeypatch):
    calls = install_minimax_stub(monkeypatch, StubResponse(payload=minimax_payload()))
    monkeypatch.setenv("MINIMAX_IMAGE_BASE_URL", "https://api.minimaxi.com/v1/")
    request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")
    assert calls["url"] == "https://api.minimaxi.com/v1/image_generation"


def test_minimax_does_not_require_the_chat_base_url(monkeypatch):
    install_minimax_stub(monkeypatch, StubResponse(payload=minimax_payload()))
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    # Would raise if image generation shared the chat helper's requirements.
    request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")


def test_minimax_non_zero_base_resp_is_a_provider_error(monkeypatch):
    install_minimax_stub(
        monkeypatch,
        StubResponse(payload=minimax_payload(status_code=1002, status_msg="rate limited")),
    )
    with pytest.raises(ImageProviderError) as caught:
        request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")
    assert "1002" in str(caught.value)


def test_minimax_sensitive_content_is_a_rejection(monkeypatch):
    install_minimax_stub(
        monkeypatch,
        StubResponse(payload=minimax_payload(status_code=1026, status_msg="sensitive content")),
    )
    with pytest.raises(ImageRequestRejected):
        request_minimax_images.minimax_image_generation(minimax_config(), "something disallowed")


def test_minimax_http_401_is_a_provider_error(monkeypatch):
    install_minimax_stub(monkeypatch, StubResponse(status_code=401, payload={"message": "bad key"}))
    with pytest.raises(ImageProviderError) as caught:
        request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")
    assert "API key" in str(caught.value)


def test_minimax_http_400_is_a_rejection(monkeypatch):
    install_minimax_stub(monkeypatch, StubResponse(status_code=400, payload={"message": "nope"}))
    with pytest.raises(ImageRequestRejected):
        request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")


def test_minimax_http_500_is_a_provider_error(monkeypatch):
    install_minimax_stub(monkeypatch, StubResponse(status_code=500, text="boom"))
    with pytest.raises(ImageProviderError):
        request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")


def test_minimax_empty_data_is_a_provider_error(monkeypatch):
    install_minimax_stub(monkeypatch, StubResponse(payload=minimax_payload(images=())))
    with pytest.raises(ImageProviderError):
        request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")


def test_minimax_over_long_prompt_is_rejected_before_any_call(monkeypatch):
    called = []
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(
        request_minimax_images.requests, "post", lambda *a, **k: called.append(a) or StubResponse()
    )

    with pytest.raises(ImageRequestRejected) as caught:
        request_minimax_images.minimax_image_generation(minimax_config(), "x" * 1501)

    assert "1500" in str(caught.value)
    assert called == []


def test_minimax_missing_key_raises_value_error(monkeypatch, tmp_path):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    # Somewhere with no .env above it, so a developer's own key cannot be found.
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as caught:
        request_minimax_images.minimax_image_generation(minimax_config(), "a volcano")
    assert "MINIMAX_API_KEY" in str(caught.value)


# --------------------------------------------------------------------------
# Shared dotenv loader
# --------------------------------------------------------------------------

def test_dotenv_values_finds_a_key_in_a_parent_directory(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text('EXAMPLE_KEY="from-the-parent"\n', encoding="utf-8")
    nested = tmp_path / "centaurus" / "src"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert _env.dotenv_values()["EXAMPLE_KEY"] == "from-the-parent"


def test_dotenv_values_prefers_the_nearest_file(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("EXAMPLE_KEY=outer\n", encoding="utf-8")
    nested = tmp_path / "inner"
    nested.mkdir()
    (nested / ".env").write_text("EXAMPLE_KEY=inner\n", encoding="utf-8")
    monkeypatch.chdir(nested)

    assert _env.dotenv_values()["EXAMPLE_KEY"] == "inner"


def test_read_dotenv_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\n\nA=1\nnot a pair\nB='two'\n", encoding="utf-8")
    assert _env.read_dotenv(path) == {"A": "1", "B": "two"}
