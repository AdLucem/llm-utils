"""MiniMax backend failures become `MinimaxBackendError` with a `retryable` flag.

No network: the Anthropic SDK exceptions are built directly around stub
`httpx` requests and responses.
"""

import sys
import unittest
from pathlib import Path

import anthropic
import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_utils import MinimaxBackendError  # noqa: E402
from llm_utils.request_minimax import _raise_as_backend_error  # noqa: E402

REQUEST = httpx.Request("POST", "https://api.minimax.io/anthropic/v1/messages")


def status_error(cls, status):
    return cls("raw sdk message", response=httpx.Response(status, request=REQUEST), body=None)


def translate(exc):
    """Run `exc` through `_raise_as_backend_error` the way the helpers do."""

    try:
        raise exc
    except anthropic.AnthropicError as caught:
        _raise_as_backend_error(caught)


class MinimaxBackendErrorTests(unittest.TestCase):
    def assert_backend_error(self, exc, *, retryable, status_code):
        with self.assertRaises(MinimaxBackendError) as ctx:
            translate(exc)
        self.assertIs(ctx.exception.retryable, retryable)
        self.assertEqual(ctx.exception.status_code, status_code)
        self.assertIs(ctx.exception.__cause__, exc)
        return ctx.exception

    def test_rate_limit_is_retryable(self):
        err = self.assert_backend_error(
            status_error(anthropic.RateLimitError, 429), retryable=True, status_code=429
        )
        self.assertIn("rate-limiting", str(err))
        self.assertNotIn("raw sdk message", str(err))

    def test_server_error_is_retryable(self):
        err = self.assert_backend_error(
            status_error(anthropic.InternalServerError, 503), retryable=True, status_code=503
        )
        self.assertIn("HTTP 503", str(err))

    @unittest.skipUnless(hasattr(anthropic, "OverloadedError"), "SDK has no OverloadedError")
    def test_overloaded_is_retryable(self):
        err = self.assert_backend_error(
            status_error(anthropic.OverloadedError, 529), retryable=True, status_code=529
        )
        self.assertIn("overloaded", str(err))

    def test_bad_request_is_not_retryable(self):
        err = self.assert_backend_error(
            status_error(anthropic.BadRequestError, 400), retryable=False, status_code=400
        )
        self.assertIn("HTTP 400", str(err))

    def test_connection_error_is_retryable(self):
        self.assert_backend_error(
            anthropic.APIConnectionError(request=REQUEST), retryable=True, status_code=None
        )

    def test_other_sdk_errors_propagate_unchanged(self):
        original = anthropic.AnthropicError("not an HTTP failure")
        with self.assertRaises(anthropic.AnthropicError) as ctx:
            translate(original)
        self.assertIs(ctx.exception, original)
        self.assertNotIsInstance(ctx.exception, MinimaxBackendError)


if __name__ == "__main__":
    unittest.main()
