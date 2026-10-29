import unittest
from unittest.mock import MagicMock, patch

import anthropic
import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError

from src.genner.Base import (
    CONTEXT_OVERFLOW_PREFIX,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.genner.Claude import ClaudeConfig, ClaudeGenner
from src.genner.OAI import OAIConfig, OAIGenner


class GennerContextOverflowTests(unittest.TestCase):
    def test_oai_context_overflow_returns_classified_err(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = BadRequestError(
            "400 - max context 24000; requested 4096 output + 19905 input = 24001",
            response=httpx.Response(
                400,
                request=httpx.Request(
                    "POST", "https://example.com/v1/chat/completions"
                ),
            ),
            body={
                "error": {
                    "message": "maximum context length is 24000 tokens",
                    "code": "context_length_exceeded",
                    "param": "input_tokens",
                }
            },
        )
        genner = OAIGenner(
            client,
            OAIConfig(model="demo-model", max_tokens=4096, temperature=0.0),
        )

        with (
            patch("src.genner.OAI.logger.warning") as logger_warning,
            patch("src.genner.OAI.logger.exception") as logger_exception,
        ):
            result = genner.plist_completion(
                [{"role": "user", "content": "hello", "meta": {}}]
            )

        self.assertTrue(result.is_err())
        self.assertTrue(result.unwrap_err().startswith(CONTEXT_OVERFLOW_PREFIX))
        logger_warning.assert_called_once()
        logger_exception.assert_not_called()

    def test_claude_context_overflow_returns_classified_err(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = anthropic.BadRequestError(
            "prompt is too long for this model",
            response=httpx.Response(
                400,
                request=httpx.Request("POST", "https://example.com/v1/messages"),
            ),
            body={
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: max_tokens exceeded",
                }
            },
        )
        genner = ClaudeGenner(
            client,
            ClaudeConfig(model="claude-demo", max_tokens=4096, temperature=0.0),
        )

        with (
            patch("src.genner.Claude.logger.warning") as logger_warning,
            patch("src.genner.Claude.logger.exception") as logger_exception,
        ):
            result = genner.plist_completion(
                [{"role": "user", "content": "hello", "meta": {}}]
            )

        self.assertTrue(result.is_err())
        self.assertTrue(result.unwrap_err().startswith(CONTEXT_OVERFLOW_PREFIX))
        logger_warning.assert_called_once()
        logger_exception.assert_not_called()


class GennerInferenceUnavailableTests(unittest.TestCase):
    def test_oai_connection_error_returns_classified_err(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = APIConnectionError(
            request=httpx.Request(
                "POST", "http://192.168.10.34:8000/v1/chat/completions"
            ),
        )
        genner = OAIGenner(
            client,
            OAIConfig(model="demo-model", max_tokens=512, temperature=0.0),
        )

        result = genner.plist_completion(
            [{"role": "user", "content": "hello", "meta": {}}]
        )

        self.assertTrue(result.is_err())
        self.assertTrue(
            result.unwrap_err().startswith(INFERENCE_UNAVAILABLE_PREFIX)
        )

    def test_oai_timeout_error_is_classified_as_retryable_timeout(self) -> None:
        # A request-level timeout (client gave up waiting, e.g. decode
        # starvation under load) is a RETRYABLE episode failure, not an
        # endpoint outage. It must carry the distinct timeout prefix and NOT
        # the inference_unavailable prefix, so the endpoint pool does not
        # quarantine the (possibly sole) endpoint and abort the run.
        client = MagicMock()
        client.chat.completions.create.side_effect = APITimeoutError(
            request=httpx.Request(
                "POST", "http://192.168.10.34:8000/v1/chat/completions"
            ),
        )
        genner = OAIGenner(
            client,
            OAIConfig(model="demo-model", max_tokens=512, temperature=0.0),
        )

        result = genner.plist_completion(
            [{"role": "user", "content": "hello", "meta": {}}]
        )

        self.assertTrue(result.is_err())
        err = result.unwrap_err()
        self.assertTrue(err.startswith(INFERENCE_TIMEOUT_PREFIX))
        self.assertFalse(err.startswith(INFERENCE_UNAVAILABLE_PREFIX))


class ClaudeIsContextOverflowTests(unittest.TestCase):
    def _make_bad_request(self, body) -> anthropic.BadRequestError:
        return anthropic.BadRequestError(
            "some message",
            response=httpx.Response(
                400,
                request=httpx.Request("POST", "https://example.com/v1/messages"),
            ),
            body=body,
        )

    def test_non_dict_error_payload_none(self) -> None:
        exc = self._make_bad_request({"error": None})
        self.assertFalse(ClaudeGenner.is_context_overflow(exc))

    def test_non_dict_error_payload_string(self) -> None:
        exc = self._make_bad_request({"error": "oops something went wrong"})
        self.assertFalse(ClaudeGenner.is_context_overflow(exc))
