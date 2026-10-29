from abc import ABC, abstractmethod
from typing import Any, List

from result import Result

from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message


CONTEXT_OVERFLOW_PREFIX = "context_overflow:"
INFERENCE_UNAVAILABLE_PREFIX = "inference_unavailable:"
# A request-level timeout (the client gave up waiting, e.g. decode starvation
# under load) is distinct from the endpoint being unreachable. It is a
# RETRYABLE episode failure, not an endpoint outage: it must NOT quarantine the
# endpoint (which, with a single endpoint, would breach the capacity floor and
# abort the whole run). It is still treated as "the backend's fault, not the
# agent's" for episode categorisation (see openai_shim / container harness).
INFERENCE_TIMEOUT_PREFIX = "inference_timeout:"


class InferenceTimeoutError(RuntimeError):
    """Raised when an inference request times out without an endpoint outage."""


class Genner(ABC):
    client: Any = None
    collector: Any = None

    def __init__(self, identifier: str):
        self.identifier = identifier
        self.client = None
        self.collector = None

    @abstractmethod
    def plist_completion(
        self,
        messages: List[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Result[InferenceResult, str]:
        pass

    @staticmethod
    @abstractmethod
    def get_usage_info(response: object) -> UsageInfo:
        pass
