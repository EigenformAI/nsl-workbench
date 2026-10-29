from pprint import pformat
from typing import Any, List, Protocol, cast

import anthropic
from loguru import logger
from result import Result, Ok, Err

from src.genner.Base import CONTEXT_OVERFLOW_PREFIX
from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message
from .Base import Genner
from dataclasses import dataclass, field

from typing import NamedTuple


class ClaudeUsageResponse(Protocol):
    input_tokens: int | None
    output_tokens: int | None


class ClaudeMessageResponse(Protocol):
    usage: ClaudeUsageResponse
    model: str
    stop_reason: str | None


@dataclass
class PList:
    messages: List[Message] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.messages)

    def __add__(self, other: "PList") -> "PList":
        return PList(messages=self.messages + other.messages)

    def __repr__(self) -> str:
        messages_repr = pformat(self.messages)
        return f"PList(\n\tmessages=[\n\t\t{messages_repr}\n\t\t]\n)"


class ClaudeConfig(NamedTuple):
    name: str = "Claude"
    model: str = "claude-3-5-sonnet-20241022"  # Latest Claude model
    max_tokens: int = 1000
    temperature: float = 0.5


class ClaudeGenner(Genner):
    def __init__(self, client: anthropic.Anthropic, config: ClaudeConfig):
        super().__init__("claude")

        self.client = client
        self.config = config

    def plist_completion(
        self,
        messages: List[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Result[InferenceResult, str]:
        try:
            # Convert messages format for Claude
            claude_messages = []
            system_message = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg.get("content") or ""
                else:
                    claude_messages.append(
                        {"role": msg["role"], "content": msg.get("content") or ""}
                    )

            # Create Claude API call
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=(
                    max_tokens if max_tokens is not None else self.config.max_tokens
                ),
                temperature=(
                    temperature if temperature is not None else self.config.temperature
                ),
                system=system_message
                if system_message
                else "You are a helpful assistant.",
                messages=claude_messages,
            )

            return Ok(
                InferenceResult(
                    content=response.content[0].text,
                    usage=self.get_usage_info(response),
                )
            )

        except anthropic.BadRequestError as e:
            if self.is_context_overflow(e):
                msg = str(getattr(e, "message", e))
                logger.warning(
                    "ClaudeGenner.plist_completion: context overflow "
                    f"(model={self.config.model}, max_tokens={self.config.max_tokens}): {msg}"
                )
                return Err(f"{CONTEXT_OVERFLOW_PREFIX} {msg}")
            logger.exception(
                f"ClaudeGenner.plist_completion failed for model {self.config.model}"
            )
            return Err(
                "ClaudeGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )
        except Exception as e:
            logger.exception(
                f"ClaudeGenner.plist_completion failed for model {self.config.model}"
            )
            return Err(
                "ClaudeGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )

    @staticmethod
    def is_context_overflow(exc: anthropic.BadRequestError) -> bool:
        body = getattr(exc, "body", None)
        raw_error = body.get("error") if isinstance(body, dict) else None
        error_payload = raw_error if isinstance(raw_error, dict) else {}
        error_type = error_payload.get("type") or getattr(exc, "type", None)
        message_parts = [
            str(getattr(exc, "message", "") or ""),
            str(error_payload.get("message", "") or ""),
            str(error_type or ""),
        ]
        message = " ".join(part for part in message_parts if part).lower()
        return "prompt is too long" in message or (
            error_type == "invalid_request_error" and "max_tokens" in message
        )

    @staticmethod
    def get_usage_info(response: object) -> UsageInfo:
        response = cast(ClaudeMessageResponse, response)
        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
        )
