from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, cast

from src.typing.message import Message


# Dense structured output often tokenizes below the generic four characters per
# token estimate used for English prose. Use a conservative configurable default.
_CHARS_PER_TOKEN = 3.0
_MESSAGE_OVERHEAD_TOKENS = 4
_PROMPT_OVERHEAD_TOKENS = 32
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PLACEHOLDER_PREFIX = "[NSL context compaction"


@dataclass(frozen=True)
class ContextCompactionSettings:
    enabled: bool = False
    trigger_tokens: int = 52_000
    target_tokens: int = 45_000
    keep_recent_tool_outputs: int = 3
    keep_recent_assistant_reasoning: int = 3
    chars_per_token: float = 3.0


@dataclass(frozen=True)
class ContextCompactionReport:
    messages: list[Message]
    original_tokens: int
    compacted_tokens: int
    compacted_tool_messages: int = 0
    compacted_reasoning_messages: int = 0

    @property
    def compacted(self) -> bool:
        return self.compacted_tool_messages > 0 or self.compacted_reasoning_messages > 0


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _estimate_payload_tokens(
    value: Any, chars_per_token: float = _CHARS_PER_TOKEN
) -> int:
    return math.ceil(len(_json_dumps(value)) / chars_per_token)


def estimate_messages_tokens(
    messages: list[Message],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    chars_per_token: float = _CHARS_PER_TOKEN,
) -> int:
    """Cheap prompt-size estimate for pre-flight compaction.

    This intentionally estimates the serialized request payload, not just
    ``message.content``. Tool schemas and assistant ``tool_calls`` can be a
    meaningful share of prompt tokens in OpenAI-compatible tool templates.

    ``chars_per_token`` calibrates the bytes-to-tokens heuristic to the model
    and content type.
    """

    tokens = _estimate_payload_tokens(messages, chars_per_token)
    tokens += _PROMPT_OVERHEAD_TOKENS + len(messages) * _MESSAGE_OVERHEAD_TOKENS
    if tools is not None:
        tokens += _estimate_payload_tokens(tools, chars_per_token)
        tokens += len(tools) * _MESSAGE_OVERHEAD_TOKENS
    if tool_choice is not None:
        tokens += _estimate_payload_tokens(tool_choice, chars_per_token)
    return tokens


def compact_messages(
    messages: list[Message],
    *,
    settings: ContextCompactionSettings,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> list[Message]:
    return compact_messages_with_report(
        messages,
        settings=settings,
        tools=tools,
        tool_choice=tool_choice,
    ).messages


def compact_messages_with_report(
    messages: list[Message],
    *,
    settings: ContextCompactionSettings,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> ContextCompactionReport:
    original_tokens = estimate_messages_tokens(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        chars_per_token=settings.chars_per_token,
    )
    if not settings.enabled or original_tokens <= settings.trigger_tokens:
        return ContextCompactionReport(
            messages=messages,
            original_tokens=original_tokens,
            compacted_tokens=original_tokens,
        )

    out: list[Message] = list(messages)
    compacted_tool_messages = 0
    compacted_reasoning_messages = 0
    recent_tool_indexes = _recent_role_indexes(
        messages,
        role="tool",
        keep=settings.keep_recent_tool_outputs,
    )

    current_tokens = original_tokens
    for index, message in enumerate(messages):
        if current_tokens <= settings.target_tokens and compacted_tool_messages > 0:
            break
        if index in recent_tool_indexes or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        new_content = compact_tool_content(content)
        if new_content == content:
            continue
        out[index] = _with_content(message, new_content)
        compacted_tool_messages += 1
        current_tokens = estimate_messages_tokens(
            out,
            tools=tools,
            tool_choice=tool_choice,
            chars_per_token=settings.chars_per_token,
        )

    if current_tokens > settings.target_tokens:
        recent_reasoning_indexes = _recent_reasoning_indexes(
            out,
            keep=settings.keep_recent_assistant_reasoning,
        )
        for index, message in enumerate(out):
            if (
                current_tokens <= settings.target_tokens
                and compacted_reasoning_messages > 0
            ):
                break
            if index in recent_reasoning_indexes or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str) or "<think>" not in content:
                continue
            new_content = compact_assistant_reasoning(content)
            if new_content == content:
                continue
            out[index] = _with_content(message, new_content)
            compacted_reasoning_messages += 1
            current_tokens = estimate_messages_tokens(
                out,
                tools=tools,
                tool_choice=tool_choice,
                chars_per_token=settings.chars_per_token,
            )

    if compacted_tool_messages == 0 and compacted_reasoning_messages == 0:
        out = messages
        current_tokens = original_tokens

    return ContextCompactionReport(
        messages=out,
        original_tokens=original_tokens,
        compacted_tokens=current_tokens,
        compacted_tool_messages=compacted_tool_messages,
        compacted_reasoning_messages=compacted_reasoning_messages,
    )


def compact_tool_content(content: str) -> str:
    if _PLACEHOLDER_PREFIX in content:
        return content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _placeholder("tool output elided", original_chars=len(content))
    if not isinstance(parsed, dict):
        return _placeholder("tool output elided", original_chars=len(content))

    payload = dict(parsed)
    for key, value in list(payload.items()):
        if key == "output":
            payload[key] = _compact_json_value(
                value,
                max_string_chars=256,
                max_container_chars=768,
            )
        elif key in {"success", "error"}:
            payload[key] = _compact_json_value(
                value,
                max_string_chars=512,
                max_container_chars=1_024,
            )
        else:
            payload[key] = _compact_json_value(
                value,
                max_string_chars=512,
                max_container_chars=1_024,
            )
    payload["_nsl_context_compaction"] = {
        "original_chars": len(content),
        "strategy": "tool_output_elision",
    }
    compacted = _json_dumps(payload)
    if len(compacted) >= len(content):
        return content
    return compacted


def compact_assistant_reasoning(content: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        original = match.group(0)
        return (
            "<think>"
            + _placeholder("reasoning elided", original_chars=len(original))
            + "</think>"
        )

    return _THINK_BLOCK_RE.sub(_replace, content)


def _compact_json_value(
    value: Any,
    *,
    max_string_chars: int,
    max_container_chars: int,
) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string_chars or _PLACEHOLDER_PREFIX in value:
            return value
        return _placeholder("value elided", original_chars=len(value))
    if value is None or isinstance(value, bool | int | float):
        return value

    rendered = _json_dumps(value)
    if len(rendered) <= max_container_chars:
        return value
    if isinstance(value, list):
        return _placeholder(
            "list elided",
            original_chars=len(rendered),
            items=len(value),
        )
    if isinstance(value, dict):
        return {
            str(key): _compact_json_value(
                nested,
                max_string_chars=max_string_chars,
                max_container_chars=max_container_chars,
            )
            for key, nested in value.items()
        }
    return _placeholder("value elided", original_chars=len(rendered))


def _placeholder(reason: str, *, original_chars: int, items: int | None = None) -> str:
    suffix = f"; items={items}" if items is not None else ""
    return f"{_PLACEHOLDER_PREFIX}: {reason}; original_chars={original_chars}{suffix}]"


def _with_content(message: Message, content: str) -> Message:
    payload = dict(message)
    payload["content"] = content
    return cast(Message, payload)


def _recent_role_indexes(messages: list[Message], *, role: str, keep: int) -> set[int]:
    if keep <= 0:
        return set()
    out: set[int] = set()
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == role:
            out.add(index)
            if len(out) >= keep:
                break
    return out


def _recent_reasoning_indexes(messages: list[Message], *, keep: int) -> set[int]:
    if keep <= 0:
        return set()
    out: set[int] = set()
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        content = message.get("content")
        if (
            message.get("role") == "assistant"
            and isinstance(content, str)
            and "<think>" in content
        ):
            out.add(index)
            if len(out) >= keep:
                break
    return out
