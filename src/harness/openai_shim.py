"""OpenAI-compatible inference shim for external-container harnesses.

Each per-episode FastAPI app exposes ``/v1/chat/completions``, delegates to
``TracedGenner``, and records requests with resolved phase tags.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from result import Err, Ok

from src.genner.Base import (
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.harness.context_compaction import (
    ContextCompactionReport,
    ContextCompactionSettings,
    compact_messages_with_report,
)
from src.harness.recorder import EventRecorder
from src.harness.tool_contract import ToolResponseClassification, allowed_tool_names
from src.harness.traced_genner import TracedGenner
from src.typing.message import Message


_PSEUDO_TOOL_CALL_RE = re.compile(
    r"<(?:tool_)?call>\s*(?P<body>.*?)\s*</(?:tool_)?call>",
    re.DOTALL,
)
_PSEUDO_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[^>\s]+)>\s*(?P<body>.*?)\s*</function>",
    re.DOTALL,
)
_PSEUDO_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)
_NAME_TAG_RE = re.compile(r"<name>\s*(?P<name>.*?)\s*</name>", re.DOTALL)
_ARGUMENTS_TAG_RE = re.compile(
    r"<arguments>\s*(?P<body>.*?)\s*</arguments>",
    re.DOTALL,
)
_ARG_CHILD_RE = re.compile(
    r"<(?P<key>[^>\s]+)>\s*(?P<value>.*?)\s*</(?P=key)>",
    re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"</?think>\s*", re.DOTALL)
_OUTER_WRAPPER_RE = re.compile(r"</?tool_calls>\s*", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:\w+\n?)?", re.DOTALL)
# Fenced block used as a tool-call container.
_CODE_FENCE_BLOCK_RE = re.compile(
    r"```(?:[\w\-]+)?\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)
# Dev note: SGLang/Qwen2.5 can copy the tool-list wrapper from chat-template
# context instead of emitting a normal <tool_call> block.
_TOOLS_WRAPPER_RE = re.compile(
    r"<tools>\s*(?P<body>.*?)\s*</tools>",
    re.DOTALL,
)
# Matches <function name="..." arguments='...'/> self-closing XML emission.
# Non-greedy capture up to the closing /> so attribute values containing /
# (e.g. paths like /tmp) or > (rare) don't truncate the match.
_FUNCTION_ATTR_RE = re.compile(
    r"<function\s+(?P<attrs>.*?)\s*/>",
    re.DOTALL,
)
# Direct tool-name XML tag, e.g. <nsl---run_python>body</nsl---run_python>.
# Requiring --- avoids false positives on regular HTML/XML.
_DIRECT_TOOL_TAG_RE = re.compile(
    r"<(?P<name>[^\s>]*---[^\s>]*)>\s*(?P<body>.*?)\s*</(?P=name)>",
    re.DOTALL,
)
# Python triple-quoted strings inside JSON-like bodies.
_TRIPLE_QUOTE_RE = re.compile(r'"""\s*(?P<inner>.*?)\s*"""', re.DOTALL)

GRACEFUL_AGENT_FAILURE_PREFIX = "graceful_agent_failure:"


def _coerce_parameter_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _fix_triple_quoted_json(body: str) -> str:
    """Replace Python triple-quoted JSON-like strings with valid JSON strings.

    Handles model output shaped like Python string syntax inside JSON.
    """

    def _escape_inner(m: re.Match) -> str:
        inner = m.group("inner")
        return json.dumps(inner)

    return _TRIPLE_QUOTE_RE.sub(_escape_inner, body)


class _ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: str | None = None
    tool_calls: Any = None
    tool_call_id: str | None = None


class _ChatCompletionRequest(BaseModel):
    model: str
    messages: list[_ChatMessage]
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    # Accept OpenAI-compatible knobs that this shim does not forward.
    model_config = {"extra": "allow"}


def _openai_message_to_framework(msg: _ChatMessage) -> Message:
    # Preserve OpenAI tool metadata while flattening list content to text.
    content = msg.content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        content = "\n".join(parts)

    payload: Message = {
        "role": msg.role,
        "content": None if content is None else str(content),
    }
    if msg.name is not None:
        payload["name"] = msg.name
    if msg.tool_calls is not None:
        payload["tool_calls"] = msg.tool_calls
    if msg.tool_call_id is not None:
        payload["tool_call_id"] = msg.tool_call_id
    return payload


def _require_token(authorization: str | None, expected: str) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    provided = authorization.split(" ", 1)[1].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _tool_choice_override(headers: dict[str, str]) -> Any | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    override = lowered.get("x-nsl-tool-choice")
    if override is None:
        return None
    override = override.strip()
    if override == "required":
        return "required"
    if override.startswith("function:"):
        name = override.split(":", 1)[1].strip()
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


def _length_retries(headers: dict[str, str]) -> int:
    raw = {key.lower(): value for key, value in headers.items()}.get(
        "x-nsl-length-retries"
    )
    if raw is None:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return min(2, max(0, value))


def _normalize_pseudo_tool_name(name: str) -> str:
    # Dev note: ms-agent ToolManager uses server---tool registry keys, so do
    # not rewrite --- to another separator.
    return name.strip()


def _has_pseudo_tool_markers(content: str) -> bool:
    return (
        "<tool_call>" in content
        or "<call>" in content
        or "<tools>" in content
        or "<function" in content
        or "```" in content
        or "---" in content
    )


def _try_extract_json_in_tag(tool_body: str) -> dict[str, Any] | None:
    """Parse JSON-in-tag tool-call bodies emitted by Hermes/Qwen-style templates."""
    stripped = tool_body.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        return None
    name = _normalize_pseudo_tool_name(name)
    if not name:
        return None
    raw_args = parsed.get("arguments", {})
    # Arguments can arrive as an object or as a JSON-encoded string.
    if isinstance(raw_args, str):
        try:
            json.loads(raw_args)
            arguments = raw_args
        except json.JSONDecodeError:
            arguments = json.dumps({"_raw": raw_args})
    else:
        arguments = json.dumps(raw_args)
    return {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _extract_pseudo_tool_calls(content: Any) -> list[dict[str, Any]] | None:
    if not isinstance(content, str) or not _has_pseudo_tool_markers(content):
        return None

    tool_calls: list[dict[str, Any]] = []

    for tool_match in _PSEUDO_TOOL_CALL_RE.finditer(content):
        tool_body = tool_match.group("body")
        function_match = _PSEUDO_FUNCTION_RE.search(tool_body)
        if function_match is None:
            name_match = _NAME_TAG_RE.search(tool_body)
            args_match = _ARGUMENTS_TAG_RE.search(tool_body)
            if name_match is None:
                json_call = _try_extract_json_in_tag(tool_body)
                if json_call is not None:
                    json_call["id"] = f"call_{len(tool_calls) + 1}"
                    tool_calls.append(json_call)
                continue

            name = _normalize_pseudo_tool_name(name_match.group("name"))
            if not name:
                continue

            arguments = "{}"
            if args_match is not None:
                args_body = args_match.group("body")
                param_pairs = [
                    (match.group("key"), match.group("value").strip())
                    for match in _ARG_CHILD_RE.finditer(args_body)
                ]
                if param_pairs:
                    seen: set[str] = set()
                    duplicates: set[str] = set()
                    payload: dict[str, Any] = {}
                    for param_name, raw_value in param_pairs:
                        if param_name in seen:
                            duplicates.add(param_name)
                            continue
                        seen.add(param_name)
                        payload[param_name] = _coerce_parameter_value(raw_value)
                    if duplicates:
                        arguments = json.dumps(
                            {
                                "_error": (
                                    f"duplicate parameter names: {sorted(duplicates)}"
                                )
                            }
                        )
                    else:
                        arguments = json.dumps(payload)
                else:
                    stripped = args_body.strip()
                    try:
                        json.loads(stripped)
                        arguments = stripped
                    except json.JSONDecodeError:
                        arguments = json.dumps({"_raw": stripped})

            tool_calls.append(
                {
                    "id": f"call_{len(tool_calls) + 1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments or "{}",
                    },
                }
            )
            continue

        name = _normalize_pseudo_tool_name(function_match.group("name"))
        function_body = function_match.group("body")
        stripped_body = function_body.strip()
        param_pairs = [
            (match.group("name"), match.group("value").strip())
            for match in _PSEUDO_PARAMETER_RE.finditer(function_body)
        ]
        arguments = "{}"
        if not name:
            continue

        if len(param_pairs) == 1 and param_pairs[0][0] == "arguments":
            arguments = param_pairs[0][1]
        elif param_pairs:
            seen: set[str] = set()
            duplicates: set[str] = set()
            payload: dict[str, Any] = {}
            for param_name, raw_value in param_pairs:
                if param_name in seen:
                    duplicates.add(param_name)
                    continue
                seen.add(param_name)
                payload[param_name] = _coerce_parameter_value(raw_value)

            if duplicates:
                arguments = json.dumps(
                    {"_error": (f"duplicate parameter names: {sorted(duplicates)}")}
                )
            else:
                arguments = json.dumps(payload)
        else:
            try:
                json.loads(stripped_body)
                arguments = stripped_body
            except json.JSONDecodeError:
                arguments = json.dumps({"_error": "no parameters found"})

        tool_calls.append(
            {
                "id": f"call_{len(tool_calls) + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments or "{}",
                },
            }
        )

    tool_call_block_spans = {
        (m.start(), m.end()) for m in _PSEUDO_TOOL_CALL_RE.finditer(content)
    }
    next_index = len(tool_calls) + 1
    for direct_match in _DIRECT_TOOL_TAG_RE.finditer(content):
        dm_start, dm_end = direct_match.start(), direct_match.end()
        if any(s <= dm_start < e or s < dm_end <= e for s, e in tool_call_block_spans):
            continue

        name = _normalize_pseudo_tool_name(direct_match.group("name"))
        if not name:
            continue

        body = direct_match.group("body").strip()
        fixed_body = _fix_triple_quoted_json(body)
        try:
            parsed = json.loads(fixed_body)
            if isinstance(parsed, dict):
                arguments = json.dumps(parsed)
            else:
                arguments = json.dumps({"_raw": body})
        except json.JSONDecodeError:
            arguments = json.dumps({"_raw": body})

        tool_calls.append(
            {
                "id": f"call_{next_index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        next_index += 1

    # Dev note: SGLang/Qwen2.5 may wrap tool-call JSON in markdown fences when
    # template instructions are not enforced by decoding constraints.
    next_index = len(tool_calls) + 1
    for fence_match in _CODE_FENCE_BLOCK_RE.finditer(content):
        body = fence_match.group("body").strip()
        if not body.startswith("{"):
            continue
        json_call = _try_extract_json_in_tag(body)
        if json_call is None:
            continue
        json_call["id"] = f"call_{next_index}"
        tool_calls.append(json_call)
        next_index += 1

    for tools_match in _TOOLS_WRAPPER_RE.finditer(content):
        body = tools_match.group("body").strip()
        if not body.startswith("{"):
            continue
        json_call = _try_extract_json_in_tag(body)
        if json_call is None:
            continue
        json_call["id"] = f"call_{next_index}"
        tool_calls.append(json_call)
        next_index += 1

    # Quote handling: arguments JSON contains "-quoted strings, so when the
    # attribute itself is '-quoted we must NOT stop at internal " (and vice
    # versa). The model also sometimes backslash-escapes the same quote inside
    # the value (e.g. arguments='{"path": "\'/tmp\'"}'); allow \. inside.
    _attr_dq = r'"((?:\\.|[^"\\])*)"'
    _attr_sq = r"'((?:\\.|[^'\\])*)'"
    name_re = re.compile(rf"name\s*=\s*(?:{_attr_dq}|{_attr_sq})")
    args_re = re.compile(rf"arguments\s*=\s*(?:{_attr_dq}|{_attr_sq})", re.DOTALL)
    for fn_match in _FUNCTION_ATTR_RE.finditer(content):
        attrs = fn_match.group("attrs")
        name_m = name_re.search(attrs)
        args_m = args_re.search(attrs)
        if name_m is None:
            continue
        name_raw = name_m.group(1) if name_m.group(1) is not None else name_m.group(2)
        name = _normalize_pseudo_tool_name(name_raw)
        if not name:
            continue
        raw_args = "{}"
        if args_m is not None:
            raw_args = (
                args_m.group(1) if args_m.group(1) is not None else args_m.group(2)
            )
        parsed = None
        for candidate in (raw_args, raw_args.replace("\\'", "'").replace('\\"', '"')):
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            arguments = json.dumps(parsed)
        else:
            arguments = json.dumps({"_raw": raw_args})
        tool_calls.append(
            {
                "id": f"call_{next_index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        next_index += 1

    return tool_calls or None


def _strip_pseudo_tool_call_markup(content: str) -> str | None:
    stripped = _PSEUDO_TOOL_CALL_RE.sub("", content)
    stripped = _DIRECT_TOOL_TAG_RE.sub("", stripped)
    stripped = _CODE_FENCE_BLOCK_RE.sub("", stripped)
    stripped = _TOOLS_WRAPPER_RE.sub("", stripped)
    stripped = _FUNCTION_ATTR_RE.sub("", stripped)
    stripped = _THINK_TAG_RE.sub("", stripped)
    stripped = _OUTER_WRAPPER_RE.sub("", stripped)
    stripped = _CODE_FENCE_RE.sub("", stripped).strip()
    return stripped or None


class OpenAiShim:
    """Per-episode OpenAI-compatible shim backed by ``TracedGenner``."""

    def __init__(
        self,
        genner: TracedGenner,
        token: str,
        episode_id: str,
        recorder: EventRecorder,
        context_compaction: ContextCompactionSettings | None = None,
    ) -> None:
        self.genner = genner
        self.token = token
        self.episode_id = episode_id
        self._recorder = recorder
        self._context_compaction = context_compaction or ContextCompactionSettings()
        self._phase_counters: defaultdict[str, int] = defaultdict(int)
        self._phase_lock = threading.Lock()
        self.last_tool_response_classification: ToolResponseClassification | None = None
        # Read after container exit to report inference outages as harness errors.
        # A genuine outage (inference_unavailable) and a request timeout
        # (inference_timeout) are latched SEPARATELY: only the former quarantines
        # the endpoint; the latter is a benign, retryable episode failure (a
        # single timeout must not breach the capacity floor and abort the run).
        self.inference_unavailable_detail: str | None = None
        self.inference_timeout_detail: str | None = None
        self.graceful_agent_failure_detail: str | None = None
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        shim = self

        @self.app.post("/v1/chat/completions")
        def _chat_completions(  # noqa: ANN202 — fastapi-dispatched
            req: _ChatCompletionRequest,
            request: Request,
            authorization: str | None = Header(default=None),
        ):
            return shim._handle_chat_completions(req, request, authorization)

    def _resolve_phase(
        self,
        req: Any,  # _ChatCompletionRequest or compatible
        headers: dict[str, str],
    ) -> str:
        # Header names are case-insensitive, but incoming casing varies by client.
        lowered = {k.lower(): v for k, v in headers.items()}
        if nsl_phase := lowered.get("x-nsl-phase"):
            return nsl_phase
        prefix = lowered.get("x-nsl-profile", "external")
        with self._phase_lock:
            self._phase_counters[prefix] += 1
            n = self._phase_counters[prefix]
        return f"external::{prefix}::step_{n}"

    def _resolve_meta(
        self,
        req: Any,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        lowered = {k.lower(): v for k, v in headers.items()}
        meta: dict[str, Any] = {
            "client": "openai_shim",
            "model": req.model,
            "episode_id": self.episode_id,
        }
        if workflow_step := lowered.get("x-nsl-workflow-step"):
            meta["workflow_step"] = workflow_step
        if actor_role := lowered.get("x-nsl-actor-role"):
            meta["actor_role"] = actor_role
        if profile := lowered.get("x-nsl-profile"):
            meta["profile"] = profile
        return meta

    def _handle_chat_completions(
        self,
        req: _ChatCompletionRequest,
        request: Request,
        authorization: str | None,
    ) -> dict[str, Any]:
        _require_token(authorization, self.token)
        # Streaming would need a separate traced capture path.
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="streaming responses not supported by NSL OpenAI shim v1",
            )
        headers = dict(request.headers)
        if (override := _tool_choice_override(headers)) is not None:
            req.tool_choice = override
        if req.tools:
            self._recorder.bump_counter("tool_requests_total")
        phase = self._resolve_phase(req, headers)
        messages = [_openai_message_to_framework(m) for m in req.messages]
        compaction_report = compact_messages_with_report(
            messages,
            settings=self._context_compaction,
            tools=req.tools,
            tool_choice=req.tool_choice,
        )
        messages = compaction_report.messages
        self._record_context_compaction(compaction_report)
        retry_limit = _length_retries(headers)
        max_tokens = (
            req.max_completion_tokens
            if req.max_completion_tokens is not None
            else req.max_tokens
        )
        for attempt in range(retry_limit + 1):
            result = self.genner.plist_completion(
                messages,
                phase=phase,
                tools=req.tools,
                tool_choice=req.tool_choice,
                max_tokens=max_tokens,
                temperature=req.temperature,
                meta=self._resolve_meta(req, headers),
            )
            match result:
                case Ok(inference_result):
                    response, classification = (
                        _framework_to_openai_response_with_classification(
                            inference_result,
                            model=req.model,
                            allowed_names=allowed_tool_names(req.tools),
                        )
                    )
                    choice = response["choices"][0]
                    has_tool_call = bool(choice["message"].get("tool_calls"))
                    truncated_without_tool = (
                        choice["finish_reason"] == "length" and not has_tool_call
                    )
                    if truncated_without_tool and attempt < retry_limit:
                        continue
                    if truncated_without_tool and retry_limit:
                        detail = (
                            f"{GRACEFUL_AGENT_FAILURE_PREFIX} output length retries "
                            f"exhausted after {attempt + 1} attempts"
                        )
                        self.graceful_agent_failure_detail = detail
                        self._recorder.log_warning(
                            "output_length_retries_exhausted",
                            {"attempts": attempt + 1, "workflow_step": headers.get(
                                "x-nsl-workflow-step"
                            ) or headers.get("X-NSL-Workflow-Step")},
                        )
                        raise HTTPException(status_code=422, detail=detail)
                    self._recorder.bump_counter("turns")
                    self.last_tool_response_classification = classification
                    self._record_tool_response_classification(req, classification)
                    return response
                case Err(error):
                    error_str = str(error)
                    # The backend, not the agent, caused these. Latch them so
                    # ContainerHarness categorises the episode as a backend fault.
                    if error_str.startswith(INFERENCE_UNAVAILABLE_PREFIX):
                        if self.inference_unavailable_detail is None:
                            self.inference_unavailable_detail = error_str
                    elif error_str.startswith(INFERENCE_TIMEOUT_PREFIX):
                        if self.inference_timeout_detail is None:
                            self.inference_timeout_detail = error_str
                    raise HTTPException(status_code=502, detail=error_str)
        raise HTTPException(status_code=500, detail="unreachable result branch")

    def _record_context_compaction(self, report: ContextCompactionReport) -> None:
        if not report.compacted:
            return
        self._recorder.bump_counter("context_compactions_total")
        if report.compacted_tool_messages:
            self._recorder.bump_counter(
                "context_compacted_tool_messages_total",
                report.compacted_tool_messages,
            )
        if report.compacted_reasoning_messages:
            self._recorder.bump_counter(
                "context_compacted_reasoning_messages_total",
                report.compacted_reasoning_messages,
            )
        self._recorder.set_label(
            "context_compaction_tokens",
            f"{report.original_tokens}->{report.compacted_tokens}",
        )

    def _record_tool_response_classification(
        self,
        req: _ChatCompletionRequest,
        classification: ToolResponseClassification,
    ) -> None:
        if classification.source == "structured":
            self._recorder.bump_counter("tool_responses_structured_total")
        elif classification.source == "synthesized":
            self._recorder.bump_counter("tool_responses_synthesized_total")
        elif req.tools and _tool_choice_requires_tool(req.tool_choice):
            self._recorder.bump_counter("tool_responses_missing_total")
        self._recorder.set_label("tool_response_source", classification.source)


def _framework_to_openai_response(
    inference: Any,
    *,
    model: str,
) -> dict[str, Any]:
    response, _classification = _framework_to_openai_response_with_classification(
        inference,
        model=model,
    )
    return response


def _framework_to_openai_response_with_classification(
    inference: Any,
    *,
    model: str,
    allowed_names: set[str] | None = None,
) -> tuple[dict[str, Any], ToolResponseClassification]:
    usage = inference.usage
    tool_calls = getattr(inference, "tool_calls", None)
    content = inference.content
    tool_response_source: ToolResponseClassification = ToolResponseClassification(
        source="structured" if tool_calls else "missing"
    )
    if not tool_calls:
        tool_calls = _extract_pseudo_tool_calls(content)
        if tool_calls and allowed_names is not None:
            tool_calls = [
                call for call in tool_calls if _tool_call_name(call) in allowed_names
            ]
        if tool_calls:
            tool_response_source = ToolResponseClassification(source="synthesized")
    if tool_calls:
        tool_calls = _collapse_consecutive_tool_calls(tool_calls)
    synthesized_tool_calls = tool_response_source.source == "synthesized"
    if synthesized_tool_calls and isinstance(content, str):
        content = _strip_pseudo_tool_call_markup(content)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        if synthesized_tool_calls or not content:
            message["content"] = None

    finish_reason = getattr(usage, "stop_reason", None)
    if tool_calls:
        finish_reason = "tool_calls"
    if finish_reason is None:
        finish_reason = "stop"
    usage_block = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_block,
    }
    return response, tool_response_source


def _tool_call_name(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) else None


def _collapse_consecutive_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    previous: tuple[str, str] | None = None
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        signature = None
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
            if isinstance(name, str) and isinstance(arguments, str):
                signature = (name, arguments)
        if signature is not None and signature == previous:
            continue
        collapsed.append(call)
        previous = signature
    return collapsed


def _tool_choice_requires_tool(tool_choice: Any) -> bool:
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "function"
    return False
