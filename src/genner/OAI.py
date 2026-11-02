from pprint import pformat
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from loguru import logger
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
from result import Result, Ok, Err

from dataclasses import dataclass, field

from src.genner.Base import (
    CONTEXT_OVERFLOW_PREFIX,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message
from .Base import Genner


def _normalize_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Rewrap ms-agent flat tool dicts into OpenAI function-call shape.

    ms-agent's continuation path (_continue_generate) forwards raw Tool
    TypedDicts {tool_name, description, parameters} instead of the
    OpenAI-expected {type: "function", function: {name, ...}}.  Detect
    the flat shape and rewrap so vLLM doesn't 400.
    """
    if not tools:
        return tools
    first = tools[0]
    if "function" in first or first.get("type") == "function":
        return tools  # already OpenAI-shaped
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("tool_name", t.get("name", "")),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in tools
    ]


@runtime_checkable
class OAICompatibleConfig(Protocol):
    name: str
    model: str
    max_tokens: int | None
    temperature: float


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


class OAIConfig(NamedTuple):
    name: str = "OpenAI"
    model: str = "gpt-3.5-turbo"
    max_tokens: int | None = 500
    temperature: float = 0.5


class OAIUsageResponse(Protocol):
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class OAIChoiceResponse(Protocol):
    finish_reason: str | None


class OAIChatResponse(Protocol):
    usage: OAIUsageResponse | None
    choices: Sequence[OAIChoiceResponse]
    model: str


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return dict(tool_call)
    if hasattr(tool_call, "model_dump"):
        dumped = tool_call.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped

    payload: dict[str, Any] = {}
    identifier = getattr(tool_call, "id", None)
    if identifier is not None:
        payload["id"] = identifier
    tool_type = getattr(tool_call, "type", None)
    if tool_type is not None:
        payload["type"] = tool_type

    function = getattr(tool_call, "function", None)
    if function is not None:
        if isinstance(function, dict):
            payload["function"] = dict(function)
        elif hasattr(function, "model_dump"):
            dumped = function.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                payload["function"] = dumped
        else:
            function_payload: dict[str, Any] = {}
            name = getattr(function, "name", None)
            if name is not None:
                function_payload["name"] = name
            arguments = getattr(function, "arguments", None)
            if arguments is not None:
                function_payload["arguments"] = arguments
            if function_payload:
                payload["function"] = function_payload

    return payload


def _tool_calls_to_jsonable(tool_calls: Any) -> list[dict[str, Any]] | None:
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        tool_calls = [tool_calls]

    payload = [_tool_call_to_dict(call) for call in tool_calls]
    return [call for call in payload if call]


class OAIGenner(Genner):
    def __init__(
        self,
        client: OpenAI,
        config: OAICompatibleConfig,
        identifier: str = "oai",
    ):
        super().__init__(identifier)

        self.client = client
        self.config = config

    def _resolve_model(self, messages: List[Message]) -> str:
        return self.config.model.strip()

    def _prepare_messages(self, messages: List[Message]) -> List[Message]:
        return messages

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
            request_model = self._resolve_model(messages)
            request_messages = self._prepare_messages(messages)
            logger.debug(f"OpenAI model: {request_model}")
            extra_kwargs: Dict[str, Any] = {}
            frequency_penalty = getattr(self.config, "frequency_penalty", None)
            if frequency_penalty is not None:
                extra_kwargs["frequency_penalty"] = frequency_penalty
            presence_penalty = getattr(self.config, "presence_penalty", None)
            if presence_penalty is not None:
                extra_kwargs["presence_penalty"] = presence_penalty
            seed = getattr(self.config, "seed", None)
            if seed is not None:
                extra_kwargs["seed"] = seed
            if tools is not None:
                extra_kwargs["tools"] = _normalize_tools(tools)
            if tool_choice is not None:
                extra_kwargs["tool_choice"] = tool_choice
            # Omit max_tokens entirely when None so vLLM uses its own
            # default (max_model_len - prompt_tokens). Passing a numeric
            # value through unchanged would 400 if it implies a request
            # that overflows max_model_len once the prompt is counted.
            request_max_tokens = (
                max_tokens if max_tokens is not None else self.config.max_tokens
            )
            if request_max_tokens is not None:
                extra_kwargs["max_tokens"] = request_max_tokens
            response = self.client.chat.completions.create(
                model=request_model,
                # response_format={"type": "json_object"},
                messages=cast(Any, request_messages),
                temperature=(
                    temperature if temperature is not None else self.config.temperature
                ),
                **extra_kwargs,
            )

            message = response.choices[0].message
            content = message.content
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)
            # Keep reasoning in live assistant replay and separately observable;
            # TracedGenner removes this exact prefix at the recorder/SFT boundary.
            reasoning = getattr(message, "reasoning_content", None)
            if not isinstance(reasoning, str) or not reasoning:
                reasoning = None
            else:
                content = f"<think>{reasoning}</think>{content}"
            tool_calls_payload = _tool_calls_to_jsonable(
                getattr(message, "tool_calls", None)
            )

            return Ok(
                InferenceResult(
                    content=content,
                    usage=self.get_usage_info(response),
                    tool_calls=tool_calls_payload,
                    reasoning_content=reasoning,
                )
            )
        except BadRequestError as e:
            if self.is_context_overflow(e):
                msg = str(getattr(e, "message", e))
                logger.warning(
                    "OAIGenner.plist_completion: context overflow "
                    f"(model={self.config.model}, max_tokens={self.config.max_tokens}): {msg}"
                )
                return Err(f"{CONTEXT_OVERFLOW_PREFIX} {msg}")
            logger.exception(
                f"OAIGenner.plist_completion failed for model {self.config.model}"
            )
            return Err(
                "OAIGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )
        except APITimeoutError as e:
            # NOTE: APITimeoutError subclasses APIConnectionError, so this
            # MUST be caught before the APIConnectionError branch below.
            #
            # A request-level timeout: the client gave up waiting (e.g. decode
            # starvation under load) but the endpoint is still reachable. This
            # is a RETRYABLE episode failure, NOT an endpoint outage. Tag it
            # with the timeout prefix so the endpoint pool does NOT quarantine
            # the (possibly sole) endpoint — quarantining the only endpoint
            # would breach the capacity floor and abort the whole run. Logged
            # at WARNING (not exception) because timeouts are expected under
            # load and a per-timeout traceback floods the run log.
            logger.warning(
                f"OAIGenner.plist_completion: inference request timed out "
                f"for model {self.config.model}: {e}"
            )
            return Err(f"{INFERENCE_TIMEOUT_PREFIX} {type(e).__name__}: {e}")
        except APIConnectionError as e:
            # Endpoint unreachable at the transport layer (e.g. vLLM crashed,
            # network blip). Tag with the inference_unavailable prefix so the
            # endpoint pool quarantines it and fails over to healthy endpoints,
            # and the harness elevates this to a HarnessError that the existing
            # consecutive_harness_error_limit circuit breaker can abort on.
            logger.exception(
                f"OAIGenner.plist_completion: inference endpoint "
                f"unavailable for model {self.config.model}"
            )
            return Err(f"{INFERENCE_UNAVAILABLE_PREFIX} {type(e).__name__}: {e}")
        except Exception as e:
            logger.exception(
                f"OAIGenner.plist_completion failed for model {self.config.model}"
            )
            return Err(
                "OAIGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )

    @staticmethod
    def is_context_overflow(exc: BadRequestError) -> bool:
        body = getattr(exc, "body", None)
        error_payload = body.get("error", {}) if isinstance(body, dict) else {}
        code = error_payload.get("code") or getattr(exc, "code", None)
        param = error_payload.get("param") or getattr(exc, "param", None)
        message_parts = [
            str(getattr(exc, "message", "") or ""),
            str(error_payload.get("message", "") or ""),
            str(code or ""),
            str(param or ""),
        ]
        message = " ".join(part for part in message_parts if part).lower()
        return (
            "maximum context length" in message
            or "context length exceeded" in message
            or "max context" in message
            or code == "context_length_exceeded"
            or (isinstance(param, str) and "input_tokens" in param.lower())
        )

    @staticmethod
    def get_usage_info(response: object) -> UsageInfo:
        response = cast(OAIChatResponse, response)
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else None
        completion_tokens = usage.completion_tokens if usage is not None else None
        total_tokens = usage.total_tokens if usage is not None else None
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        stop_reason = response.choices[0].finish_reason if response.choices else None

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=response.model,
            stop_reason=stop_reason,
        )
