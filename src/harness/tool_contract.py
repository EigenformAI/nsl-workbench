from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi.testclient import TestClient
from loguru import logger

from src.backend._tool_parsers import infer_tool_call_parser
from src.genner.Base import Genner


ToolResponseSource = Literal["structured", "synthesized", "missing", "malformed"]


@dataclass(frozen=True)
class ToolCallContractProbe:
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    expected_tool_name: str
    expected_argument_keys: set[str]
    expected_arguments: dict[str, Any] = field(default_factory=dict)
    tool_choice: Any | None = None
    headers: dict[str, str] = field(default_factory=dict)
    stream: Literal[False] = False


@dataclass(frozen=True)
class ToolResponseClassification:
    source: ToolResponseSource
    dialect: str | None = None


@dataclass(frozen=True)
class ToolContractStaticWarning:
    backend: str | None
    model: str
    harness_profile: str | None
    tool_call_parser: str | None
    chat_template_path: str | None
    stream_mode: str
    warning_code: str
    suggested_fix: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "harness_profile": self.harness_profile,
            "tool_call_parser": self.tool_call_parser,
            "chat_template_path": self.chat_template_path,
            "stream_mode": self.stream_mode,
            "warning_code": self.warning_code,
            "suggested_fix": self.suggested_fix,
            **self.details,
        }


@dataclass(frozen=True)
class ToolContractProbeRunResult:
    status: Literal["passed", "failed", "skipped"]
    diagnostics: dict[str, Any]
    tool_response_source: ToolResponseSource | None = None
    observed_arguments: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolContractValidationResult:
    tool_response_source: ToolResponseSource
    observed_arguments: dict[str, Any]
    diagnostics: dict[str, Any]


class ToolContractValidationError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


_VLLM_KNOWN_PARSERS = {
    "deepseek_v3",
    "granite",
    "hermes",
    "internlm",
    "jamba",
    "llama3_json",
    "llama4_pythonic",
    "mistral",
    "pythonic",
}
_SGLANG_KNOWN_PARSERS = {
    "deepseekv3",
    "gemma3",
    "hermes",
    "kimi_k2",
    "llama3",
    "mistral",
    "qwen25",
}


def allowed_tool_names(tools: list[dict[str, Any]] | None) -> set[str] | None:
    if tools is None:
        return None
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def collect_static_tool_contract_warnings(config: Any) -> list[ToolContractStaticWarning]:
    profile = _resolve_contract_profile(config)
    probe = profile.tool_call_contract_probe() if profile is not None else None
    if probe is None:
        return []

    backend = _backend_name(config)
    model = _source_model_name(config)
    parser = _effective_tool_call_parser(config, backend, model)
    chat_template_path = _chat_template_path(config, backend)
    stream_mode = "stream=false" if probe.stream is False else "stream=true"
    base = {
        "backend": backend,
        "model": model,
        "harness_profile": getattr(profile, "name", None),
        "tool_call_parser": parser,
        "chat_template_path": chat_template_path,
        "stream_mode": stream_mode,
    }
    warnings: list[ToolContractStaticWarning] = []

    def add(code: str, fix: str, **details: Any) -> None:
        warnings.append(
            ToolContractStaticWarning(
                warning_code=code,
                suggested_fix=fix,
                details=details,
                **base,
            )
        )

    if probe.stream is not False:
        add(
            "streaming_tool_contract_unsupported",
            "Set the profile/backend path to non-streaming; OpenAiShim v1 validates only stream=false.",
        )

    if backend == "vllm":
        vllm_cfg = getattr(config, "vllm", None)
        auto = _effective_vllm_auto_tool_choice(config, parser)
        if vllm_cfg is not None and vllm_cfg.enable_auto_tool_choice is False:
            add(
                "vllm_auto_tool_choice_disabled",
                "Set [vllm].enable_auto_tool_choice=true or use a profile path that deterministically forces tool calls.",
            )
        if auto and parser is None:
            add(
                "vllm_missing_tool_call_parser",
                "Set [vllm].tool_call_parser to the parser matching the model/chat template.",
            )
        if parser is not None and parser not in _VLLM_KNOWN_PARSERS:
            add(
                "unknown_tool_call_parser",
                "Use a parser supported by vLLM or update the contract metadata for this backend version.",
                known_parsers=sorted(_VLLM_KNOWN_PARSERS),
            )
        if _is_qwen_family(model) and parser not in {None, "hermes"}:
            add(
                "qwen_parser_template_risk",
                "Qwen/QwQ vLLM configs usually need tool_call_parser='hermes' with a matching tool chat template.",
            )

    if backend == "sglang":
        sglang_cfg = getattr(config, "sglang", None)
        grammar_backend = getattr(sglang_cfg, "grammar_backend", "xgrammar")
        if parser is None:
            add(
                "sglang_missing_tool_call_parser",
                "Set [sglang].tool_call_parser to the parser matching the model/chat template.",
            )
        if _probe_forces_required_tool_choice(probe) and grammar_backend != "xgrammar":
            add(
                "sglang_forced_tool_choice_requires_xgrammar",
                "Set [sglang].grammar_backend='xgrammar' when exercising forced tool choice.",
            )
        if parser is not None and parser not in _SGLANG_KNOWN_PARSERS:
            add(
                "unknown_tool_call_parser",
                "Use a parser supported by SGLang or update the contract metadata for this backend version.",
                known_parsers=sorted(_SGLANG_KNOWN_PARSERS),
            )
        if _is_qwen_family(model):
            if parser not in {None, "qwen25"}:
                add(
                    "qwen_parser_template_risk",
                    "Qwen/QwQ SGLang configs usually need tool_call_parser='qwen25' with a matching Qwen tool chat template.",
                )
            if not chat_template_path:
                add(
                    "qwen_chat_template_missing",
                    "Set [sglang].chat_template_path to the Qwen2.5/QwQ tool-call chat template used by the parser.",
                )

    return warnings


def emit_static_tool_contract_warnings(config: Any) -> list[ToolContractStaticWarning]:
    warnings = collect_static_tool_contract_warnings(config)
    for warning in warnings:
        logger.bind(**warning.to_payload()).warning("tool_contract_static_warning")
    return warnings


def validate_tool_contract_response(
    response: dict[str, Any],
    probe: ToolCallContractProbe,
    *,
    classification: ToolResponseClassification | None = None,
) -> ToolContractValidationResult:
    choice = _first_choice(response)
    message = choice.get("message") if isinstance(choice, dict) else None
    message = message if isinstance(message, dict) else {}
    tool_calls = message.get("tool_calls")
    source: ToolResponseSource = (
        classification.source
        if classification is not None
        else ("structured" if tool_calls else "missing")
    )
    diagnostics = _response_diagnostics(
        response=response,
        probe=probe,
        tool_response_source=source,
    )

    if not isinstance(tool_calls, list) or not tool_calls:
        diagnostics["tool_response_source"] = "missing"
        raise ToolContractValidationError(
            "tool contract probe failed: missing tool_calls",
            diagnostics,
        )

    first = tool_calls[0]
    if not isinstance(first, dict):
        diagnostics["tool_response_source"] = "malformed"
        raise ToolContractValidationError(
            "tool contract probe failed: malformed tool call",
            diagnostics,
        )
    if first.get("type") != "function":
        diagnostics["tool_response_source"] = "malformed"
        raise ToolContractValidationError(
            "tool contract probe failed: first tool call type is not function",
            diagnostics,
        )
    function = first.get("function")
    if not isinstance(function, dict):
        diagnostics["tool_response_source"] = "malformed"
        raise ToolContractValidationError(
            "tool contract probe failed: missing function payload",
            diagnostics,
        )

    observed_name = function.get("name")
    if observed_name != probe.expected_tool_name:
        diagnostics["tool_response_source"] = "malformed"
        diagnostics["observed_tool_name"] = observed_name
        raise ToolContractValidationError(
            "tool contract probe failed: wrong tool name",
            diagnostics,
        )

    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        diagnostics["tool_response_source"] = "malformed"
        raise ToolContractValidationError(
            "tool contract probe failed: function.arguments is not a JSON string",
            diagnostics,
        )
    try:
        parsed_arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        diagnostics["tool_response_source"] = "malformed"
        diagnostics["observed_arguments"] = raw_arguments
        raise ToolContractValidationError(
            "tool contract probe failed: function.arguments is not valid JSON",
            diagnostics,
        )
    if not isinstance(parsed_arguments, dict):
        diagnostics["tool_response_source"] = "malformed"
        diagnostics["observed_arguments"] = parsed_arguments
        raise ToolContractValidationError(
            "tool contract probe failed: function.arguments JSON is not an object",
            diagnostics,
        )

    missing = sorted(probe.expected_argument_keys - set(parsed_arguments))
    if missing:
        diagnostics["tool_response_source"] = "malformed"
        diagnostics["observed_arguments"] = parsed_arguments
        diagnostics["missing_argument_keys"] = missing
        raise ToolContractValidationError(
            "tool contract probe failed: missing required argument keys",
            diagnostics,
        )
    wrong_values = {
        key: {"expected": value, "observed": parsed_arguments.get(key)}
        for key, value in probe.expected_arguments.items()
        if parsed_arguments.get(key) != value
    }
    if wrong_values:
        diagnostics["tool_response_source"] = "malformed"
        diagnostics["observed_arguments"] = parsed_arguments
        diagnostics["wrong_argument_values"] = wrong_values
        raise ToolContractValidationError(
            "tool contract probe failed: wrong sentinel argument value",
            diagnostics,
        )

    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if finish_reason != "tool_calls":
        diagnostics["tool_response_source"] = "malformed"
        diagnostics["observed_arguments"] = parsed_arguments
        raise ToolContractValidationError(
            "tool contract probe failed: finish_reason is not tool_calls",
            diagnostics,
        )

    diagnostics["observed_arguments"] = parsed_arguments
    return ToolContractValidationResult(
        tool_response_source=source,
        observed_arguments=parsed_arguments,
        diagnostics=diagnostics,
    )


def run_tool_call_contract_probe(
    *,
    config: Any,
    profile: Any,
    genner: Genner,
    backend_session: Any,
    run_id: str,
) -> ToolContractProbeRunResult:
    probe = profile.tool_call_contract_probe()
    metadata = _dynamic_metadata(config, profile, backend_session, probe)
    enforcement = getattr(config.tool_contract, "enforcement", "fail")
    unsafe_reason = getattr(config.tool_contract, "unsafe_reason", None)
    metadata["enforcement"] = enforcement
    if unsafe_reason:
        metadata["unsafe_reason"] = unsafe_reason

    if probe is None:
        payload = {**metadata, "skip_reason": "profile has no tool-call contract probe"}
        logger.bind(**payload).info("tool_contract_probe_skipped")
        return ToolContractProbeRunResult(status="skipped", diagnostics=payload)
    if enforcement == "skip":
        payload = {**metadata, "skip_reason": "tool_contract.enforcement=skip"}
        logger.bind(**payload).warning("tool_contract_probe_skipped")
        return ToolContractProbeRunResult(status="skipped", diagnostics=payload)

    logger.bind(**metadata).info("tool_contract_probe_started")
    try:
        response, classification = _send_probe_through_shim(
            config=config,
            probe=probe,
            genner=genner,
            backend_model=metadata["model"],
            run_id=run_id,
        )
        validation = validate_tool_contract_response(
            response,
            probe,
            classification=classification,
        )
    except ToolContractValidationError as exc:
        diagnostics = {**metadata, **exc.diagnostics}
        logger.bind(**diagnostics).warning("tool_contract_probe_failed")
        failure = ToolContractProbeRunResult(
            status="failed",
            diagnostics=diagnostics,
            tool_response_source=diagnostics.get("tool_response_source"),
        )
        if enforcement == "warn":
            return failure
        raise ToolContractValidationError(str(exc), diagnostics) from exc
    except Exception as exc:
        diagnostics = {
            **metadata,
            "tool_response_source": "missing",
            "error": str(exc),
        }
        logger.bind(**diagnostics).warning("tool_contract_probe_failed")
        if enforcement == "warn":
            return ToolContractProbeRunResult(
                status="failed",
                diagnostics=diagnostics,
                tool_response_source="missing",
            )
        raise ToolContractValidationError(
            "tool contract probe failed before receiving a valid response",
            diagnostics,
        ) from exc

    diagnostics = {**metadata, **validation.diagnostics}
    logger.bind(**diagnostics).info("tool_contract_probe_passed")
    return ToolContractProbeRunResult(
        status="passed",
        diagnostics=diagnostics,
        tool_response_source=validation.tool_response_source,
        observed_arguments=validation.observed_arguments,
    )


def _send_probe_through_shim(
    *,
    config: Any,
    probe: ToolCallContractProbe,
    genner: Genner,
    backend_model: str,
    run_id: str,
) -> tuple[dict[str, Any], ToolResponseClassification | None]:
    from src.harness.openai_shim import OpenAiShim
    from src.harness.recorder import EventRecorder
    from src.harness.traced_genner import TracedGenner

    episode_id = "contract_probe"
    token = "tool-contract-probe-token"
    output_path = (
        Path(config.train_data_save_folder)
        / "tool_contract_probe"
        / f"{run_id}.jsonl"
    )
    recorder = EventRecorder(episode_id=episode_id, output_path=output_path)
    traced = TracedGenner(
        inner=genner,
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id=episode_id,
    )
    shim = OpenAiShim(traced, token=token, episode_id=episode_id, recorder=recorder)
    body: dict[str, Any] = {
        "model": backend_model,
        "messages": probe.messages,
        "tools": probe.tools,
        "stream": probe.stream,
    }
    if probe.tool_choice is not None:
        body["tool_choice"] = probe.tool_choice
    headers = {"Authorization": f"Bearer {token}", **probe.headers}
    with TestClient(shim.app) as client:
        response = client.post("/v1/chat/completions", json=body, headers=headers)
    if response.status_code >= 400:
        raise ToolContractValidationError(
            f"tool contract probe HTTP request failed with {response.status_code}",
            {
                "http_status": response.status_code,
                "http_body_prefix": response.text[:500],
                "tool_response_source": "missing",
            },
        )
    return response.json(), shim.last_tool_response_classification


def _resolve_contract_profile(config: Any) -> Any | None:
    harness = getattr(config, "harness", None)
    container = getattr(harness, "container", None)
    if getattr(harness, "name", None) != "container" or container is None:
        return None
    from src.harness.profiles import resolve_profile

    return resolve_profile(container.profile, container.profile_config)


def _backend_name(config: Any) -> str | None:
    model_name = getattr(config, "model_name", "") or ""
    if model_name.startswith("vllm:"):
        return "vllm"
    if model_name.startswith("sglang:"):
        return "sglang"
    if getattr(config, "vllm", None) is not None:
        return "vllm"
    if getattr(config, "sglang", None) is not None:
        return "sglang"
    return None


def _source_model_name(config: Any) -> str:
    model_name = getattr(config, "model_name", "") or ""
    if model_name.startswith(("vllm:", "sglang:")):
        return model_name.split(":", 1)[1].strip()
    return model_name.strip()


def _effective_tool_call_parser(
    config: Any,
    backend: str | None,
    source_model: str,
) -> str | None:
    if backend not in {"vllm", "sglang"}:
        return None
    backend_cfg = getattr(config, backend, None)
    configured = getattr(backend_cfg, "tool_call_parser", None)
    if configured:
        return configured
    return infer_tool_call_parser(backend, source_model)


def _effective_vllm_auto_tool_choice(config: Any, parser: str | None) -> bool:
    vllm_cfg = getattr(config, "vllm", None)
    configured = getattr(vllm_cfg, "enable_auto_tool_choice", None)
    if configured is None:
        return parser is not None
    return bool(configured)


def _chat_template_path(config: Any, backend: str | None) -> str | None:
    backend_cfg = getattr(config, backend or "", None)
    return getattr(backend_cfg, "chat_template_path", None)


def _probe_forces_required_tool_choice(probe: ToolCallContractProbe) -> bool:
    headers = {key.lower(): value for key, value in probe.headers.items()}
    return headers.get("x-nsl-tool-choice") == "required" or probe.tool_choice == "required"


def _is_qwen_family(model: str) -> bool:
    lowered = model.lower()
    return "qwen" in lowered or "qwq" in lowered


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _response_diagnostics(
    *,
    response: dict[str, Any],
    probe: ToolCallContractProbe,
    tool_response_source: ToolResponseSource,
) -> dict[str, Any]:
    choice = _first_choice(response)
    message = choice.get("message") if isinstance(choice, dict) else None
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    prefix = content[:500] if isinstance(content, str) else None
    return {
        "validation_path": "shim",
        "expected_tool_name": probe.expected_tool_name,
        "expected_argument_keys": sorted(probe.expected_argument_keys),
        "tool_choice_mode": probe.tool_choice,
        "headers_used": dict(probe.headers),
        "stream": probe.stream,
        "observed_content_prefix": prefix,
        "observed_tool_calls": message.get("tool_calls"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "tool_response_source": tool_response_source,
        "suggested_fix": _default_suggested_fix(),
    }


def _dynamic_metadata(
    config: Any,
    profile: Any,
    backend_session: Any,
    probe: ToolCallContractProbe | None,
) -> dict[str, Any]:
    backend = _backend_name(config)
    session_config = getattr(backend_session, "config", None)
    model = getattr(session_config, "model", None) or _source_model_name(config)
    parser = getattr(session_config, "tool_call_parser", None)
    if parser is None:
        parser = _effective_tool_call_parser(config, backend, _source_model_name(config))
    return {
        "validation_path": "shim",
        "backend": backend,
        "model": model,
        "harness_profile": getattr(profile, "name", None),
        "tool_call_parser": parser,
        "chat_template_path": _chat_template_path(config, backend),
        "tool_choice_mode": getattr(probe, "tool_choice", None),
        "headers_used": dict(getattr(probe, "headers", {}) or {}),
        "stream": False if probe is None else probe.stream,
        "expected_tool_name": getattr(probe, "expected_tool_name", None),
        "suggested_fix": _default_suggested_fix(),
    }


def _default_suggested_fix() -> str:
    return (
        "Check chat_template_path, tool_call_parser, backend auto/forced tool-choice "
        "settings, streaming mode, or opt out explicitly with [tool_contract] and an unsafe_reason."
    )


__all__ = [
    "ToolCallContractProbe",
    "ToolContractProbeRunResult",
    "ToolContractStaticWarning",
    "ToolContractValidationError",
    "ToolContractValidationResult",
    "ToolResponseClassification",
    "allowed_tool_names",
    "collect_static_tool_contract_warnings",
    "emit_static_tool_contract_warnings",
    "run_tool_call_contract_probe",
    "validate_tool_contract_response",
]
