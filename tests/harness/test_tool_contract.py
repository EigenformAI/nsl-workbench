from __future__ import annotations

from pathlib import Path

import pytest
from result import Ok

from src.genner.Base import Genner
from src.harness.profiles.ms_agent import MsAgentProfile, MsAgentProfileConfig
from src.harness.tool_contract import (
    ToolContractValidationError,
    ToolResponseClassification,
    collect_static_tool_contract_warnings,
    run_tool_call_contract_probe,
    validate_tool_contract_response,
)
from src.observability.types import InferenceResult, UsageInfo
from src.typing.config import AppConfig


class _ProbeGenner(Genner):
    def __init__(
        self,
        *,
        content: str = "ok",
        tool_calls: list[dict] | None = None,
        stop_reason: str = "stop",
    ) -> None:
        super().__init__("probe")
        self.content = content
        self.tool_calls = tool_calls
        self.stop_reason = stop_reason
        self.call_count = 0
        self.last_tools = None
        self.last_tool_choice = None

    def plist_completion(self, messages, *, tools=None, tool_choice=None):
        self.call_count += 1
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        return Ok(
            InferenceResult(
                content=self.content,
                usage=UsageInfo(
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                    stop_reason=self.stop_reason,
                ),
                tool_calls=self.tool_calls,
            )
        )

    @staticmethod
    def get_usage_info(response):
        return UsageInfo()


def _ms_profile() -> MsAgentProfile:
    return MsAgentProfile(MsAgentProfileConfig(model="nsl-model"))


def _container_config(
    tmp_path: Path,
    *,
    model_name: str = "vllm:Qwen/Qwen2.5-Coder-7B-Instruct",
    profile: str = "ms_agent",
    profile_config: dict | None = None,
    vllm: AppConfig.VllmConfig | None = None,
    sglang: AppConfig.SglangConfig | None = None,
    tool_contract: dict | None = None,
) -> AppConfig:
    if profile_config is None:
        profile_config = {"model": "nsl-model"}
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name=model_name,
        code_host_cache_path=str(tmp_path / "cache"),
        container_ids=[],
        train_data_save_folder=str(tmp_path / "train"),
        vllm=vllm,
        sglang=sglang,
        tool_contract=tool_contract or {},
        harness={
            "name": "container",
            "container": {
                "profile": profile,
                "image": "nsl/test:latest",
                "profile_config": profile_config,
            },
        },
    )


def _response(
    *,
    name: str = "nsl---contract_probe",
    arguments: str = '{"message":"ok"}',
    finish_reason: str = "tool_calls",
    tool_calls: list[dict] | None = None,
    content: str | None = None,
) -> dict:
    if tool_calls is None:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": finish_reason,
            }
        ]
    }


def test_ms_agent_contract_probe_uses_ms_agent_tool_shape() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    assert probe is not None
    assert probe.expected_tool_name == "nsl---contract_probe"
    assert probe.expected_argument_keys == {"message"}
    assert probe.expected_arguments == {"message": "ok"}
    assert probe.tool_choice == "auto"
    assert probe.stream is False
    assert probe.tools[0]["function"]["name"] == "nsl---contract_probe"
    assert "MUST call exactly" in probe.messages[-1]["content"]


def test_validate_tool_contract_response_accepts_structured_tool_calls() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    result = validate_tool_contract_response(
        _response(),
        probe,
        classification=ToolResponseClassification(source="structured"),
    )

    assert result.tool_response_source == "structured"
    assert result.observed_arguments == {"message": "ok"}


def test_validate_tool_contract_response_accepts_synthesized_tool_calls() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    result = validate_tool_contract_response(
        _response(),
        probe,
        classification=ToolResponseClassification(source="synthesized"),
    )

    assert result.tool_response_source == "synthesized"


def test_validate_tool_contract_response_rejects_missing_tool_calls() -> None:
    probe = _ms_profile().tool_call_contract_probe()
    response = _response(tool_calls=[], content="plain text instead of a tool call")

    with pytest.raises(ToolContractValidationError) as exc:
        validate_tool_contract_response(response, probe)

    assert exc.value.diagnostics["tool_response_source"] == "missing"
    assert "plain text" in exc.value.diagnostics["observed_content_prefix"]


def test_validate_tool_contract_response_rejects_wrong_function_name() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    with pytest.raises(ToolContractValidationError, match="wrong tool name"):
        validate_tool_contract_response(_response(name="nsl---wrong"), probe)


def test_validate_tool_contract_response_rejects_non_json_arguments() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    with pytest.raises(ToolContractValidationError, match="valid JSON"):
        validate_tool_contract_response(_response(arguments="not json"), probe)


def test_validate_tool_contract_response_rejects_missing_required_argument() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    with pytest.raises(ToolContractValidationError, match="missing required"):
        validate_tool_contract_response(_response(arguments='{"other":"ok"}'), probe)


def test_validate_tool_contract_response_rejects_non_tool_finish_reason() -> None:
    probe = _ms_profile().tool_call_contract_probe()

    with pytest.raises(ToolContractValidationError, match="finish_reason"):
        validate_tool_contract_response(_response(finish_reason="stop"), probe)


def test_static_warnings_warn_for_vllm_auto_tool_choice_disabled(
    tmp_path: Path,
) -> None:
    config = _container_config(
        tmp_path,
        vllm=AppConfig.VllmConfig(enable_auto_tool_choice=False),
    )

    warnings = collect_static_tool_contract_warnings(config)

    assert "vllm_auto_tool_choice_disabled" in {w.warning_code for w in warnings}


def test_static_warnings_warn_for_sglang_missing_parser(tmp_path: Path) -> None:
    config = _container_config(
        tmp_path,
        model_name="sglang:Acme/NoTools-7B",
        sglang=AppConfig.SglangConfig(),
    )

    warnings = collect_static_tool_contract_warnings(config)

    assert "sglang_missing_tool_call_parser" in {w.warning_code for w in warnings}


def test_static_warnings_warn_for_unknown_parser(tmp_path: Path) -> None:
    config = _container_config(
        tmp_path,
        vllm=AppConfig.VllmConfig(tool_call_parser="not-a-parser"),
    )

    warnings = collect_static_tool_contract_warnings(config)

    assert "unknown_tool_call_parser" in {w.warning_code for w in warnings}


def test_dynamic_probe_passes_structured_tool_call(tmp_path: Path) -> None:
    profile = _ms_profile()
    genner = _ProbeGenner(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "nsl---contract_probe",
                    "arguments": '{"message":"ok"}',
                },
            }
        ],
        stop_reason="tool_calls",
    )
    config = _container_config(tmp_path)

    result = run_tool_call_contract_probe(
        config=config,
        profile=profile,
        genner=genner,
        backend_session=type("Session", (), {"config": type("Cfg", (), {"model": "nsl-model"})()})(),
        run_id="run-1",
    )

    assert result.status == "passed"
    assert result.tool_response_source == "structured"
    assert genner.last_tool_choice == "auto"


def test_dynamic_probe_warn_enforcement_logs_failure_but_continues(
    tmp_path: Path,
) -> None:
    profile = _ms_profile()
    genner = _ProbeGenner(content="plain text", stop_reason="stop")
    config = _container_config(
        tmp_path,
        tool_contract={"enforcement": "warn", "unsafe_reason": "rollout"},
    )

    result = run_tool_call_contract_probe(
        config=config,
        profile=profile,
        genner=genner,
        backend_session=type("Session", (), {"config": type("Cfg", (), {"model": "nsl-model"})()})(),
        run_id="run-1",
    )

    assert result.status == "failed"
    assert result.diagnostics["enforcement"] == "warn"


def test_dynamic_probe_fail_enforcement_aborts_on_failure(tmp_path: Path) -> None:
    profile = _ms_profile()
    genner = _ProbeGenner(content="plain text", stop_reason="stop")
    config = _container_config(tmp_path)

    with pytest.raises(ToolContractValidationError):
        run_tool_call_contract_probe(
            config=config,
            profile=profile,
            genner=genner,
            backend_session=type("Session", (), {"config": type("Cfg", (), {"model": "nsl-model"})()})(),
            run_id="run-1",
        )


def test_dynamic_probe_skip_enforcement_does_not_call_genner(tmp_path: Path) -> None:
    profile = _ms_profile()
    genner = _ProbeGenner(content="plain text", stop_reason="stop")
    config = _container_config(
        tmp_path,
        tool_contract={"enforcement": "skip", "unsafe_reason": "custom template"},
    )

    result = run_tool_call_contract_probe(
        config=config,
        profile=profile,
        genner=genner,
        backend_session=type("Session", (), {"config": type("Cfg", (), {"model": "nsl-model"})()})(),
        run_id="run-1",
    )

    assert result.status == "skipped"
    assert genner.call_count == 0
