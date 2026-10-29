"""ContainerHarness elevates `agent_failure` to `endpoint_unavailable` when the
inference endpoint went away mid-episode.

A non-zero container exit normally lands as ``agent_failure`` (the agent
chose to terminate / errored). When the shim observed an
``inference_unavailable`` Err from the genner, the cause is the backend,
not the agent. The harness re-categorises so endpoint-level routing, not the
per-slot harness breaker, handles the outage.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.genner.Base import INFERENCE_TIMEOUT_PREFIX, INFERENCE_UNAVAILABLE_PREFIX
from src.harness.budget import BudgetLedger, ToolCallRequest
from src.harness.container import ContainerHarness
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import BudgetConstraints, Capability, TaskPromptSpec, Variation


class _ExitFailContainer:
    """Fake Docker container that exits non-zero immediately."""

    def __init__(self) -> None:
        self.id = "exit-1"
        self.removed = False

    def wait(self, **_kwargs):
        return {"StatusCode": 1}

    def reload(self) -> None: ...

    @property
    def status(self) -> str:
        return "exited"

    def logs(self, **_kwargs) -> bytes:
        return b"agent crashed"

    def kill(self) -> None: ...

    def remove(self, **_kwargs) -> None:
        self.removed = True


class _StubTask:
    metric_name = "dummy"
    metric_unit = ""
    name = "stub"
    description = "stub"

    def parse_response(self, raw_response, *, invoked_capability=None):
        return []

    def execute_capability(self, invocation, containers, variation, ctx):
        from src.task.types import CapabilityResult

        return CapabilityResult(name=invocation.name, output={}, success=True)


def _ctx(tmp_path: Path, docker_client: Any) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=MagicMock(),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=[Capability(name="analyzer", description="r")],
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=_StubTask(),  # type: ignore[arg-type]
        variation=Variation(name="v", description="d"),
        prompt_spec=prompt_spec,
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings={},
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        docker_client=docker_client,
    )


def _profile_mock() -> MagicMock:
    p = MagicMock()
    p.render_query.return_value = "q"
    p.default_args.return_value = ["python", "/opt/nsl/run.py"]
    p.env.return_value = {}
    p.count_llm_turns.return_value = 0
    p.read_transcript.return_value = None
    p.to_artifacts.return_value = MagicMock(
        capability_invocations=[], capability_results=[], final_response=None
    )
    return p


def _patch_harness_env(
    monkeypatch: pytest.MonkeyPatch,
    profile_mock: MagicMock,
    flag_shim: bool,
    timeout_shim: bool = False,
    graceful_agent_failure_detail: str | None = None,
) -> None:
    import src.harness.container as container_mod

    monkeypatch.setattr(
        container_mod,
        "_serve_host_shim",
        lambda app: MagicMock(port=9001, stop=lambda: None),
    )
    monkeypatch.setattr(
        container_mod.CapabilityMcpBridge,
        "serve_http",
        lambda self: MagicMock(port=9002, stop=lambda: None),
    )
    monkeypatch.setattr(
        container_mod,
        "resolve_profile",
        lambda name, cfg: profile_mock,
    )
    monkeypatch.setattr(
        container_mod,
        "_docker_host_gateway",
        lambda client, network_mode=None: "172.17.0.1",
    )

    original_shim_cls = container_mod.OpenAiShim

    class _ShimWithFlag(original_shim_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if flag_shim:
                self.inference_unavailable_detail = (
                    f"{INFERENCE_UNAVAILABLE_PREFIX} Connection refused"
                )
            if timeout_shim:
                self.inference_timeout_detail = (
                    f"{INFERENCE_TIMEOUT_PREFIX} APITimeoutError: Request timed out."
                )
            if graceful_agent_failure_detail is not None:
                self.graceful_agent_failure_detail = graceful_agent_failure_detail

    monkeypatch.setattr(container_mod, "OpenAiShim", _ShimWithFlag)


def _harness() -> ContainerHarness:
    return ContainerHarness(
        harness_config={
            "profile": "ms_agent",
            "image": "img",
            "max_wall_seconds": 5,
        }
    )


def test_status_1_alone_is_agent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_harness_env(monkeypatch, _profile_mock(), flag_shim=False)
    docker = MagicMock()
    docker.containers.run.return_value = _ExitFailContainer()

    transcript = _harness().run_episode(ctx=_ctx(tmp_path, docker))

    assert transcript.termination_category == "agent_failure"


def test_inference_unavailable_elevates_to_endpoint_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_harness_env(monkeypatch, _profile_mock(), flag_shim=True)
    docker = MagicMock()
    docker.containers.run.return_value = _ExitFailContainer()

    transcript = _harness().run_episode(ctx=_ctx(tmp_path, docker))

    assert transcript.termination_category == "endpoint_unavailable"
    assert "inference endpoint unavailable" in transcript.termination_reason
    assert INFERENCE_UNAVAILABLE_PREFIX in transcript.termination_reason


def test_inference_timeout_is_its_own_category_not_endpoint_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A request timeout (decode starvation) must NOT be categorised as
    # endpoint_unavailable: the parallel worker loop quarantines the endpoint on
    # that category, and with a single endpoint that breaches the capacity floor
    # and aborts the whole run. It gets its own benign, non-quarantining
    # category instead.
    _patch_harness_env(monkeypatch, _profile_mock(), flag_shim=False, timeout_shim=True)
    docker = MagicMock()
    docker.containers.run.return_value = _ExitFailContainer()

    transcript = _harness().run_episode(ctx=_ctx(tmp_path, docker))

    assert transcript.termination_category == "inference_timeout"
    assert "inference request timed out" in transcript.termination_reason
    assert transcript.termination_category != "endpoint_unavailable"


def test_graceful_agent_failure_overrides_prior_sticky_budget_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = "graceful_agent_failure: length retries exhausted"
    _patch_harness_env(
        monkeypatch,
        _profile_mock(),
        flag_shim=False,
        graceful_agent_failure_detail=detail,
    )
    docker = MagicMock()
    docker.containers.run.return_value = _ExitFailContainer()
    ctx = _ctx(tmp_path, docker)
    ctx.budget_ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=0))
    rejection = ctx.budget_ledger.try_consume_tool_calls(
        [ToolCallRequest(capability_name="submit_hypothesis")]
    )
    assert rejection.accepted is False
    assert ctx.budget_ledger.exhausted() is not None

    transcript = _harness().run_episode(ctx=ctx)

    assert transcript.termination_category == "agent_failure"
    assert detail in transcript.termination_reason
    assert transcript.extra["budget_exhausted_kind"] == "task_tool_calls"
