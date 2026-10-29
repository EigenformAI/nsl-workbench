from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.harness.container import ContainerHarness
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import (
    BudgetConstraints,
    Capability,
    EpisodeConstraints,
    StepConstraints,
    TaskPromptSpec,
    Variation,
    Workflow,
    WorkflowStep,
)


def _ctx(tmp_path: Path) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    return HarnessContext(
        episode_id="ep-1",
        genner=TracedGenner(
            inner=MagicMock(),
            recorder=recorder,
            cancel_event=threading.Event(),
            episode_id="ep-1",
        ),
        task=MagicMock(),
        variation=Variation(name="v", description="d"),
        prompt_spec=TaskPromptSpec(
            system_instruction="sys",
            capabilities=[Capability(name="run_python", description="run")],
        ),
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
        metrics=None,
        recorder=recorder,
        cancel_event=threading.Event(),
    )


def _harness(monkeypatch: pytest.MonkeyPatch, profile: MagicMock) -> ContainerHarness:
    import src.harness.container as container_mod

    monkeypatch.setattr(container_mod, "resolve_profile", lambda name, cfg: profile)
    return ContainerHarness(
        harness_config={"profile": "ms_agent", "image": "img", "max_wall_seconds": 10}
    )


def test_run_workflow_uses_native_path_when_profile_supports_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = MagicMock()
    profile.supports_native_workflow.return_value = True
    harness = _harness(monkeypatch, profile)
    workflow = Workflow(steps=(WorkflowStep(name="s", prompt="p"),))
    ctx = _ctx(tmp_path)
    expected = MagicMock()

    def _run_episode(*, ctx: HarnessContext):
        assert ctx.workflow is workflow
        return expected

    monkeypatch.setattr(harness, "run_episode", _run_episode)

    assert harness.run_workflow(workflow, ctx) is expected
    profile.supports_native_workflow.assert_called_once_with(workflow)
    assert ctx.workflow is None


def test_run_workflow_falls_back_to_driver_when_profile_does_not_support_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = MagicMock()
    profile.supports_native_workflow.return_value = False
    harness = _harness(monkeypatch, profile)
    workflow = Workflow(steps=(WorkflowStep(name="s", prompt="p"),))
    ctx = _ctx(tmp_path)
    expected = MagicMock()

    monkeypatch.setattr(
        "src.harness.workflow_driver.WorkflowDriver.run",
        lambda self, wf, run_ctx: expected if wf is workflow and run_ctx is ctx else None,
    )
    monkeypatch.setattr(
        harness,
        "run_episode",
        lambda *, ctx: (_ for _ in ()).throw(AssertionError("native path used")),
    )

    assert harness.run_workflow(workflow, ctx) is expected


def test_run_episode_passes_native_workflow_to_render_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.test_container_harness_cleanup import _FakeContainer, _Spy

    import src.harness.container as container_mod

    spy = _Spy()
    profile = MagicMock()
    profile.supports_native_workflow.return_value = True
    profile.render_query.return_value = "q"
    profile.default_args.return_value = ["python", "/opt/nsl/run.py"]
    profile.env.return_value = {}
    profile.read_transcript.return_value = {"messages": []}
    profile.count_llm_turns.return_value = 0
    profile.to_artifacts.return_value = MagicMock(
        capability_invocations=[], capability_results=[], final_response=None
    )
    monkeypatch.setattr(container_mod, "resolve_profile", lambda name, cfg: profile)
    monkeypatch.setattr(
        container_mod, "_serve_host_shim", lambda app: spy.make_shim_handle()
    )
    monkeypatch.setattr(
        container_mod.CapabilityMcpBridge,
        "serve_http",
        lambda self: spy.make_bridge_handle(),
    )
    monkeypatch.setattr(
        container_mod,
        "_docker_host_gateway",
        lambda client, network_mode=None: "127.0.0.1",
    )
    monkeypatch.setattr(
        container_mod,
        "_wait_with_cancel",
        lambda c, ev, max_wall: MagicMock(reason="exited", category="success"),
    )
    docker = MagicMock()
    docker.containers.run.return_value = _FakeContainer(spy)
    harness = ContainerHarness(
        harness_config={"profile": "ms_agent", "image": "img", "max_wall_seconds": 10}
    )
    workflow = Workflow(steps=(WorkflowStep(name="s", prompt="p"),))
    constraints = EpisodeConstraints(
        step_overrides={"s": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=1))}
    )

    harness.run_episode(
        ctx=replace(
            _ctx(tmp_path),
            docker_client=docker,
            workflow=workflow,
            constraints=constraints,
        )
    )

    assert profile.render_config.call_args.kwargs["workflow"] is workflow
    assert profile.render_config.call_args.kwargs["constraints"] is constraints


def test_ms_agent_native_predicate_accepts_isolated_and_on_error() -> None:
    from src.harness.profiles.ms_agent import MsAgentProfile, MsAgentProfileConfig

    profile = MsAgentProfile(MsAgentProfileConfig(model="m"))
    isolated = Workflow(
        steps=(WorkflowStep(name="s", prompt="p", context_mode="isolated"),)
    )
    recovery = Workflow(
        steps=(
            WorkflowStep(name="try", prompt="try", on_error="recover", is_entry=True),
            WorkflowStep(name="recover", prompt="recover"),
        )
    )

    assert profile.supports_native_workflow(isolated) is True
    assert profile.supports_native_workflow(recovery) is True
