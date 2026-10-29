from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.harness.container import ContainerHarness
from src.task.types import Workflow, WorkflowStep
from tests.harness.test_container_harness_cleanup import _FakeContainer, _Spy, _ctx


def _profile_mock(transcript: dict) -> MagicMock:
    profile = MagicMock()
    profile.supports_native_workflow.return_value = True
    profile.render_query.return_value = "q"
    profile.default_args.return_value = ["python", "/opt/nsl/run.py"]
    profile.env.return_value = {}
    profile.read_transcript.return_value = transcript
    profile.count_llm_turns.return_value = 1
    profile.to_artifacts.return_value = MagicMock(
        capability_invocations=[], capability_results=[], final_response="done"
    )
    return profile


def _run_with_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: MagicMock,
    workflow: Workflow,
):
    import src.harness.container as container_mod

    spy = _Spy()
    docker = MagicMock()
    docker.containers.run.return_value = _FakeContainer(spy)
    monkeypatch.setattr(
        container_mod,
        "_serve_host_shim",
        lambda app: spy.make_shim_handle(),
    )
    monkeypatch.setattr(
        container_mod.CapabilityMcpBridge,
        "serve_http",
        lambda self: spy.make_bridge_handle(),
    )
    monkeypatch.setattr(container_mod, "resolve_profile", lambda name, cfg: profile)
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
    harness = ContainerHarness(
        harness_config={"profile": "ms_agent", "image": "img", "max_wall_seconds": 10}
    )
    ctx = replace(_ctx(tmp_path, docker), workflow=workflow)

    result = harness.run_episode(ctx=ctx)

    return result, ctx


def test_no_call_on_tool_capable_step_bumps_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(name="explore_container", prompt="Explore", next_steps=("plan",)),
            WorkflowStep(
                name="plan",
                prompt="Plan",
                inherit_all_capabilities=False,
                capabilities=(),
            ),
        )
    )
    transcript = {
        "messages": [{"role": "assistant", "content": "I would inspect files."}],
        "workflow": {
            "explore_container": {
                "messages": [
                    {"role": "assistant", "content": "I would inspect files."}
                ],
                "tool_end_count": 0,
                "first_assistant_content": "I would inspect files.",
            },
            "plan": {
                "messages": [{"role": "assistant", "content": "Plan only."}],
                "tool_end_count": 0,
                "first_assistant_content": "Plan only.",
            },
        },
    }
    profile = _profile_mock(transcript)
    profile.tool_capable_step_names.return_value = {"explore_container"}

    _result, ctx = _run_with_profile(
        tmp_path, monkeypatch, profile=profile, workflow=workflow
    )

    assert ctx.recorder.snapshot_counters()["tool_calling_no_calls"] == 1
    event = next(
        event
        for event in ctx.recorder.events
        if event.category == "tool_calling_no_calls"
    )
    assert event.payload == {
        "step": "explore_container",
        "first_chars": "I would inspect files.",
    }


def test_no_call_on_reasoning_only_step_does_not_bump_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="plan",
                prompt="Plan",
                inherit_all_capabilities=False,
                capabilities=(),
                next_steps=("execute",),
            ),
            WorkflowStep(
                name="execute",
                prompt="Execute",
                inherit_all_capabilities=False,
                capabilities=("run_python",),
            ),
        )
    )
    transcript = {
        "messages": [
            {"role": "assistant", "content": "Plan only."},
            {"role": "tool", "name": "capabilities__run_python", "content": "ok"},
        ],
        "workflow": {
            "plan": {
                "messages": [{"role": "assistant", "content": "Plan only."}],
                "tool_end_count": 0,
                "first_assistant_content": "Plan only.",
            },
            "execute": {
                "messages": [
                    {
                        "role": "tool",
                        "name": "capabilities__run_python",
                        "content": "ok",
                    }
                ],
                "tool_end_count": 1,
                "first_assistant_content": None,
            },
        },
    }
    profile = _profile_mock(transcript)
    profile.tool_capable_step_names.return_value = {"execute"}

    _result, ctx = _run_with_profile(
        tmp_path, monkeypatch, profile=profile, workflow=workflow
    )

    assert "tool_calling_no_calls" not in ctx.recorder.snapshot_counters()
    assert not any(
        event.category == "tool_calling_no_calls" for event in ctx.recorder.events
    )
