from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.harness.base import HarnessError
from src.harness.container import _validate_no_tool_retry_supported
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import (
    Capability,
    EpisodeConstraints,
    NoToolReplyPolicy,
    StepConstraints,
    TaskPromptSpec,
    Variation,
    Workflow,
    WorkflowStep,
)


def _ctx(tmp_path: Path, constraints: EpisodeConstraints) -> HarnessContext:
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
        constraints=constraints,
    )


def test_profile_without_no_tool_retry_support_rejects_retry_policy(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        tmp_path,
        EpisodeConstraints(no_tool_reply=NoToolReplyPolicy(retry=True, max_retries=1)),
    )
    profile = MagicMock(supports_no_tool_retry=False)

    with pytest.raises(HarnessError, match="does not support no-tool retry"):
        _validate_no_tool_retry_supported(ctx, profile, profile_name="stub")


def test_zero_capability_step_auto_degrades_retry_policy(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        EpisodeConstraints(
            no_tool_reply=NoToolReplyPolicy(retry=False),
            step_overrides={
                "plan": StepConstraints(
                    no_tool_reply=NoToolReplyPolicy(retry=True, max_retries=1)
                )
            },
        ),
    )
    workflow = Workflow(
        steps=(
            WorkflowStep(
                name="plan",
                prompt="plan",
                inherit_all_capabilities=False,
                capabilities=(),
            ),
        )
    )
    profile = MagicMock(supports_no_tool_retry=False)

    _validate_no_tool_retry_supported(
        ctx,
        profile,
        profile_name="stub",
        workflow=workflow,
    )

    projected = ctx.with_capability_allowlist(set()).with_step_constraints("plan")
    assert projected.constraints is not None
    assert projected.constraints.no_tool_reply.retry is False
