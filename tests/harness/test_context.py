from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import (
    CapabilityInvocation,
    CapabilityResult,
    TaskPromptSpec,
    Variation,
)


class _Task:
    metric_name = "dummy"
    metric_unit = ""
    name = "stub"
    description = "stub"

    def __init__(self, *, raise_on_call: Exception | None = None) -> None:
        self.calls: list[CapabilityInvocation] = []
        self.contexts: list[object] = []
        self.container_calls: list[list[object]] = []
        self._raise_on_call = raise_on_call

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers,
        variation,
        ctx,
    ) -> CapabilityResult:
        self.calls.append(invocation)
        self.contexts.append(ctx)
        self.container_calls.append(containers)
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return CapabilityResult(name=invocation.name, output={"ok": True})


def _ctx(tmp_path: Path, *, task: _Task | None = None) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    agent_container = MagicMock(id="agent")
    traced = TracedGenner(
        inner=MagicMock(),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=task or _Task(),  # type: ignore[arg-type]
        variation=Variation(name="v", description="d"),
        prompt_spec=TaskPromptSpec(system_instruction="sys"),
        episode_context={},
        containers=[agent_container, MagicMock(id="sibling")],
        agent_container=agent_container,
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


def test_execute_capability_counts_and_sets_last_tool(tmp_path: Path) -> None:
    task = _Task()
    ctx = _ctx(tmp_path, task=task)
    invocation = CapabilityInvocation(name="analyzer", input={"x": 1})

    result = ctx.execute_capability(invocation)

    assert result.name == "analyzer"
    assert task.calls == [invocation]
    assert task.container_calls == [ctx.containers]
    assert ctx.recorder.snapshot_counters()["tool_calls"] == 1
    assert ctx.recorder.snapshot_labels()["last_tool"] == "analyzer"


def test_execute_capability_forwards_live_recorder(tmp_path: Path) -> None:
    task = _Task()
    ctx = _ctx(tmp_path, task=task)

    ctx.execute_capability(CapabilityInvocation(name="analyzer", input={}))

    assert task.contexts[0].recorder is ctx.recorder


def test_execute_capability_counts_even_when_task_raises(tmp_path: Path) -> None:
    task = _Task(raise_on_call=RuntimeError("boom"))
    ctx = _ctx(tmp_path, task=task)
    invocation = CapabilityInvocation(name="analyzer", input={})

    with pytest.raises(RuntimeError, match="boom"):
        ctx.execute_capability(invocation)

    assert task.calls == [invocation]
    assert ctx.recorder.snapshot_counters()["tool_calls"] == 1
    assert ctx.recorder.snapshot_labels()["last_tool"] == "analyzer"
