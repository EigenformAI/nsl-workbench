"""``ContainerHarness`` honors ``cancel_event`` at the wait boundary.

A harness container that sleeps forever must terminate within one tick of
``cancel_event.set()``; the transcript category lands on ``wall_clock``.

Documented limitation: a capability call that started before cancellation
will run to completion — ``execute_capability`` has no cooperative hook.
This test asserts that limitation explicitly so we do not regress it
silently into a hang.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.harness.container import ContainerHarness
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import Capability, TaskPromptSpec, Variation


class _SleepForever:
    """Fake Docker container whose ``wait`` blocks until externally killed."""

    def __init__(self) -> None:
        self.id = "slow"
        self._killed = threading.Event()
        self.removed = False

    def wait(self, **_kwargs):
        self._killed.wait(timeout=30)
        return {"StatusCode": 137}

    def reload(self) -> None: ...

    @property
    def status(self) -> str:
        return "running"

    def logs(self, **_kwargs) -> bytes:
        return b""

    def kill(self) -> None:
        self._killed.set()

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


def _ctx(
    tmp_path: Path, docker_client: Any, cancel_event: threading.Event
) -> HarnessContext:
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
        cancel_event=cancel_event,
        docker_client=docker_client,
    )


def _patch_harness_env(
    monkeypatch: pytest.MonkeyPatch,
    profile_mock: MagicMock,
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


def test_cancel_event_terminates_sleeping_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile_mock()
    _patch_harness_env(monkeypatch, profile)

    docker = MagicMock()
    fake = _SleepForever()
    docker.containers.run.return_value = fake
    cancel = threading.Event()
    ctx = _ctx(tmp_path, docker, cancel)

    harness = ContainerHarness(
        harness_config={
            "profile": "ms_agent",
            "image": "img",
            "max_wall_seconds": 30,
        }
    )

    # Fire cancel shortly after run_episode starts.
    def _cancel_soon() -> None:
        time.sleep(0.1)
        cancel.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()

    started = time.monotonic()
    transcript = harness.run_episode(ctx=ctx)
    elapsed = time.monotonic() - started

    # Should terminate within a second or so — not 30 seconds.
    assert elapsed < 5.0
    assert transcript.termination_category == "wall_clock"
    assert fake.removed is True
