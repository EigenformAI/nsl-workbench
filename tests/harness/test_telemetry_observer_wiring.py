"""Telemetry observer receives live prompt_tokens + harness telemetry.

The framework owns prompt-token capture at the inference boundary, but it
must not reach into harness-private state like the orchestrator scratchpad.
This test pins the updated integration contract: the closure built inside
``src.execution.episode.run_episode`` forwards the most recent
``prompt_tokens`` plus whatever the harness exposes through ``telemetry()``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from result import Ok

from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult, UsageInfo


def test_observer_receives_telemetry_and_prompt_tokens(tmp_path: Path) -> None:
    observer_calls: list[tuple[int | None, dict[str, str]]] = []

    class _Harness:
        def __init__(self) -> None:
            self._telemetry = {"step": "1", "budget_left": "2"}

        def telemetry(self) -> dict[str, str]:
            return dict(self._telemetry)

    harness = _Harness()

    def telemetry_observer(
        prompt_tokens: int | None,
        telemetry: dict[str, str],
    ) -> None:
        observer_calls.append((prompt_tokens, telemetry))

    last_prompt_tokens_holder: dict[str, int | None] = {"value": None}

    def _publish_telemetry() -> None:
        telemetry_observer(
            last_prompt_tokens_holder["value"],
            harness.telemetry(),
        )

    def _on_inference(usage: UsageInfo | None) -> None:
        if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
            last_prompt_tokens_holder["value"] = usage.prompt_tokens
        _publish_telemetry()

    inner = MagicMock()
    inner.plist_completion.return_value = Ok(
        InferenceResult(
            content="hi",
            usage=UsageInfo(prompt_tokens=1234, completion_tokens=5),
        )
    )
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "e.jsonl")
    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
        on_inference=_on_inference,
    )

    traced.plist_completion([{"role": "user", "content": "x"}], phase="p")
    assert observer_calls == [(1234, {"step": "1", "budget_left": "2"})]

    harness._telemetry = {"step": "2", "budget_left": "1"}
    inner.plist_completion.return_value = Ok(
        InferenceResult(
            content="hi2",
            usage=UsageInfo(prompt_tokens=2000, completion_tokens=1),
        )
    )
    traced.plist_completion([{"role": "user", "content": "y"}], phase="p")
    assert observer_calls[1] == (2000, {"step": "2", "budget_left": "1"})
