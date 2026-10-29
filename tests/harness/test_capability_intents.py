from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from result import Ok

from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import Capability, TaskPromptSpec, Variation


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


def _make_inner(response_by_phase: dict[str, str]) -> MagicMock:
    inner = MagicMock()

    def _complete(messages):
        phase = next(
            (m.get("meta", {}).get("phase") for m in messages if m.get("meta")),
            "orchestrator",
        )
        result = MagicMock()
        result.content = response_by_phase.get(phase, "Findings: ok")
        result.usage = None
        return Ok(result)

    inner.plist_completion.side_effect = _complete
    inner.collector = None
    return inner


def _settings_with_modes(modes: dict) -> dict:
    return {
        "max_harness_iterations": 1,
        "scratchpad_max_chars": 5000,
        "orchestrator_prompt": (
            "{scratchpad_content}\n{budget_remaining}/{total_budget}"
        ),
        "modes": modes,
    }


def test_orchestrator_reads_scratchpad_dispositions_from_modes(
    tmp_path: Path,
) -> None:
    """Mode-config dispositions populate the scratchpad. ``annotations``
    on the capability is no longer read by the harness."""
    cap = Capability(name="run_python", description="exec")
    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=[cap],
    )
    recorder = EventRecorder(
        episode_id="ep-1",
        output_path=tmp_path / "events.jsonl",
    )
    traced = TracedGenner(
        inner=_make_inner(
            {
                "orchestrator": "MODE: analyzer\nINSTRUCTION: go\nREASONING: x",
                "analyzer": "Findings: drained via flash()\n",
            }
        ),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    ctx = HarnessContext(
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
            harness_settings=_settings_with_modes(
                {
                    "analyzer": {
                        "prompt": "{instruction}\n{scratchpad_content}",
                        "writes_scratchpad": True,
                        "scratchpad_label": "Findings",
                    }
                }
            ),
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        harness_session={"scratchpad": CrossEpisodeScratchpad(max_chars=5000)},
    )
    harness = OrchestratorModeHarness({})
    harness.run_episode(ctx=ctx)
    scratchpad = ctx.harness_session["scratchpad"].get_content()
    assert "drained via flash" in scratchpad
    assert "[Analyzer]" in scratchpad


def test_orchestrator_fails_loud_when_no_mode_writes_scratchpad(
    tmp_path: Path,
) -> None:
    """At least one mode must opt into scratchpad writes — otherwise the
    episode silently runs empty."""
    from src.harness.base import HarnessError

    cap = Capability(name="run_python", description="exec")
    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=[cap],
    )
    recorder = EventRecorder(
        episode_id="ep-1",
        output_path=tmp_path / "events.jsonl",
    )
    traced = TracedGenner(
        inner=_make_inner(
            {
                "orchestrator": "MODE: analyzer\nINSTRUCTION: go\nREASONING: x",
                "analyzer": "Findings: ok",
            }
        ),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    ctx = HarnessContext(
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
            harness_settings=_settings_with_modes(
                {
                    "analyzer": {
                        "prompt": "{instruction}\n{scratchpad_content}",
                        "writes_scratchpad": False,
                    }
                }
            ),
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
        harness_session={"scratchpad": CrossEpisodeScratchpad(max_chars=5000)},
    )
    harness = OrchestratorModeHarness({})
    with pytest.raises(HarnessError, match="scratchpad"):
        harness.run_episode(ctx=ctx)
