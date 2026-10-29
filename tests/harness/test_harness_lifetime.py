"""Harness instances are per-episode and must not share mutable state
across parallel episodes.

Two guarantees:

1. Consecutive ``construct_harness`` calls return distinct instances.
2. Running two harness instances concurrently on the same task spec
   produces independent budget/scratchpad/recorder state — no bleed
   through shared mutable attributes.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from result import Ok

from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.loader import construct_harness
from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import Capability, TaskPromptSpec, Variation
from src.typing.config import HarnessConfig, OrchestratorModesConfig


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


def _make_inner(response_by_phase):
    inner = MagicMock()

    def _complete(messages):
        phase = next(
            (m.get("meta", {}).get("phase") for m in messages if m.get("meta")),
            "orchestrator",
        )
        result = MagicMock()
        result.content = response_by_phase.get(phase, "Results: ok")
        result.usage = None
        return Ok(result)

    inner.plist_completion.side_effect = _complete
    inner.collector = None
    return inner


def _prompt_spec():
    return TaskPromptSpec(
        system_instruction="sys",
        capabilities=[Capability(name="run_python", description="exec")],
    )


# Phase 2: HarnessConfig requires a populated section matching `name`.
# Per-episode harness construction still returns distinct instances — the
# config is shared, the instances are not.
_VALID_CONFIG = HarnessConfig(
    orchestrator_modes=OrchestratorModesConfig(
        orchestrator_prompt="{scratchpad_content}"
    ),
)


def test_two_constructions_yield_distinct_instances():
    a = construct_harness(_VALID_CONFIG)
    b = construct_harness(_VALID_CONFIG)
    assert a is not b
    assert type(a) is type(b)


def test_parallel_episodes_have_disjoint_state(tmp_path: Path) -> None:
    """Two harness instances on one task spec must not share mutable state."""
    shared_task = _StubTask()
    response_by_phase = {
        "orchestrator": "MODE: explorer\nINSTRUCTION: go\nREASONING: tick",
        "explorer": "Results: fine",
    }

    def _run(slot_id: int, budget: int, out: dict) -> None:
        recorder = EventRecorder(
            episode_id=f"ep-{slot_id}",
            output_path=tmp_path / f"events_{slot_id}.jsonl",
        )
        traced = TracedGenner(
            inner=_make_inner(response_by_phase),
            recorder=recorder,
            cancel_event=threading.Event(),
            episode_id=f"ep-{slot_id}",
        )
        scratchpad = CrossEpisodeScratchpad(max_chars=5000)
        scratchpad.append(f"seed-{slot_id}")
        ctx = HarnessContext(
            episode_id=f"ep-{slot_id}",
            genner=traced,
            task=shared_task,  # type: ignore[arg-type]
            variation=Variation(name="v", description="d"),
            prompt_spec=_prompt_spec(),
            episode_context={},
            containers=[],
            agent_container=None,  # type: ignore[arg-type]
            host_cache_folder=tmp_path,
            config=HarnessConfigView(
                harness_settings={
                    "max_harness_iterations": budget,
                    "scratchpad_max_chars": 5000,
                    "tool_output_max_chars": 0,
                    "orchestrator_prompt": "{scratchpad_content}\n{budget_remaining}/{total_budget}",
                    "modes": {
                        "explorer": {
                            "prompt": "{instruction}\n{scratchpad_content}",
                            "timeout_s": 30,
                            "writes_scratchpad": True,
                            "scratchpad_label": "Results",
                        }
                    },
                },
                model_name="test",
                train_data_save_folder=str(tmp_path),
                code_host_cache_path=str(tmp_path),
            ),
            metrics=None,  # type: ignore[arg-type]
            recorder=recorder,
            cancel_event=threading.Event(),
            harness_session={"scratchpad": scratchpad},
        )
        # Fresh harness per episode — this is what run_episode does.
        harness = OrchestratorModeHarness({})
        transcript = harness.run_episode(ctx=ctx)
        out["transcript"] = transcript
        out["inference_count"] = len(recorder.inference_records)
        out["scratchpad"] = scratchpad.get_content()

    out_a: dict = {}
    out_b: dict = {}
    threads = [
        threading.Thread(target=_run, args=(0, 2, out_a)),
        threading.Thread(target=_run, args=(1, 4, out_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert out_a["transcript"].extra["harness_iterations_count"] == 2
    assert out_b["transcript"].extra["harness_iterations_count"] == 4
    # Each slot sees only its own seed (no cross-slot bleed).
    assert "seed-0" in out_a["scratchpad"]
    assert "seed-1" not in out_a["scratchpad"]
    assert "seed-1" in out_b["scratchpad"]
    assert "seed-0" not in out_b["scratchpad"]
    # Inference counts reflect each slot's own budget.
    assert out_a["inference_count"] == 2 * 2
    assert out_b["inference_count"] == 2 * 4
