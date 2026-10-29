"""Capability ``runs_code`` is a harness-neutral hint flag.

Mode dispositions (writes_scratchpad, scratchpad_label, prompt) live on
``OrchestratorModesConfig.modes`` and are exercised by
``test_capability_intents.py`` — those harness behaviours are not
reproduced here.
"""

from __future__ import annotations

from src.task.types import Capability


def test_runs_code_intent_flag_lives_on_capability() -> None:
    """``Capability.runs_code`` remains a harness-neutral hint about what
    a capability does. Modes (harness-side) carry their own ``runs_code``
    flag for the dispatch decision."""
    cap = Capability(name="run_python", description="exec", runs_code=True)
    assert cap.runs_code is True
    other = Capability(name="recorder", description="notes")
    assert other.runs_code is False
