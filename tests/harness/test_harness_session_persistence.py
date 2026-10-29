"""Regression: harness-managed session state must persist across episodes
within a worker slot.

The old framework-level field was ``cross_episode_scratchpad``. The new
surface is a generic ``harness_session`` dict that stays attached to the
slot and carries the orchestrator scratchpad under the harness-owned key.
"""

from __future__ import annotations

from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad


def test_harness_session_scratchpad_content_survives_between_uses() -> None:
    harness_session = {
        "scratchpad": CrossEpisodeScratchpad(max_chars=10000),
    }

    harness_session["scratchpad"].append("note from episode A", episode_id="ep-a")
    same_slot_session = harness_session
    same_slot_session["scratchpad"].append("note from episode B", episode_id="ep-b")

    content = same_slot_session["scratchpad"].get_content()
    assert "episode A" in content
    assert "episode B" in content
