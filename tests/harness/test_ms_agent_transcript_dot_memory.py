"""``MsAgentProfile.read_transcript`` resolves ``.memory/<tag>.json`` (hidden).

A prior design draft assumed ``memory/`` — verified wrong against
``ms_agent/utils/utils.py:save_history`` which writes to the hidden
``.memory/`` directory. This test pins the correct path and regression-
guards the negative case.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from src.harness.profiles.ms_agent import MsAgentProfile, MsAgentProfileConfig


def _profile(tag: str = "episode") -> MsAgentProfile:
    return MsAgentProfile(
        MsAgentProfileConfig(model="claude-sonnet-4-6", transcript_tag=tag)
    )


def test_transcript_reads_from_dot_memory(tmp_path: Path) -> None:
    (tmp_path / "output" / ".memory").mkdir(parents=True)
    (tmp_path / "output" / ".memory" / "episode.json").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "hi"}]})
    )
    transcript = _profile().read_transcript(tmp_path)
    assert transcript is not None
    assert transcript["messages"][0]["content"] == "hi"


@pytest.mark.parametrize(
    "create_non_hidden",
    [
        pytest.param(False, id="absent"),
        pytest.param(True, id="non-hidden-only"),
    ],
)
def test_transcript_ignores_non_hidden_memory_directory(
    tmp_path: Path, create_non_hidden: bool
) -> None:
    """A file in ``memory/`` (not ``.memory/``) must not be picked up —
    ms-agent uses the hidden path exclusively."""
    if create_non_hidden:
        (tmp_path / "output" / "memory").mkdir(parents=True)
        (tmp_path / "output" / "memory" / "episode.json").write_text(
            json.dumps(
                {"messages": [{"role": "assistant", "content": "wrong path"}]}
            )
        )
    # No .memory/ directory exists.
    assert _profile().read_transcript(tmp_path) is None


def test_transcript_honors_transcript_tag(tmp_path: Path) -> None:
    (tmp_path / "output" / ".memory").mkdir(parents=True)
    (tmp_path / "output" / ".memory" / "custom_tag.json").write_text(
        json.dumps({"messages": []})
    )
    profile = _profile(tag="custom_tag")
    assert profile.read_transcript(tmp_path) == {"messages": []}


def test_transcript_falls_back_to_single_memory_json_and_normalizes_list(
    tmp_path: Path,
) -> None:
    (tmp_path / "output" / ".memory").mkdir(parents=True)
    (tmp_path / "output" / ".memory" / "Agent-default.json").write_text(
        json.dumps(
            [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "hi"},
            ]
        )
    )
    transcript = _profile(tag="episode").read_transcript(tmp_path)
    assert transcript == {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "hi"},
        ]
    }


def test_transcript_aggregates_workflow_step_memories(tmp_path: Path) -> None:
    (tmp_path / "output" / ".memory").mkdir(parents=True)
    (tmp_path / "workflow.yaml").write_text(
        yaml.safe_dump(
            {
                "plan": {"agent_config": "plan.yaml", "next": ["act"]},
                "act": {"agent_config": "act.yaml"},
            }
        )
    )
    (tmp_path / "output" / ".memory" / "plan.json").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "planned"}]})
    )
    (tmp_path / "output" / ".memory" / "act.json").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "acted"}]})
    )

    transcript = _profile(tag="episode").read_transcript(tmp_path)

    assert transcript == {
        "messages": [
            {"role": "assistant", "content": "planned"},
            {"role": "assistant", "content": "acted"},
        ],
        "workflow": {
            "plan": {"messages": [{"role": "assistant", "content": "planned"}]},
            "act": {"messages": [{"role": "assistant", "content": "acted"}]},
        },
        "last_workflow_step": "act",
    }


def test_transcript_raises_when_workflow_memory_is_empty(tmp_path: Path) -> None:
    (tmp_path / "output" / ".memory").mkdir(parents=True)
    (tmp_path / "workflow.yaml").write_text(
        yaml.safe_dump(
            {"plan": {"agent_config": "plan.yaml", "context_mode": "inherit"}}
        )
    )

    with pytest.raises(RuntimeError, match="ms-agent workflow transcript missing"):
        _profile(tag="episode").read_transcript(tmp_path)
