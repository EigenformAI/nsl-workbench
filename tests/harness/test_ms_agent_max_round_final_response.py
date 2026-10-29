"""``MsAgentProfile.to_artifacts`` handles max-round cutoffs gracefully.

When ms-agent hits ``max_chat_round``, it appends a synthetic trailing
assistant message (empty or marked as cutoff). The profile's final-response
extractor must skip that and fall back to the last content-bearing
assistant message.
"""

from __future__ import annotations

from src.harness.profiles.ms_agent import MsAgentProfile, MsAgentProfileConfig


def _profile() -> MsAgentProfile:
    return MsAgentProfile(MsAgentProfileConfig(model="claude-sonnet-4-6"))


def test_to_artifacts_prefers_last_content_bearing_assistant() -> None:
    """Last assistant turn is an empty cutoff — skip back to the prior one."""
    transcript = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "intermediate thought"},
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "real final answer"},
            # Cutoff synthetic trailer — empty content.
            {"role": "assistant", "content": ""},
        ],
    }
    artifacts = _profile().to_artifacts(transcript=transcript, capability_pairs=[])
    assert artifacts.final_response == "real final answer"


def test_count_llm_turns_ignores_trailing_empty_cutoff_assistant() -> None:
    transcript = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "tool call"},
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "final answer"},
            {"role": "assistant", "content": ""},
        ],
    }
    assert _profile().count_llm_turns(transcript) == 2
