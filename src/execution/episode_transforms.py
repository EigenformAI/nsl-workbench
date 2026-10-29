from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _value(payload: Any, name: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def categorize_termination(
    transcript: Any,
) -> tuple[bool, str | None, str | None]:
    category = _value(transcript, "termination_category")
    reason = _value(transcript, "termination_reason")
    if category in {
        "context_overflow",
        "inference_timeout",
        "repetition_collapse",
        "wall_clock",
        "harness_error",
        "endpoint_unavailable",
    }:
        return True, str(category), None if reason is None else str(reason)
    return False, None, None


def to_trajectory(
    *,
    episode_id: str,
    llm_turns: int,
    termination_reason: str | None,
    termination_category: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "llm_turns": llm_turns,
        "termination_reason": termination_reason,
        "termination_category": termination_category,
        "extra": dict(extra),
    }


def to_prompt_responses(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": row["prompt"],
            "raw_response": row["raw_response"],
            "timestamp": row["timestamp"],
            "interaction_type": row["interaction_type"],
            "success": row["success"],
            "error_message": row["error_message"],
        }
        for row in train_rows
    ]


__all__ = [
    "categorize_termination",
    "to_prompt_responses",
    "to_trajectory",
]
