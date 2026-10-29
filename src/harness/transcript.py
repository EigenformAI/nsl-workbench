"""HarnessTranscript + TrajectoryRecord data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.task.types import EpisodeArtifacts


# "success" means the agent process exited cleanly (exit code 0), NOT that
# the task objective was achieved.  Task success is determined by the reward
# computed downstream (EpisodeOutcome.success / reward.success).
TerminationCategory = Literal[
    "success",
    "agent_failure",
    "harness_error",
    "context_overflow",
    "repetition_collapse",
    "budget_exhausted",
    "wall_clock",
    # Backend faults (not the agent's): a true outage that quarantines the
    # endpoint, vs a request timeout that is benign and retryable (must NOT
    # quarantine — a single timeout would breach the capacity floor).
    "endpoint_unavailable",
    "inference_timeout",
]


@dataclass
class HarnessTranscript:
    """What the harness reports back to the framework after ``run_episode``.

    NOT the source of truth for training data — that lives in
    ``EventRecorder.inference_records``, owned by the framework. This
    transcript carries only what ``task.finalize_episode`` and reward-side
    debugging need.
    """

    artifacts: EpisodeArtifacts
    llm_turns: int
    termination_reason: str  # free text; harness-chosen for debugging
    termination_category: TerminationCategory
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepTranscript(HarnessTranscript):
    """Transcript returned by framework-driven workflow step dispatch."""
