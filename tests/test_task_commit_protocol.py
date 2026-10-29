from __future__ import annotations

from src.task.base import TaskSpec
from src.execution.generation_persistence import (
    GenerationIdentity,
    checkpoint_payload_identity,
    task_commit_intents,
    validate_resume_checkpoint,
)
from src.typing.trajectory import EpisodeTrajectory


def test_default_task_commit_protocol_is_opt_out() -> None:
    assert TaskSpec.episode_commit_enabled(None) is False
    assert TaskSpec.parallel_checkpoint_safe(None) is False


def test_commit_metadata_round_trips_in_episode_trajectory() -> None:
    episode = EpisodeTrajectory(
        episode_id="ep-1",
        generation_id=2,
        episode_index=3,
        prompt_responses=[],
        trajectory={},
        score=1.0,
        episode_runtime_success=True,
        success=True,
        llm_turns_count=1,
        container_variation="v1",
        started_at="start",
        completed_at="end",
        duration_seconds=1.0,
        task_commit_token="token",
        task_commit_digest="digest",
    )

    restored = EpisodeTrajectory.from_dict(episode.to_dict())

    assert restored.task_commit_token == "token"
    assert restored.task_commit_digest == "digest"


def test_journaled_task_intents_are_validated_for_recovery() -> None:
    episode = EpisodeTrajectory(
        episode_id="ep-1",
        generation_id=0,
        episode_index=0,
        prompt_responses=[],
        trajectory={},
        score=0.0,
        episode_runtime_success=False,
        success=False,
        llm_turns_count=0,
        container_variation="v1",
        started_at="start",
        completed_at="end",
        duration_seconds=0.0,
        task_commit_token="token",
        task_commit_digest="digest",
    )

    assert task_commit_intents([episode]) == {"token": "digest"}


def test_commit_safe_resume_allows_checkpoint_to_trail_durable_journal() -> None:
    identity = GenerationIdentity(
        generation_id=0,
        run_id="run",
        task_name="task",
        task_identity="tests.Task",
        export_recipe_hash="recipe",
        execution_mode="parallel",
    )
    checkpoint = {
        **checkpoint_payload_identity(identity),
        "committed_episode_indices": [0],
    }

    validate_resume_checkpoint(
        checkpoint=checkpoint,
        journal_indices={0, 1},
        identity=identity,
        allow_journal_ahead=True,
    )
