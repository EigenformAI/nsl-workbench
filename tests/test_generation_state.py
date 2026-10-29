from __future__ import annotations

import random

import pytest

from src.execution.generation import select_variation_index
from src.execution import generation_state
from src.execution.generation_state import IntrinsicBudgetController
from src.typing.trajectory import EpisodeTrajectory


def _episode(
    index: int,
    *,
    bootstrap_active: bool = False,
    admitted: bool = False,
) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        episode_id=f"ep-{index}",
        generation_id=3,
        episode_index=index,
        prompt_responses=[],
        trajectory={},
        score=0.0,
        episode_runtime_success=False,
        success=False,
        llm_turns_count=0,
        container_variation="v0",
        started_at="2026-07-15T00:00:00",
        completed_at="2026-07-15T00:00:01",
        duration_seconds=1.0,
        task_breakdown={
            "bootstrap_active": bootstrap_active,
            "admitted": admitted,
        },
    )


def test_pre_admission_checks_regular_and_bootstrap_caps() -> None:
    regular = IntrinsicBudgetController(
        max_episodes=2,
        max_bootstrap_episodes=10,
        committed=[_episode(0), _episode(1)],
    )
    bootstrap = IntrinsicBudgetController(
        max_episodes=10,
        max_bootstrap_episodes=2,
        committed=[
            _episode(0, bootstrap_active=True),
            _episode(1, bootstrap_active=True),
        ],
    )

    assert regular.reserve() is None
    assert regular.stop_reason == "max_episodes"
    assert bootstrap.reserve() is None
    assert bootstrap.stop_reason == "max_bootstrap_episodes"


def test_budget_class_and_admission_are_intrinsic_and_order_independent() -> None:
    episodes = [
        _episode(5, bootstrap_active=True),
        _episode(2, admitted=True),
        _episode(8, bootstrap_active=True),
    ]
    controller = IntrinsicBudgetController(
        max_episodes=2,
        max_bootstrap_episodes=1,
        committed=list(reversed(episodes)),
    )

    assert controller.bootstrap_done == 2
    assert controller.regular_done == 1
    assert controller.has_admission
    assert controller.reserve() == 9


def test_committed_episode_accounting_centralizes_checkpoint_fields() -> None:
    episodes = [
        _episode(5, bootstrap_active=True),
        _episode(2, admitted=True),
        _episode(8, bootstrap_active=True),
    ]

    accounting = generation_state.committed_episode_accounting(episodes)

    assert accounting.committed_episode_indices == (2, 5, 8)
    assert accounting.next_episode_index == 9
    assert accounting.bootstrap_episodes_run == 2
    assert accounting.regular_episodes_run == 1
    assert accounting.has_admission is True


def test_parallel_bootstrap_reservations_stop_at_threshold_and_drain_bounded() -> None:
    controller = IntrinsicBudgetController(
        max_episodes=20,
        max_bootstrap_episodes=1,
        committed=[],
    )
    reservations = [controller.reserve() for _ in range(3)]
    assert reservations == [0, 1, 2]

    controller.commit(_episode(0, bootstrap_active=True))
    assert controller.reserve() is None
    controller.commit(_episode(1, bootstrap_active=True))
    controller.commit(_episode(2, bootstrap_active=True))

    assert controller.bootstrap_done == 3
    assert controller.in_flight == 0
    assert controller.bootstrap_done <= 1 + 3 - 1


def test_parallel_regular_reservations_stop_at_threshold_and_drain_bounded() -> None:
    controller = IntrinsicBudgetController(
        max_episodes=1,
        max_bootstrap_episodes=20,
        committed=[],
    )
    reservations = [controller.reserve() for _ in range(4)]
    assert reservations == [0, 1, 2, 3]

    controller.commit(_episode(0))
    assert controller.reserve() is None
    for episode_index in (1, 2, 3):
        controller.commit(_episode(episode_index))

    assert controller.regular_done == 4
    assert controller.in_flight == 0
    assert controller.regular_done <= 1 + 4 - 1


@pytest.mark.parametrize(
    ("max_episodes", "max_bootstrap_episodes", "reason"),
    [
        (0, 2, "max_episodes"),
        (2, 0, "max_bootstrap_episodes"),
    ],
)
def test_zero_pre_admission_cap_issues_nothing(
    max_episodes: int,
    max_bootstrap_episodes: int,
    reason: str,
) -> None:
    controller = IntrinsicBudgetController(
        max_episodes=max_episodes,
        max_bootstrap_episodes=max_bootstrap_episodes,
        committed=[],
    )

    assert controller.reserve() is None
    assert controller.stop_reason == reason


def test_late_bootstrap_commit_keeps_intrinsic_class_after_admission() -> None:
    controller = IntrinsicBudgetController(
        max_episodes=1,
        max_bootstrap_episodes=1,
        committed=[],
    )
    admitted_index = controller.reserve()
    bootstrap_index = controller.reserve()
    assert admitted_index == 0
    assert bootstrap_index == 1

    controller.commit(_episode(admitted_index, admitted=True))
    controller.commit(_episode(bootstrap_index, bootstrap_active=True))

    assert controller.regular_done == 1
    assert controller.bootstrap_done == 1
    assert controller.has_admission
    assert controller.reserve() is None
    assert controller.stop_reason == "max_episodes"


def test_seeded_resume_discards_one_draw_per_committed_episode() -> None:
    uninterrupted_rng = random.Random(731)
    uninterrupted = [
        select_variation_index("random", index, 7, rng=uninterrupted_rng)
        for index in range(6)
    ]
    resumed_rng = random.Random(731)
    for index in range(4):
        select_variation_index("random", index, 7, rng=resumed_rng)

    resumed = [
        select_variation_index("random", index, 7, rng=resumed_rng)
        for index in range(4, 6)
    ]

    assert uninterrupted[4:] == resumed
