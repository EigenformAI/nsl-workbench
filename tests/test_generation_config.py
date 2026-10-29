from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.typing.config import AppConfig


@pytest.mark.parametrize(
    "field",
    [
        "target_training_rows",
        "max_episodes",
        "max_bootstrap_episodes",
        "parallel_episodes",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_generation_workload_limits_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        AppConfig.GenerationConfig(**{field: value})


def test_optional_bootstrap_limit_accepts_none() -> None:
    config = AppConfig.GenerationConfig(max_bootstrap_episodes=None)

    assert config.max_bootstrap_episodes is None


def test_zero_disable_intervals_remain_allowed() -> None:
    config = AppConfig.GenerationConfig(
        container_restart_interval=0,
        container_rebuild_interval=0,
        resource_snapshot_interval_episodes=0,
    )

    assert config.container_restart_interval == 0
    assert config.container_rebuild_interval == 0
    assert config.resource_snapshot_interval_episodes == 0
