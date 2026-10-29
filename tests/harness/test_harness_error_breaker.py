"""Consecutive-HarnessError circuit breaker — behavior + config contract.

A single HarnessError does not trip the breaker (noise is tolerated); N
consecutive ones do, with a distinct alarm distinguishing systematic
harness breakage from environment flake or one-off agent failure.

The decision itself lives in ``src.execution.generation.
update_harness_error_breaker`` as a pure function so the generation loop
and this test can share the same implementation.
"""

from __future__ import annotations

from src.execution.generation import update_harness_error_breaker
from src.typing.config import HarnessConfig


def test_default_limit_is_three():
    config = HarnessConfig()
    assert config.consecutive_harness_error_limit == 3


def test_limit_is_configurable():
    config = HarnessConfig(consecutive_harness_error_limit=5)
    assert config.consecutive_harness_error_limit == 5


def test_single_harness_error_does_not_trip():
    count, trip = update_harness_error_breaker(
        error_category="harness_error", consecutive=0, limit=3
    )
    assert count == 1
    assert trip is False


def test_trips_exactly_at_limit():
    # Second — still below limit.
    count, trip = update_harness_error_breaker(
        error_category="harness_error", consecutive=1, limit=3
    )
    assert count == 2 and trip is False
    # Third — breaker trips.
    count, trip = update_harness_error_breaker(
        error_category="harness_error", consecutive=2, limit=3
    )
    assert count == 3 and trip is True


def test_non_harness_error_resets_counter():
    count, trip = update_harness_error_breaker(
        error_category=None, consecutive=2, limit=3
    )
    assert count == 0 and trip is False
    count, trip = update_harness_error_breaker(
        error_category="wall_clock", consecutive=2, limit=3
    )
    assert count == 0 and trip is False


def test_zero_limit_disables_breaker():
    count, trip = update_harness_error_breaker(
        error_category="harness_error", consecutive=99, limit=0
    )
    assert count == 100 and trip is False


def test_sequence_simulation():
    """Simulate 5 episodes: ok, err, err, ok, err — breaker must not trip
    because the intervening ok resets."""
    counts: list[int] = []
    trips: list[bool] = []
    consecutive = 0
    for cat in ["ok", "harness_error", "harness_error", "ok", "harness_error"]:
        consecutive, trip = update_harness_error_breaker(
            error_category=cat, consecutive=consecutive, limit=3
        )
        counts.append(consecutive)
        trips.append(trip)
    assert counts == [0, 1, 2, 0, 1]
    assert not any(trips)
