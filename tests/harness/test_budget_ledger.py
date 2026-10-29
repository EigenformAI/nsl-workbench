"""Tests for BudgetLedger.

TDD: these tests are written before implementation and should fail until
BudgetLedger, BudgetSnapshot, BudgetExhaustion, ToolCallRequest, and
ToolCallReservation are added to src/harness/budget.py.
"""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.harness.budget import (
    BudgetExhaustion,
    BudgetLedger,
    BudgetSnapshot,
    ToolCallRequest,
    ToolCallReservation,
)
from src.task.types import BudgetConstraints


def _req(name: str) -> ToolCallRequest:
    return ToolCallRequest(capability_name=name)


def test_snapshot_on_empty_ledger() -> None:
    ledger = BudgetLedger(BudgetConstraints())
    snap = ledger.snapshot()
    assert isinstance(snap, BudgetSnapshot)
    assert snap.task_tool_calls_used == 0
    assert snap.llm_turns_used == 0
    assert snap.task_tool_calls_limit is None
    assert snap.llm_turns_limit is None


def test_record_llm_turn_increments_counter() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_llm_turns=10))
    snap = ledger.record_llm_turn()
    assert snap.llm_turns_used == 1
    ledger.record_llm_turn()
    assert ledger.snapshot().llm_turns_used == 2


def test_record_llm_turn_never_raises_even_past_limit() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_llm_turns=1))
    ledger.record_llm_turn()
    # advisory — should NOT raise even though limit is exceeded
    snap = ledger.record_llm_turn()
    assert snap.llm_turns_used == 2


def test_try_consume_accepts_within_global_limit() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    result = ledger.try_consume_tool_calls([_req("run"), _req("run")])
    assert isinstance(result, ToolCallReservation)
    assert result.accepted is True
    assert ledger.snapshot().task_tool_calls_used == 2


def test_try_consume_rejects_whole_batch_on_overflow() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=3))
    ledger.try_consume_tool_calls([_req("run"), _req("run")])  # used=2
    # This batch of 2 would push total to 4, exceeding limit of 3 — reject all
    result = ledger.try_consume_tool_calls([_req("run"), _req("run")])
    assert result.accepted is False
    assert result.rejection_reason is not None
    # Budget not consumed on rejection
    assert ledger.snapshot().task_tool_calls_used == 2


def test_try_consume_rejects_per_name_overflow() -> None:
    ledger = BudgetLedger(
        BudgetConstraints(
            max_task_tool_calls=20,
            max_task_tool_calls_by_name={"deploy": 2},
        )
    )
    ledger.try_consume_tool_calls([_req("deploy"), _req("deploy")])  # per-name used=2
    result = ledger.try_consume_tool_calls([_req("deploy")])
    assert result.accepted is False
    assert "deploy" in (result.rejection_reason or "")


def test_per_name_overflow_is_local_and_consumes_nothing() -> None:
    ledger = BudgetLedger(
        BudgetConstraints(
            max_task_tool_calls=20,
            max_task_tool_calls_by_name={
                "probe_feature": 3,
                "get_accepted_hypothesis": 1,
            },
        )
    )
    for _ in range(3):
        assert ledger.try_consume_tool_calls([_req("probe_feature")]).accepted

    rejected = ledger.try_consume_tool_calls([_req("probe_feature")])

    assert rejected.accepted is False
    assert rejected.rejection_reason == "task_tool_calls_by_name:probe_feature"
    snapshot = ledger.snapshot()
    assert snapshot.task_tool_calls_used == 3
    assert snapshot.task_tool_calls_by_name_used == {"probe_feature": 3}
    assert ledger.exhausted() is None
    assert ledger.try_consume_tool_calls([_req("get_accepted_hypothesis")]).accepted


def test_per_name_call_decrements_both_counters() -> None:
    ledger = BudgetLedger(
        BudgetConstraints(
            max_task_tool_calls=10,
            max_task_tool_calls_by_name={"deploy": 3},
        )
    )
    ledger.try_consume_tool_calls([_req("deploy")])
    snap = ledger.snapshot()
    # Both global and per-name counters incremented
    assert snap.task_tool_calls_used == 1
    assert snap.task_tool_calls_by_name_used.get("deploy") == 1


def test_unknown_capability_is_tracked_for_later_per_name_limits() -> None:
    ledger = BudgetLedger(
        BudgetConstraints(
            max_task_tool_calls=10,
            max_task_tool_calls_by_name={"deploy": 3},
        )
    )
    ledger.try_consume_tool_calls([_req("run_shell")])
    snap = ledger.snapshot()
    assert snap.task_tool_calls_used == 1
    # A later step can introduce a named limit, so prior usage must be retained.
    assert snap.task_tool_calls_by_name_used.get("run_shell") == 1


def test_exhausted_returns_none_when_not_exhausted() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    assert ledger.exhausted() is None


def test_exhausted_returns_exhaustion_after_rejection() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=1))
    ledger.try_consume_tool_calls([_req("run"), _req("run")])  # would exceed limit
    ex = ledger.exhausted()
    assert isinstance(ex, BudgetExhaustion)
    assert ex.kind == "task_tool_calls"
    subsequent = ledger.try_consume_tool_calls([_req("other")])
    assert subsequent.accepted is False
    assert subsequent.rejection_reason == "task_tool_calls"


def test_short_circuit_after_exhaustion() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=1))
    ledger.try_consume_tool_calls([_req("r"), _req("r")])  # reject
    # Further calls should short-circuit immediately
    result = ledger.try_consume_tool_calls([_req("r")])
    assert result.accepted is False


def test_no_limit_accepts_unlimited_calls() -> None:
    ledger = BudgetLedger(BudgetConstraints())  # no limits
    for _ in range(100):
        result = ledger.try_consume_tool_calls([_req("run")])
        assert result.accepted is True
    assert ledger.snapshot().task_tool_calls_used == 100


def test_snapshot_budget_to_agent_dict_structure() -> None:
    ledger = BudgetLedger(
        BudgetConstraints(
            max_task_tool_calls=20,
            max_task_tool_calls_by_name={"deploy": 5},
            max_llm_turns=30,
        )
    )
    ledger.try_consume_tool_calls([_req("deploy")])
    ledger.record_llm_turn()
    d = ledger.snapshot().to_agent_dict()
    assert d["budget"]["task_tool_calls"]["used"] == 1
    assert d["budget"]["task_tool_calls"]["limit"] == 20
    assert d["budget"]["task_tool_calls"]["remaining"] == 19
    assert d["budget"]["task_tool_calls"]["exhausted"] is False
    assert d["budget"]["task_tool_calls_by_name"]["deploy"]["used"] == 1
    assert d["budget"]["task_tool_calls_by_name"]["deploy"]["limit"] == 5
    assert d["budget"]["llm_turns"]["used"] == 1
    assert d["budget"]["llm_turns"]["limit"] == 30
    assert d["budget"]["llm_turns"]["remaining"] == 29


def test_step_push_changes_effective_limits_without_resetting_counters() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    ledger.try_consume_tool_calls([_req("run"), _req("run")])

    ledger.push_step("tight", BudgetConstraints(max_task_tool_calls=3))

    snap = ledger.snapshot()
    assert snap.task_tool_calls_used == 2
    assert snap.task_tool_calls_limit == 3
    assert ledger.try_consume_tool_calls([_req("run")]).accepted is True
    assert ledger.try_consume_tool_calls([_req("run")]).accepted is False
    assert ledger.exhausted() is not None


def test_step_exact_limit_entry_does_not_mark_exhaustion_until_next_call() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    ledger.try_consume_tool_calls([_req("run"), _req("run")])

    ledger.push_step("exact", BudgetConstraints(max_task_tool_calls=2))


    assert ledger.exhausted() is None
    result = ledger.try_consume_tool_calls([_req("run")])
    assert result.accepted is False
    assert result.rejection_reason == "task_tool_calls"


def test_step_entry_marks_exhaustion_when_prior_usage_already_exceeds_limit() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))
    ledger.try_consume_tool_calls([_req("run"), _req("run"), _req("run")])

    ledger.push_step("too-tight", BudgetConstraints(max_task_tool_calls=2))

    ex = ledger.exhausted()
    assert ex is not None
    assert ex.kind == "task_tool_calls"


def test_step_per_name_limits_use_accumulated_prior_usage() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=10))
    ledger.try_consume_tool_calls([_req("deploy"), _req("deploy")])

    ledger.push_step(
        "deploy-step",
        BudgetConstraints(max_task_tool_calls=10, max_task_tool_calls_by_name={"deploy": 2}),
    )

    assert ledger.exhausted() is None
    result = ledger.try_consume_tool_calls([_req("deploy")])
    assert result.accepted is False
    assert result.rejection_reason == "task_tool_calls_by_name:deploy"
    assert ledger.exhausted() is None
    assert ledger.try_consume_tool_calls([_req("other")]).accepted


@pytest.mark.parametrize("prior_usage", [2, 3])
def test_step_per_name_cap_at_or_below_prior_usage_does_not_latch(
    prior_usage: int,
) -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=10))
    assert ledger.try_consume_tool_calls(
        [_req("deploy") for _ in range(prior_usage)]
    ).accepted

    ledger.push_step(
        "deploy-step",
        BudgetConstraints(
            max_task_tool_calls=10,
            max_task_tool_calls_by_name={"deploy": 2},
        ),
    )

    assert ledger.exhausted() is None
    rejected = ledger.try_consume_tool_calls([_req("deploy")])
    assert rejected.rejection_reason == "task_tool_calls_by_name:deploy"
    assert ledger.exhausted() is None
    assert ledger.try_consume_tool_calls([_req("other")]).accepted


def test_concurrent_per_name_overflow_accepts_exactly_limit_without_latching() -> None:
    contender_count = 32
    limit = 7
    barrier = threading.Barrier(contender_count)
    ledger = BudgetLedger(
        BudgetConstraints(
            max_task_tool_calls=contender_count,
            max_task_tool_calls_by_name={"probe_feature": limit},
        )
    )

    def _attempt():
        barrier.wait()
        return ledger.try_consume_tool_calls([_req("probe_feature")])

    with ThreadPoolExecutor(max_workers=contender_count) as executor:
        reservations = list(
            executor.map(lambda _index: _attempt(), range(contender_count))
        )

    accepted = [reservation for reservation in reservations if reservation.accepted]
    rejected = [reservation for reservation in reservations if not reservation.accepted]
    assert len(accepted) == limit
    assert len(rejected) == contender_count - limit
    assert {reservation.rejection_reason for reservation in rejected} == {
        "task_tool_calls_by_name:probe_feature"
    }
    snapshot = ledger.snapshot()
    assert snapshot.task_tool_calls_used == limit
    assert snapshot.task_tool_calls_by_name_used["probe_feature"] == limit
    assert ledger.exhausted() is None


def test_step_pop_restores_episode_limits() -> None:
    ledger = BudgetLedger(BudgetConstraints(max_task_tool_calls=5))

    ledger.push_step("tight", BudgetConstraints(max_task_tool_calls=1))
    assert ledger.snapshot().task_tool_calls_limit == 1
    ledger.pop_step("tight")

    assert ledger.snapshot().task_tool_calls_limit == 5
    assert ledger.active_step_name() is None
