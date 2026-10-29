"""BudgetLedger — framework-owned per-episode accounting for task constraints.

The ledger tracks two independent counters:
  - ``task_tool_calls``: hard-enforced at ``CapabilityMcpBridge``.
  - ``llm_turns``: advisory only; tracked but never used to block a model call.

Only the global task-tool-call limit can trigger episode-level exhaustion.
Per-name overages reject the offending reservation without blocking other
capabilities. ``record_llm_turn`` is unconditionally advisory — it increments
the counter regardless of whether ``max_llm_turns`` is exceeded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Sequence

from src.task.types import BudgetConstraints


@dataclass
class ToolCallRequest:
    capability_name: str


@dataclass
class ToolCallReservation:
    accepted: bool
    rejection_reason: str | None = None


@dataclass
class BudgetExhaustion:
    kind: str  # "task_tool_calls"
    step: str | None = None


@dataclass
class BudgetSnapshot:
    task_tool_calls_used: int
    task_tool_calls_limit: int | None
    task_tool_calls_by_name_used: dict[str, int]
    task_tool_calls_by_name_limits: dict[str, int]
    llm_turns_used: int
    llm_turns_limit: int | None

    def to_agent_dict(self) -> dict[str, Any]:
        """Return the agent-visible budget object as a plain dict."""
        tc_limit = self.task_tool_calls_limit
        tc_used = self.task_tool_calls_used
        tc_remaining = (tc_limit - tc_used) if tc_limit is not None else None
        tc_exhausted = tc_remaining is not None and tc_remaining <= 0

        lt_limit = self.llm_turns_limit
        lt_used = self.llm_turns_used
        lt_remaining = (lt_limit - lt_used) if lt_limit is not None else None
        lt_exhausted = lt_remaining is not None and lt_remaining <= 0

        by_name: dict[str, Any] = {}
        for name, limit in self.task_tool_calls_by_name_limits.items():
            used = self.task_tool_calls_by_name_used.get(name, 0)
            remaining = limit - used
            by_name[name] = {
                "used": used,
                "limit": limit,
                "remaining": remaining,
                "exhausted": remaining <= 0,
            }

        result: dict[str, Any] = {
            "budget": {
                "task_tool_calls": {
                    "used": tc_used,
                    "limit": tc_limit,
                    "remaining": tc_remaining,
                    "exhausted": tc_exhausted,
                },
            }
        }
        if by_name:
            result["budget"]["task_tool_calls_by_name"] = by_name
        result["budget"]["llm_turns"] = {
            "used": lt_used,
            "limit": lt_limit,
            "remaining": lt_remaining,
            "exhausted": lt_exhausted,
        }
        return result


class BudgetLedger:
    """Per-episode budget ledger. Thread-safe; all mutation under a lock."""

    def __init__(self, constraints: BudgetConstraints) -> None:
        self._constraints = constraints
        self._step_stack: list[tuple[str, BudgetConstraints]] = []
        self._task_tool_calls_used: int = 0
        self._task_tool_calls_by_name_used: dict[str, int] = {}
        self._llm_turns_used: int = 0
        self._exhaustion: BudgetExhaustion | None = None
        self._lock = threading.Lock()

    def _effective_constraints(self) -> BudgetConstraints:
        """Return top-of-stack override or episode default. Lock must be held."""
        if self._step_stack:
            return self._step_stack[-1][1]
        return self._constraints

    def push_step(self, name: str, constraints: BudgetConstraints) -> BudgetSnapshot:
        """Push per-step budget constraints. Does NOT reset used counters."""
        with self._lock:
            self._step_stack.append((name, constraints))
            if self._exhaustion is not None:
                return self._snapshot_locked()
            self._mark_global_exhaustion_if_already_over_limit_locked(constraints)
            return self._snapshot_locked()

    def pop_step(self, name: str) -> BudgetSnapshot:
        """Pop the named step's budget constraints from the stack."""
        with self._lock:
            if not self._step_stack:
                raise ValueError(f"cannot pop workflow step {name!r}; no active step")
            active_name, _ = self._step_stack[-1]
            if active_name != name:
                raise ValueError(
                    f"cannot pop workflow step {name!r}; active step is {active_name!r}"
                )
            self._step_stack.pop()
            return self._snapshot_locked()

    def active_step_name(self) -> str | None:
        """Return the name of the currently active step override, or None."""
        with self._lock:
            return self._step_stack[-1][0] if self._step_stack else None

    def record_llm_turn(self) -> BudgetSnapshot:
        """Increment the llm_turn counter. Advisory only — never raises."""
        with self._lock:
            self._llm_turns_used += 1
            return self._snapshot_locked()

    def try_consume_tool_calls(
        self, requests: Sequence[ToolCallRequest]
    ) -> ToolCallReservation:
        """Atomically check and consume budget for a batch of tool calls.

        Returns an accepted reservation if all calls fit within remaining
        budget (global and per-name). If any call in the batch would exceed
        budget the entire batch is rejected and consumes nothing. Only global
        overflow latches episode-level exhaustion; per-name overflow is local
        to the rejected reservation.
        """
        with self._lock:
            if self._exhaustion is not None:
                return ToolCallReservation(
                    accepted=False,
                    rejection_reason=self._exhaustion.kind,
                )

            batch_size = len(requests)
            constraints = self._effective_constraints()

            max_calls = constraints.max_task_tool_calls
            if max_calls is not None:
                if self._task_tool_calls_used + batch_size > max_calls:
                    kind = "task_tool_calls"
                    self._exhaustion = BudgetExhaustion(
                        kind=kind,
                        step=self._active_step_name_locked(),
                    )
                    return ToolCallReservation(accepted=False, rejection_reason=kind)

            batch_counts: dict[str, int] = {}
            for req in requests:
                batch_counts[req.capability_name] = (
                    batch_counts.get(req.capability_name, 0) + 1
                )

            for name, batch_count in batch_counts.items():
                per_name_limit = constraints.max_task_tool_calls_by_name.get(name)
                if per_name_limit is not None:
                    used = self._task_tool_calls_by_name_used.get(name, 0)
                    if used + batch_count > per_name_limit:
                        kind = f"task_tool_calls_by_name:{name}"
                        return ToolCallReservation(accepted=False, rejection_reason=kind)

            self._task_tool_calls_used += batch_size
            for name, count in batch_counts.items():
                # Track ALL names so a later step can introduce a per-name cap
                # and still see prior usage from before it became active.
                self._task_tool_calls_by_name_used[name] = (
                    self._task_tool_calls_by_name_used.get(name, 0) + count
                )

            return ToolCallReservation(accepted=True)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def exhausted(self) -> BudgetExhaustion | None:
        with self._lock:
            return self._exhaustion

    def _snapshot_locked(self) -> BudgetSnapshot:
        constraints = self._effective_constraints()
        return BudgetSnapshot(
            task_tool_calls_used=self._task_tool_calls_used,
            task_tool_calls_limit=constraints.max_task_tool_calls,
            task_tool_calls_by_name_used=dict(self._task_tool_calls_by_name_used),
            task_tool_calls_by_name_limits=dict(
                constraints.max_task_tool_calls_by_name
            ),
            llm_turns_used=self._llm_turns_used,
            llm_turns_limit=constraints.max_llm_turns,
        )

    def _active_step_name_locked(self) -> str | None:
        return self._step_stack[-1][0] if self._step_stack else None

    def _mark_global_exhaustion_if_already_over_limit_locked(
        self,
        constraints: BudgetConstraints,
    ) -> None:
        max_calls = constraints.max_task_tool_calls
        if max_calls is not None and self._task_tool_calls_used > max_calls:
            self._exhaustion = BudgetExhaustion(
                kind="task_tool_calls",
                step=self._active_step_name_locked(),
            )


__all__ = [
    "BudgetExhaustion",
    "BudgetLedger",
    "BudgetSnapshot",
    "ToolCallRequest",
    "ToolCallReservation",
]
