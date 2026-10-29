"""Tests for CapabilityMcpBridge budget enforcement.

TDD: these tests verify that the bridge rejects tool calls when the budget is
exhausted, returns one error per request, and still logs recorder events.
"""

from unittest.mock import MagicMock
import asyncio
import json

from mcp import types as mcp_types

from src.framework.capability_bridge import CapabilityMcpBridge
from src.harness.budget import BudgetLedger
from src.task.types import (
    BudgetConstraints,
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeConstraints,
    TaskPromptSpec,
)


def _make_ctx(
    *,
    max_task_tool_calls: int | None = None,
    max_task_tool_calls_by_name: dict[str, int] | None = None,
    capability_names: tuple[str, ...] = ("run",),
):
    """Build a minimal HarnessContext mock with budget ledger attached."""
    constraints = EpisodeConstraints(
        budgets=BudgetConstraints(
            max_task_tool_calls=max_task_tool_calls,
            max_task_tool_calls_by_name=max_task_tool_calls_by_name or {},
        )
    )
    ledger = BudgetLedger(constraints.budgets)

    recorder = MagicMock()
    recorder.bump_counter = MagicMock()
    recorder.set_label = MagicMock()

    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=[
            Capability(name=name, description=f"call {name}")
            for name in capability_names
        ],
    )
    ctx = MagicMock()
    ctx.recorder = recorder
    ctx.cancel_event = MagicMock()
    ctx.cancel_event.is_set.return_value = False
    ctx.constraints = constraints
    ctx.budget_ledger = ledger
    ctx.config.get.return_value = 0  # tool_output_max_chars = 0
    ctx.prompt_spec = prompt_spec
    return ctx, ledger


def test_bridge_accepts_call_within_budget() -> None:
    ctx, ledger = _make_ctx(max_task_tool_calls=5)
    bridge = CapabilityMcpBridge(ctx, token="")

    from src.task.types import CapabilityResult

    ctx.execute_capability.return_value = CapabilityResult(
        name="run", output={"ok": True}, success=True
    )

    result = bridge.invoke(CapabilityInvocation(name="run", input={}))
    assert result.success is True
    assert ledger.snapshot().task_tool_calls_used == 1


def test_bridge_rejects_call_when_budget_exhausted() -> None:
    ctx, ledger = _make_ctx(max_task_tool_calls=1)
    bridge = CapabilityMcpBridge(ctx, token="")

    from src.task.types import CapabilityResult

    ctx.execute_capability.return_value = CapabilityResult(
        name="run", output={}, success=True
    )

    # First call consumes the budget
    bridge.invoke(CapabilityInvocation(name="run", input={}))
    # Second call should be rejected
    result = bridge.invoke(CapabilityInvocation(name="run", input={}))
    assert result.success is False
    assert result.error is not None
    assert "budget" in result.error.lower() or "exhausted" in result.error.lower()
    # Budget not double-consumed
    assert ledger.snapshot().task_tool_calls_used == 1


def test_bridge_without_budget_ledger_accepts_unlimited() -> None:
    """If ctx.budget_ledger is None, the bridge imposes no budget."""
    ctx, _ = _make_ctx(max_task_tool_calls=None)
    ctx.budget_ledger = None  # type: ignore[assignment]
    bridge = CapabilityMcpBridge(ctx, token="")

    from src.task.types import CapabilityResult

    ctx.execute_capability.return_value = CapabilityResult(
        name="run", output={}, success=True
    )

    for _ in range(10):
        result = bridge.invoke(CapabilityInvocation(name="run", input={}))
        assert result.success is True


def test_bridge_rejection_does_not_log_action_event() -> None:
    """Rejected calls should not fire action/observation recorder events."""
    ctx, ledger = _make_ctx(max_task_tool_calls=0)
    bridge = CapabilityMcpBridge(ctx, token="")

    result = bridge.invoke(CapabilityInvocation(name="run", input={}))
    assert result.success is False
    ctx.recorder.log_action.assert_not_called()
    ctx.recorder.log_observation.assert_not_called()


def test_bridge_result_includes_budget_field_when_ledger_present() -> None:
    """Tool result payload should include a 'budget' key when ledger is attached."""
    ctx, ledger = _make_ctx(max_task_tool_calls=10)
    bridge = CapabilityMcpBridge(ctx, token="")

    from src.task.types import CapabilityResult

    ctx.execute_capability.return_value = CapabilityResult(
        name="run", output={"x": 1}, success=True
    )

    result = bridge.invoke(CapabilityInvocation(name="run", input={}))
    payload = bridge._result_to_tool_payload(result)
    assert "budget" in payload
    assert payload["budget"]["task_tool_calls"]["used"] == 1


def _call_tool(server, name: str, arguments: dict) -> mcp_types.CallToolResult:
    handler = server.request_handlers[mcp_types.CallToolRequest]
    req = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = asyncio.run(handler(req))
    return result.root


def test_mcp_call_tool_rejects_with_is_error_when_budget_exhausted() -> None:
    ctx, ledger = _make_ctx(max_task_tool_calls=0)
    server = CapabilityMcpBridge(ctx, token="")._build_mcp_server()

    result = _call_tool(server, "run", {})

    assert result.isError is True
    payload = json.loads(result.content[0].text)
    assert payload["success"] is False
    assert "budget exhausted" in payload["error"]
    assert payload["budget"]["task_tool_calls"]["used"] == 0
    assert ledger.exhausted() is not None


def test_per_name_rejection_payload_does_not_block_other_tool() -> None:
    ctx, ledger = _make_ctx(
        max_task_tool_calls=20,
        max_task_tool_calls_by_name={
            "probe_feature": 3,
            "get_accepted_hypothesis": 1,
        },
        capability_names=("probe_feature", "get_accepted_hypothesis"),
    )
    ctx.execute_capability.side_effect = lambda invocation: CapabilityResult(
        name=invocation.name,
        output={"called": invocation.name},
        success=True,
    )
    server = CapabilityMcpBridge(ctx, token="")._build_mcp_server()

    for _ in range(3):
        assert _call_tool(server, "probe_feature", {}).isError is False
    rejected = _call_tool(server, "probe_feature", {})

    assert rejected.isError is True
    rejected_payload = json.loads(rejected.content[0].text)
    assert rejected_payload["error"] == (
        "budget exhausted: task_tool_calls_by_name:probe_feature"
    )
    by_name = rejected_payload["budget"]["task_tool_calls_by_name"]
    assert by_name["probe_feature"] == {
        "used": 3,
        "limit": 3,
        "remaining": 0,
        "exhausted": True,
    }
    assert by_name["get_accepted_hypothesis"] == {
        "used": 0,
        "limit": 1,
        "remaining": 1,
        "exhausted": False,
    }

    accepted = _call_tool(server, "get_accepted_hypothesis", {})

    assert accepted.isError is False
    accepted_payload = json.loads(accepted.content[0].text)
    assert accepted_payload["success"] is True
    assert (
        accepted_payload["budget"]["task_tool_calls_by_name"][
            "get_accepted_hypothesis"
        ]["used"]
        == 1
    )
    assert ledger.exhausted() is None
    assert ctx.execute_capability.call_count == 4
