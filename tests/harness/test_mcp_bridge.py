"""``CapabilityMcpBridge`` translates tool calls to ``execute_capability``.

The bridge is the framework-owned dispatcher by which harnesses reach
task-side capability execution. The load-bearing invariants:

- Every invocation emits a paired ``action`` + ``observation`` event with a
  shared ``correlation_id`` — unpaired events are a bridge bug.
- Exceptions inside ``execute_capability`` still produce a paired
  observation (failure pairing) and a structured error to the MCP client.
- Parallel invocations (ms-agent fires tool calls concurrently) still pair
  correctly — correlation_id, not order, is the join key.
- ``execute_capability`` runs in a worker thread — the bridge's event loop
  is never blocked by a long capability call.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from mcp import types as mcp_types

from src.framework.capability_bridge import CapabilityMcpBridge, _CURRENT_WORKFLOW_STEP
from src.harness.budget import BudgetLedger
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.task.types import (
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    BudgetConstraints,
    EpisodeConstraints,
    StepConstraints,
    TaskPromptSpec,
    Variation,
)


class _DummyTask:
    metric_name = "dummy"
    metric_unit = ""
    name = "stub"
    description = "stub"

    def __init__(
        self,
        *,
        sleep_seconds: float = 0.0,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.calls: list[CapabilityInvocation] = []
        self._sleep = sleep_seconds
        self._raise = raise_on_call
        self._lock = threading.Lock()

    def parse_response(self, raw_response, *, invoked_capability=None):
        return []

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers,
        variation,
        ctx,
    ) -> CapabilityResult:
        with self._lock:
            self.calls.append(invocation)
        if self._sleep:
            time.sleep(self._sleep)
        if self._raise is not None:
            raise self._raise
        return CapabilityResult(
            name=invocation.name,
            output={"echo": invocation.input},
            success=True,
        )


def _build_ctx(tmp_path: Path, *, capabilities: list[Capability]) -> HarnessContext:
    recorder = EventRecorder(episode_id="ep-1", output_path=tmp_path / "events.jsonl")
    traced = TracedGenner(
        inner=None,  # type: ignore[arg-type]  — bridge does not use genner
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id="ep-1",
    )
    prompt_spec = TaskPromptSpec(
        system_instruction="sys",
        capabilities=capabilities,
    )
    return HarnessContext(
        episode_id="ep-1",
        genner=traced,
        task=_DummyTask(),  # type: ignore[arg-type]
        variation=Variation(name="v", description="d"),
        prompt_spec=prompt_spec,
        episode_context={},
        containers=[],
        agent_container=None,  # type: ignore[arg-type]
        host_cache_folder=tmp_path,
        config=HarnessConfigView(
            harness_settings={},
            model_name="test",
            train_data_save_folder=str(tmp_path),
            code_host_cache_path=str(tmp_path),
        ),
        metrics=None,  # type: ignore[arg-type]
        recorder=recorder,
        cancel_event=threading.Event(),
    )


def _correlation_ids_in_order(events) -> list[tuple[str, str, str]]:
    """Return (kind, category, correlation_id) for events that carry one."""
    out = []
    for ev in events:
        cid = ev.payload.get("correlation_id")
        if cid is None:
            continue
        out.append((ev.kind, ev.category, cid))
    return out


def test_invoke_executes_capability_and_pairs_events(tmp_path: Path) -> None:
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    dummy = ctx.task  # type: ignore[assignment]
    bridge = CapabilityMcpBridge(ctx, token="t")

    result = asyncio.run(bridge._invoke(cap, {"x": 1}))

    # task.execute_capability was called with the right args.
    assert len(dummy.calls) == 1  # type: ignore[attr-defined]
    assert dummy.calls[0].name == "analyzer"  # type: ignore[attr-defined]
    assert dummy.calls[0].input == {"x": 1}  # type: ignore[attr-defined]

    # Full CapabilityResult shape. Transport-only truncation/JSON conversion
    # happens in the MCP call_tool adapter, not in the dispatcher.
    assert result.success is True
    assert result.error in (None, "")
    assert result.output == {"echo": {"x": 1}}

    # action + observation share one correlation_id.
    pairs = _correlation_ids_in_order(ctx.recorder.events)
    assert len(pairs) == 2
    assert pairs[0][0] == "action"
    assert pairs[1][0] == "observation"
    assert pairs[0][2] == pairs[1][2]
    assert ctx.recorder.snapshot_counters()["tool_calls"] == 1
    assert ctx.recorder.snapshot_labels()["last_tool"] == "analyzer"


def test_inprocess_invoke_does_not_truncate_large_stream_outputs(
    tmp_path: Path,
) -> None:
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])

    class _LargeOutputTask(_DummyTask):
        def execute_capability(self, invocation, containers, variation, ctx):
            return CapabilityResult(
                name=invocation.name,
                output={
                    "stdout": "x" * 5000,
                    "stderr": "",
                    "return_code": 0,
                    "success": True,
                },
                success=True,
            )

    ctx.task = _LargeOutputTask()  # type: ignore[assignment]
    bridge = CapabilityMcpBridge(ctx, token="t")

    result = bridge.invoke(CapabilityInvocation(name=cap.name, input={}))

    assert len(result.output["stdout"]) == 5000
    observation = next(
        e for e in ctx.recorder.events if e.category == "mcp_capability_result"
    )
    observed = observation.payload["result"]["output"]["stdout"]
    assert len(observed) == 5000


def test_call_tool_truncates_large_stream_outputs_for_agent(tmp_path: Path) -> None:
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])

    class _LargeOutputTask(_DummyTask):
        def execute_capability(self, invocation, containers, variation, ctx):
            return CapabilityResult(
                name=invocation.name,
                output={
                    "stdout": "x" * 5000,
                    "stderr": "",
                    "return_code": 0,
                    "success": True,
                },
                success=True,
            )

    ctx.task = _LargeOutputTask()  # type: ignore[assignment]
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    result = _call_tool(server, cap.name, {})

    payload = json.loads(result.content[0].text)
    stdout = payload["output"]["stdout"]
    assert len(stdout) < 5000
    assert "[truncated" in stdout
    # Recorder keeps the task's full structured result; only the agent-facing
    # transport payload is truncated.
    observation = next(
        e for e in ctx.recorder.events if e.category == "mcp_capability_result"
    )
    observed = observation.payload["result"]["output"]["stdout"]
    assert len(observed) == 5000


def test_inprocess_invoke_accepts_synthetic_invocation_not_advertised(
    tmp_path: Path,
) -> None:
    cap = Capability(name="run_python", description="exec")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    bridge = CapabilityMcpBridge(ctx, token="t")

    result = bridge.invoke(CapabilityInvocation(name="metric_report", input={"x": 1}))

    assert result.success is True
    assert result.name == "metric_report"
    pairs = ctx.recorder.capability_pairs()
    assert pairs[0][0].name == "metric_report"
    assert pairs[0][0].input == {"x": 1}


def test_invoke_exception_still_pairs_events_and_returns_error(tmp_path: Path) -> None:
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    # Swap in a task that raises
    ctx.task = _DummyTask(raise_on_call=RuntimeError("boom"))  # type: ignore[assignment]
    bridge = CapabilityMcpBridge(ctx, token="t")

    result = asyncio.run(bridge._invoke(cap, {}))

    # Structured error returned to MCP client — does NOT propagate traceback.
    assert result.success is False
    assert "boom" in (result.error or "")

    # Pairing invariant preserved through the exception path.
    pairs = _correlation_ids_in_order(ctx.recorder.events)
    assert len(pairs) == 2
    assert pairs[0][0] == "action"
    assert pairs[1][0] == "observation"
    assert pairs[0][2] == pairs[1][2]


def test_invoke_after_cancel_raises_before_execute(tmp_path: Path) -> None:
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    ctx.cancel_event.set()
    bridge = CapabilityMcpBridge(ctx, token="t")

    with pytest.raises(Exception):  # noqa: BLE001 — McpError / HarnessError
        asyncio.run(bridge._invoke(cap, {}))

    # Nothing executed; recorder stayed clean.
    assert ctx.task.calls == []  # type: ignore[attr-defined]


def test_parallel_invocations_pair_by_correlation_id(tmp_path: Path) -> None:
    """ms-agent fires tool calls concurrently. Correlation_id, not order,
    is the join key — pairs must line up even when action/observation
    records interleave."""
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    # Non-trivial sleep so executions overlap under asyncio.gather
    ctx.task = _DummyTask(sleep_seconds=0.05)  # type: ignore[assignment]
    bridge = CapabilityMcpBridge(ctx, token="t")

    async def run_many() -> list[CapabilityResult]:
        return await asyncio.gather(*[bridge._invoke(cap, {"i": i}) for i in range(10)])

    results = asyncio.run(run_many())
    assert len(results) == 10
    assert all(r.success is True for r in results)

    pairs = _correlation_ids_in_order(ctx.recorder.events)
    # 10 actions + 10 observations — 20 total
    actions = [cid for kind, _, cid in pairs if kind == "action"]
    observations = [cid for kind, _, cid in pairs if kind == "observation"]
    assert len(actions) == 10
    assert len(observations) == 10
    # Every action's correlation_id has an observation counterpart.
    assert set(actions) == set(observations)
    assert ctx.recorder.snapshot_counters()["tool_calls"] == 10


def test_bridge_uses_capability_pairs_helper(tmp_path: Path) -> None:
    """``EventRecorder.capability_pairs()`` joins action + observation by
    correlation_id into ``(invocation, result)`` tuples — that's the public
    reconstruction API that profiles consume."""
    cap = Capability(name="analyzer", description="read")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    bridge = CapabilityMcpBridge(ctx, token="t")

    asyncio.run(bridge._invoke(cap, {"x": 1}))
    asyncio.run(bridge._invoke(cap, {"x": 2}))

    pairs = ctx.recorder.capability_pairs()
    assert len(pairs) == 2
    invs = [p[0] for p in pairs]
    assert [i.input.get("x") for i in invs] == [1, 2]


def _list_tools(server) -> list[mcp_types.Tool]:
    """Drive ``Server.list_tools`` request handler and unwrap the tools."""
    handler = server.request_handlers[mcp_types.ListToolsRequest]
    req = mcp_types.ListToolsRequest(method="tools/list")
    result = asyncio.run(handler(req))
    return list(result.root.tools)


def _call_tool(server, name: str, arguments: dict) -> mcp_types.CallToolResult:
    handler = server.request_handlers[mcp_types.CallToolRequest]
    req = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = asyncio.run(handler(req))
    return result.root


def test_bridge_advertises_every_declared_capability(tmp_path: Path) -> None:
    """Advertise-by-default: every entry in ``prompt_spec.capabilities`` is
    exposed as MCP. The previous ``mcp.advertise`` annotation gate is gone
    — tasks own the MCP surface, and there is no internal capability list."""
    caps = [
        Capability(name="deploy_attack_sol", description="deploy"),
        Capability(name="run_shell", description="exec"),
    ]
    ctx = _build_ctx(tmp_path, capabilities=caps)
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    tools = _list_tools(server)
    assert sorted(t.name for t in tools) == ["deploy_attack_sol", "run_shell"]


def test_bridge_dispatch_delivers_flat_kwargs_end_to_end(tmp_path: Path) -> None:
    cap = Capability(
        name="deploy_attack_sol",
        description="deploy",
    )
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    result = _call_tool(server, cap.name, {"attack_sol": "x"})

    assert ctx.task.calls[0].input == {"attack_sol": "x"}  # type: ignore[attr-defined]
    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["success"] is True
    assert payload["output"] == {"echo": {"attack_sol": "x"}}


def test_call_tool_applies_workflow_step_budget_from_request_context(
    tmp_path: Path,
) -> None:
    cap = Capability(name="run_python", description="exec")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    ctx.constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=5),
        step_overrides={
            "plan": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=0))
        },
    )
    ctx.budget_ledger = BudgetLedger(ctx.constraints.budgets)
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    token = _CURRENT_WORKFLOW_STEP.set("plan")
    try:
        result = _call_tool(server, cap.name, {})
    finally:
        _CURRENT_WORKFLOW_STEP.reset(token)

    payload = json.loads(result.content[0].text)
    assert result.isError is True
    assert payload["error"] == "budget exhausted: task_tool_calls"
    assert payload["budget"]["task_tool_calls"]["limit"] == 0
    assert ctx.task.calls == []  # type: ignore[attr-defined]
    assert ctx.budget_ledger.active_step_name() is None
    assert ctx.recorder.snapshot_labels()["last_workflow_step"] == "plan"


def test_bridge_budget_scope_unwinds_when_push_logging_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = Capability(name="run_python", description="exec")
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    ctx.constraints = EpisodeConstraints(
        budgets=BudgetConstraints(max_task_tool_calls=5),
        step_overrides={
            "plan": StepConstraints(budgets=BudgetConstraints(max_task_tool_calls=2))
        },
    )
    ctx.budget_ledger = BudgetLedger(ctx.constraints.budgets)
    bridge = CapabilityMcpBridge(ctx, token="t")

    original_log_state = ctx.recorder.log_state

    def fail_push_log(category, payload):
        if category == "budget_ledger_step_push":
            raise RuntimeError("recorder failed")
        return original_log_state(category, payload)

    monkeypatch.setattr(ctx.recorder, "log_state", fail_push_log)

    token = _CURRENT_WORKFLOW_STEP.set("plan")
    try:
        with pytest.raises(RuntimeError, match="recorder failed"):
            with bridge._workflow_step_budget_scope():
                pass
    finally:
        _CURRENT_WORKFLOW_STEP.reset(token)

    assert ctx.budget_ledger.active_step_name() is None


def test_bridge_advertises_cap_schema_when_set(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"attack_sol": {"type": "string"}},
        "required": ["attack_sol"],
    }
    cap = Capability(
        name="deploy_attack_sol",
        description="deploy",
        schema=schema,
    )
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    tools = _list_tools(server)
    assert tools[0].inputSchema == schema


def test_bridge_advertises_open_object_when_no_schema(tmp_path: Path) -> None:
    cap = Capability(
        name="deploy_attack_sol",
        description="deploy",
    )
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    tools = _list_tools(server)
    assert tools[0].inputSchema == {"type": "object", "additionalProperties": True}


def test_bridge_typed_schema_rejects_missing_required(tmp_path: Path) -> None:
    """Low-level SDK validates ``arguments`` against ``inputSchema`` via
    ``jsonschema``. A typed schema's required field, when missing, must
    surface as ``isError=True`` before ``execute_capability`` runs."""
    cap = Capability(
        name="deploy_attack_sol",
        description="deploy",
        schema={
            "type": "object",
            "properties": {"attack_sol": {"type": "string"}},
            "required": ["attack_sol"],
        },
    )
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    result = _call_tool(server, cap.name, {})

    assert result.isError is True
    assert "validation" in result.content[0].text.lower()
    # task was never called — validation rejected at the MCP boundary
    assert ctx.task.calls == []  # type: ignore[attr-defined]


def test_bridge_open_schema_accepts_arbitrary_keys(tmp_path: Path) -> None:
    cap = Capability(
        name="deploy_attack_sol",
        description="deploy",
    )
    ctx = _build_ctx(tmp_path, capabilities=[cap])
    server = CapabilityMcpBridge(ctx, token="t")._build_mcp_server()

    result = _call_tool(server, cap.name, {"foo": 1, "bar": "baz"})

    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["success"] is True
    assert ctx.task.calls[0].input == {"foo": 1, "bar": "baz"}  # type: ignore[attr-defined]
