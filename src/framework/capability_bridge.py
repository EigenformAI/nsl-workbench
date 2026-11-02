"""CapabilityMcpBridge — framework-owned capability dispatch and MCP transport.

Per-episode bridge. Each declared :class:`Capability` can be exposed as an
MCP tool for external harnesses, while in-process harnesses can dispatch
``CapabilityInvocation`` objects directly through the same bridge-owned
recorder/event path.

Invariants
----------

- **Action/observation pairing**. Every invocation emits paired
  ``mcp_capability_call`` + ``mcp_capability_result`` events on the recorder,
  sharing a ``correlation_id``. Exceptions in ``execute_capability`` still
  emit a paired observation (failure pairing).
- **No event-loop blocking for MCP transport**. ``execute_capability`` is
  synchronous; the async MCP adapter dispatches it through
  ``asyncio.to_thread`` so a long-running capability does not stall
  concurrent MCP tool calls.
- **Auth at transport**. Bearer token is required on every request; the HTTP
  layer (when served via ``serve_http``) rejects missing/wrong tokens
  with 401.

Cancellation caveat
-------------------
``execute_capability`` has no cooperative-cancellation hook. The bridge checks
``cancel_event`` at the pre-call boundary only; a capability that started
executing before cancellation runs to completion.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import socket
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from src.task.base import TaskEnvironmentError

from src.harness.base import HarnessError
from src.harness.budget import ToolCallRequest
from src.task.types import Capability, CapabilityInvocation, CapabilityResult

if TYPE_CHECKING:
    from src.harness.context import HarnessContext


@dataclass
class BridgeHandle:
    """Returned from :meth:`CapabilityMcpBridge.serve_http`.

    ``port`` is the wildcard-bound ephemeral port the harness container reaches
    through the Docker-bridge host gateway; ``stop`` shuts the server down and
    is registered with ``ContainerHarness``'s ExitStack for reverse-order
    teardown.
    """

    port: int
    stop: Any  # Callable[[], None]


@dataclass
class _DispatchRecord:
    correlation_id: str
    invocation: CapabilityInvocation


def _result_to_payload(result: Any) -> dict[str, Any]:
    """Normalize a CapabilityResult (or dataclass-ish) to JSON-friendly dict."""
    if is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {"output": {}, "success": True, "error": None}
    payload.setdefault("output", {})
    payload.setdefault("success", True)
    payload.setdefault("error", None)
    return payload


def _payload_to_result(capability_name: str, payload: dict[str, Any]) -> CapabilityResult:
    output = payload.get("output") or {}
    if not isinstance(output, dict):
        output = {"value": output}
    return CapabilityResult(
        name=str(payload.get("name") or capability_name),
        output=dict(output),
        success=bool(payload.get("success", True)),
        error=payload.get("error"),
    )


_MCP_TOOL_OUTPUT_MAX_CHARS = 4000
_WORKFLOW_STEP_HEADER = b"x-nsl-workflow-step"
_CURRENT_WORKFLOW_STEP: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nsl_current_workflow_step",
    default=None,
)


def _truncate_text_for_agent(content: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(content) <= max_chars:
        return content, False
    truncated_chars = len(content) - max_chars
    return content[:max_chars] + f"\n...[truncated {truncated_chars} chars]", True


def _truncate_output_for_agent(output: Any, max_chars: int) -> tuple[Any, bool]:
    if isinstance(output, str):
        return _truncate_text_for_agent(output, max_chars)
    if not isinstance(output, dict):
        return output, False

    truncated = False
    out = dict(output)
    for key in ("stdout", "stderr"):
        value = out.get(key)
        if isinstance(value, str):
            out[key], was_truncated = _truncate_text_for_agent(value, max_chars)
            truncated = truncated or was_truncated
    if truncated:
        out["truncated"] = True
    return out, truncated


class _StreamableHttpApp:
    """Trivial ASGI wrapper — delegates to ``StreamableHTTPSessionManager``.

    Inlined from ``mcp.server.fastmcp.server.StreamableHTTPASGIApp`` so the
    bridge has no compile-time dependency on FastMCP.
    """

    def __init__(self, session_manager: Any) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.session_manager.handle_request(scope, receive, send)


class _BearerTokenMiddleware:
    """Minimal ASGI middleware rejecting requests without the right
    ``Authorization: Bearer <token>`` header. Returns 401 as plain text.

    Declared as a class (not a function) so Starlette's dispatch path picks it
    up verbatim as the wrapped ASGI app, no lifespan sharing needed.
    """

    def __init__(self, app: Any, *, token: str) -> None:
        self._app = app
        self._token = token

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            # Lifespan / websocket etc. Pass through unchanged.
            await self._app(scope, receive, send)
            return
        headers = {k: v for k, v in scope.get("headers") or []}
        auth = headers.get(b"authorization", b"").decode("latin-1")
        expected = f"bearer {self._token}".lower()
        if auth.lower() != expected:
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"missing or invalid bearer token\n",
                }
            )
            return
        await self._app(scope, receive, send)


class _WorkflowStepHeaderMiddleware:
    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        headers = {k: v for k, v in scope.get("headers") or []}
        raw_step = headers.get(_WORKFLOW_STEP_HEADER, b"").decode(
            "latin-1", errors="ignore"
        )
        step = raw_step.strip() or None
        token = _CURRENT_WORKFLOW_STEP.set(step)
        try:
            await self._app(scope, receive, send)
        finally:
            _CURRENT_WORKFLOW_STEP.reset(token)


class CapabilityMcpBridge:
    """Per-episode bridge exposing task capabilities to harnesses.

    External-container harnesses call through MCP/HTTP. In-process harnesses
    call :meth:`invoke` directly with a ``CapabilityInvocation``. Both paths
    emit the same paired recorder events.
    """

    def __init__(
        self,
        ctx: "HarnessContext",
        token: str = "",
        *,
        enforce_allowlist: bool = False,
    ) -> None:
        self.ctx = ctx
        self.token = token
        # Native workflow workers use their projected phase capabilities as a
        # dispatch boundary. The flat loop still accepts synthetic invocations.
        self.enforce_allowlist = enforce_allowlist
        self._port: int | None = None
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # --- Tool invocation (core path exercised by tests) ---

    def invoke(self, invocation: CapabilityInvocation) -> CapabilityResult:
        """Dispatch an in-process capability invocation through the bridge.

        Checks the budget ledger before execution. Rejected calls return a
        structured error without logging action/observation events.
        """
        self._raise_if_cancelled(invocation.name)
        if self.enforce_allowlist:
            allowed = {cap.name for cap in self.ctx.prompt_spec.capabilities}
            if invocation.name not in allowed:
                return CapabilityResult(
                    name=invocation.name,
                    output={},
                    success=False,
                    error=(
                        f"capability {invocation.name!r} not permitted in this "
                        f"phase (allowed: {sorted(allowed)})"
                    ),
                )
        rejection = self._check_budget([invocation])
        if rejection is not None:
            return rejection
        record = self._start_invocation(invocation)
        try:
            result = self.ctx.execute_capability(record.invocation)
        except TaskEnvironmentError as exc:
            self._finish_exception(record, exc)
            self.ctx.extra["task_environment_error"] = exc
            raise
        except Exception as exc:  # noqa: BLE001 - normalized into CapabilityResult
            return self._finish_exception(record, exc)
        return self._finish_success(record, result)

    async def _invoke(
        self,
        cap: Capability,
        arguments: dict[str, Any],
    ) -> CapabilityResult:
        """Async MCP adapter over the same bridge-owned dispatch path."""
        invocation = CapabilityInvocation(name=cap.name, input=dict(arguments or {}))
        self._raise_if_cancelled(invocation.name)
        rejection = self._check_budget([invocation])
        if rejection is not None:
            return rejection
        record = self._start_invocation(invocation)
        dispatch_ctx = self.ctx
        step = _CURRENT_WORKFLOW_STEP.get()
        if step:
            dispatch_ctx = self.ctx.with_workflow_step(step)
        try:
            result = await asyncio.to_thread(
                dispatch_ctx.execute_capability,
                record.invocation,
            )
        except TaskEnvironmentError as exc:
            self.ctx.extra["task_environment_error"] = exc
            return self._finish_exception(record, exc)
        except Exception as exc:  # noqa: BLE001 - normalized into CapabilityResult
            return self._finish_exception(record, exc)
        return self._finish_success(record, result)

    def _start_invocation(self, invocation: CapabilityInvocation) -> _DispatchRecord:
        self._raise_if_cancelled(invocation.name)

        clean_invocation = CapabilityInvocation(
            name=invocation.name,
            input=dict(invocation.input or {}),
        )
        record = _DispatchRecord(
            correlation_id=f"mcp-{uuid4().hex[:12]}",
            invocation=clean_invocation,
        )

        # Action event BEFORE execution — structured payload + correlation.
        self.ctx.recorder.log_action(
            category="mcp_capability_call",
            payload={
                "correlation_id": record.correlation_id,
                "capability": clean_invocation.name,
                "invocation": {
                    "name": clean_invocation.name,
                    "input": dict(clean_invocation.input),
                },
            },
        )
        return record

    @contextmanager
    def _workflow_step_budget_scope(self) -> Iterator[None]:
        step = _CURRENT_WORKFLOW_STEP.get()
        if not step:
            yield
            return

        self.ctx.recorder.set_label("last_workflow_step", step)
        constraints = getattr(self.ctx, "constraints", None)
        ledger = getattr(self.ctx, "budget_ledger", None)
        override = (
            constraints.step_overrides.get(step)
            if constraints is not None and ledger is not None
            else None
        )
        if override is None:
            yield
            return

        ledger.push_step(step, override.budgets)
        try:
            self.ctx.recorder.log_state(
                "budget_ledger_step_push",
                {
                    "step": step,
                    "effective_limit": override.budgets.max_task_tool_calls,
                    "used": ledger.snapshot().task_tool_calls_used,
                },
            )
            yield
        finally:
            snap = ledger.pop_step(step)
            self.ctx.recorder.log_state(
                "budget_ledger_step_pop",
                {
                    "step": step,
                    "effective_limit": snap.task_tool_calls_limit,
                    "used": snap.task_tool_calls_used,
                },
            )

    def _raise_if_cancelled(self, capability_name: str) -> None:
        if self.ctx.cancel_event.is_set():
            raise HarnessError(
                f"episode cancelled before capability {capability_name!r}"
            )

    def _finish_exception(
        self,
        record: _DispatchRecord,
        exc: Exception,
    ) -> CapabilityResult:
        error_msg = f"{type(exc).__name__}: {exc}"
        result = CapabilityResult(
            name=record.invocation.name,
            output={},
            success=False,
            error=str(exc),
        )
        # Observation MUST pair even on failure — the recorder invariant
        # survives the exception path.
        self.ctx.recorder.log_observation(
            category="mcp_capability_result",
            payload={
                "correlation_id": record.correlation_id,
                "capability": record.invocation.name,
                "success": False,
                "error": error_msg,
                "result": _result_to_payload(result),
            },
        )
        return result

    def _finish_success(
        self,
        record: _DispatchRecord,
        raw_result: Any,
    ) -> CapabilityResult:
        payload = _result_to_payload(raw_result)
        result = _payload_to_result(record.invocation.name, payload)
        # Observation event AFTER execution — the full structured result.
        self.ctx.recorder.log_observation(
            category="mcp_capability_result",
            payload={
                "correlation_id": record.correlation_id,
                "capability": record.invocation.name,
                "result": _result_to_payload(result),
            },
        )
        return result

    def _check_budget(
        self, invocations: list[CapabilityInvocation]
    ) -> CapabilityResult | None:
        """Return a rejection CapabilityResult if the budget is exhausted, else None.

        Rejected calls do NOT log action/observation events — the rejection
        happens before the invocation enters the bridge dispatch path.
        """
        ledger = getattr(self.ctx, "budget_ledger", None)
        if ledger is None:
            return None
        requests = [ToolCallRequest(capability_name=inv.name) for inv in invocations]
        reservation = ledger.try_consume_tool_calls(requests)
        if reservation.accepted:
            return None
        reason = reservation.rejection_reason or "budget exhausted"
        # One rejection result per invocation (batch always size 1 here for
        # in-process; the MCP async path handles its own batching).
        name = invocations[0].name if invocations else "unknown"
        return CapabilityResult(
            name=name,
            output={},
            success=False,
            error=f"budget exhausted: {reason}",
        )

    def _result_to_tool_payload(self, result: CapabilityResult) -> dict[str, Any]:
        payload = _result_to_payload(result)
        max_chars = int(
            self.ctx.config.get("tool_output_max_chars", _MCP_TOOL_OUTPUT_MAX_CHARS)
        )
        agent_output, was_truncated = _truncate_output_for_agent(
            payload.get("output", ""),
            max_chars,
        )
        if was_truncated:
            self.ctx.recorder.log_warning(
                "tool_output_truncated",
                {
                    "capability": result.name,
                    "max_chars": max_chars,
                },
            )
        tool_payload: dict[str, Any] = {
            "output": agent_output,
            "success": bool(payload.get("success", True)),
            "error": payload.get("error"),
        }
        ledger = getattr(self.ctx, "budget_ledger", None)
        if ledger is not None:
            tool_payload.update(ledger.snapshot().to_agent_dict())
        return tool_payload

    # --- Transport ---

    def _build_mcp_server(self) -> Any:
        """Construct the low-level MCP ``Server`` with list/call handlers
        bound to the episode's advertised capabilities. Extracted so tests can
        exercise the dispatch + schema planes without standing up a full HTTP
        transport.

        The MCP SDK imports stay local because ``mcp.types`` currently pulls
        in uvicorn/starlette transport modules; direct in-process dispatch
        should not pay that import cost.
        """
        from mcp import types as mcp_types
        from mcp.server.lowlevel.server import Server

        ctx = self.ctx
        bridge = self

        # Advertise-by-default: every capability the task declares is exposed
        # as an MCP tool. Tasks own the MCP surface — there is no internal
        # capability list. The previous ``annotations["mcp"]["advertise"]``
        # gate was easy to forget and let mode-shaped capabilities silently
        # leak into the task contract.
        caps_to_advertise = list(ctx.prompt_spec.capabilities)
        caps_by_name = {c.name: c for c in caps_to_advertise}

        mcp_server: Server = Server("nsl")

        @mcp_server.list_tools()
        async def _list_tools() -> list[mcp_types.Tool]:
            return [
                mcp_types.Tool(
                    name=c.name,
                    description=c.description,
                    inputSchema=c.schema
                    or {"type": "object", "additionalProperties": True},
                )
                for c in caps_to_advertise
            ]

        @mcp_server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]):
            cap = caps_by_name.get(name)
            if cap is None:
                return mcp_types.CallToolResult(
                    isError=True,
                    content=[
                        mcp_types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "output": "",
                                    "success": False,
                                    "error": f"unknown capability: {name}",
                                }
                            ),
                        )
                    ],
                )
            with bridge._workflow_step_budget_scope():
                result = await bridge._invoke(cap, arguments)
                return mcp_types.CallToolResult(
                    isError=not result.success,
                    content=[
                        mcp_types.TextContent(
                            type="text",
                            text=json.dumps(bridge._result_to_tool_payload(result)),
                        )
                    ],
                )

        return mcp_server

    def serve_http(self) -> BridgeHandle:
        """Bind an MCP streamable-HTTP server to 0.0.0.0 on an ephemeral
        port, reachable through the Docker bridge gateway. Returns a handle
        with ``port`` + ``stop`` for ExitStack teardown.

        Uses the low-level MCP SDK (``mcp.server.lowlevel.server.Server``)
        wired directly to ``StreamableHTTPSessionManager`` — the same
        transport class FastMCP calls internally. Tools advertise
        ``inputSchema`` declaratively via ``@server.list_tools``; dispatch
        lands at ``@server.call_tool`` with arguments as a plain dict, so there
        is no signature-derived Pydantic model to keep in sync with the schema
        plane.
        """
        # Transport-only imports — kept inline so dispatch-plane tests (which
        # call _build_mcp_server but not serve_http) don't pay the
        # uvicorn/starlette import cost.
        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Route

        mcp_server = self._build_mcp_server()

        session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            stateless=True,
            json_response=True,
        )
        asgi_handler = _StreamableHttpApp(session_manager)

        app = Starlette(
            routes=[Route("/mcp", endpoint=asgi_handler)],
            lifespan=lambda _app: session_manager.run(),
        )
        wrapped = _BearerTokenMiddleware(
            _WorkflowStepHeaderMiddleware(app),
            token=self.token,
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("0.0.0.0", 0))
        sock.listen(128)
        port = sock.getsockname()[1]

        config = uvicorn.Config(
            wrapped,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        ready = threading.Event()
        _serve_error: list[BaseException] = []

        def _serve() -> None:
            # Hand uvicorn our already-bound socket explicitly so it does not
            # re-bind the same host/port during startup.
            original_startup = server.startup

            async def _startup(sockets=None):  # type: ignore[override]
                await original_startup(sockets=sockets)
                ready.set()

            server.startup = _startup  # type: ignore[assignment]
            try:
                asyncio.run(server.serve(sockets=[sock]))
            except Exception as exc:  # noqa: BLE001
                _serve_error.append(exc)
                ready.set()

        thread = threading.Thread(target=_serve, name="mcp-bridge", daemon=True)
        thread.start()
        started = ready.wait(timeout=10)
        if _serve_error:
            sock.close()
            raise HarnessError("MCP bridge server failed to start") from _serve_error[0]
        if not started:
            server.should_exit = True
            sock.close()
            thread.join(timeout=5)
            raise HarnessError("MCP bridge server did not become ready within 10 seconds")

        def _stop() -> None:
            server.should_exit = True
            thread.join(timeout=5)

        self._port = port
        self._server_thread = thread
        return BridgeHandle(port=port, stop=_stop)


__all__ = ["CapabilityMcpBridge", "BridgeHandle"]
