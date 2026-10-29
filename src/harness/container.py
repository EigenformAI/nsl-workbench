"""ContainerHarness — drives a Docker-packaged external harness.

Responsibilities:

1. Mint a per-episode scratch dir (host-side), per-episode bearer token.
2. Bring up two host servers bound to 0.0.0.0 on ephemeral ports:
   the :class:`OpenAiShim` (inference) and :class:`CapabilityMcpBridge`
   (capabilities).
3. Ask the profile to render its native config against those two URLs.
4. Launch the harness container, passing the scratch dir bind-mounted at
   ``/work``; the container reaches the host-side servers through the
   Docker-bridge host gateway.
5. Wait for the container to exit (cooperatively honoring
   ``cancel_event``); recover the transcript; reconstruct artifacts.
6. Tear everything down in reverse order on every exit path (success,
   exception, cancellation).

The tests patch :func:`_serve_host_shim`,
:func:`CapabilityMcpBridge.serve_http`, :func:`resolve_profile`,
and :func:`_docker_host_gateway` so the driver's cleanup logic is
exercised without a real Docker daemon, real listener, or a real profile.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from src.framework.capability_bridge import BridgeHandle, CapabilityMcpBridge
from src.harness.base import HarnessError, HarnessSpec
from src.harness.context import HarnessContext, project_step_constraints
from src.harness.context_compaction import ContextCompactionSettings
from src.harness.openai_shim import OpenAiShim
from src.harness.profiles import resolve_profile
from src.harness.profiles.base import MissingTranscriptError
from src.harness.recorder import EventRecorder
from src.harness.transcript import HarnessTranscript, TerminationCategory
from src.task.types import EpisodeArtifacts, Workflow
from src.typing.config import ContainerHarnessConfig

_NSL_EVENT_PREFIX = "[nsl-event] "


@dataclass
class _ShimHandle:
    """Handle returned from :func:`_serve_host_shim` — mirrors :class:`BridgeHandle`."""

    port: int
    stop: Any  # Callable[[], None]


@dataclass
class _Termination:
    reason: str
    category: TerminationCategory


def _serve_host_shim(app: Any) -> _ShimHandle:
    """Bind a FastAPI app to 0.0.0.0 on an ephemeral port in a daemon
    thread. Returns once the server signals readiness.

    Not exercised by unit tests — ``test_container_harness_cleanup`` monkey-
    patches this function to inject a spy handle.
    """
    import socket

    import uvicorn

    # Must bind on all interfaces (not 127.0.0.1) — the harness container
    # reaches us via the docker-bridge gateway (172.17.0.1 on Linux), which
    # arrives on the host's docker0 interface and is NOT accepted by a
    # loopback-only listener. The shim checks the bearer token on completion
    # routes; its debugging documentation/schema routes remain unauthenticated.
    # The MCP bridge checks the token in HTTP middleware.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sock.listen(128)
    port = sock.getsockname()[1]

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    ready = threading.Event()

    def _serve() -> None:
        import asyncio

        original_startup = server.startup

        async def _startup(sockets=None):  # type: ignore[override]
            await original_startup(sockets=sockets)
            ready.set()

        server.startup = _startup  # type: ignore[assignment]
        try:
            asyncio.run(server.serve(sockets=[sock]))
        except Exception:  # noqa: BLE001
            ready.set()

    thread = threading.Thread(target=_serve, name="openai-shim", daemon=True)
    thread.start()
    ready.wait(timeout=10)

    def _stop() -> None:
        server.should_exit = True
        thread.join(timeout=5)

    return _ShimHandle(port=port, stop=_stop)


def _docker_host_gateway(docker_client: Any, network_mode: str = "bridge") -> str:
    """Resolve the address the harness container uses to reach the
    framework shim + MCP bridge listening on the host.

    - ``network_mode="host"``: container shares the host netns; loopback
      reaches the listeners directly. No NAT, no host-firewall traversal.
    - ``network_mode="bridge"``: query the default bridge's Gateway via
      the Docker API. This varies by host (172.17.0.1 is the factory
      default, but ``default-address-pools`` and collision avoidance
      frequently push it to 10.x/172.x/192.x). Bridge mode additionally
      requires arbitrary high ports to be reachable from the bridge
      subnet — many Linux hosts drop those by default (only well-known
      ports like 22 go through), which turns TCP connects into silent
      timeouts. Prefer host mode when that's biting.

    Overridable via monkeypatch in tests.
    """
    if network_mode == "host":
        return "127.0.0.1"
    try:
        bridge = docker_client.networks.get("bridge")
        for cfg in bridge.attrs.get("IPAM", {}).get("Config", []) or []:
            gw = cfg.get("Gateway")
            if gw:
                return gw
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"bridge gateway lookup failed: {exc}; falling back to 172.17.0.1"
        )
    return "172.17.0.1"


def _force_remove_quietly(container: Any) -> None:
    """Remove a container; log and swallow exceptions.

    Teardown of the shim + bridge depends on this NOT raising when the
    Docker daemon is unreachable — otherwise a transient docker outage
    would leak the servers.
    """
    try:
        container.remove(force=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"container.remove failed: {type(exc).__name__}: {exc}")


def _wait_with_cancel(
    container: Any,
    cancel_event: threading.Event,
    max_wall: int,
) -> _Termination:
    """Wait for the container to exit, honoring ``cancel_event`` and
    ``max_wall``. On cancellation or wall-clock expiry the container is
    killed (``kill()``) so the blocking ``wait()`` returns.

    Not exercised by unit tests — ``test_container_harness_cancel`` monkey-
    patches this to trigger the cancel path deterministically.
    """
    done: dict[str, Any] = {}

    def _waiter() -> None:
        try:
            result = container.wait()
            done["status_code"] = result.get("StatusCode", 0)
        except Exception as exc:  # noqa: BLE001
            done["error"] = f"{type(exc).__name__}: {exc}"

    waiter = threading.Thread(target=_waiter, name="container-wait", daemon=True)
    waiter.start()

    deadline = time.monotonic() + max_wall
    while waiter.is_alive():
        if cancel_event.is_set():
            try:
                container.kill()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"container.kill failed: {exc}")
            waiter.join(timeout=5)
            return _Termination(reason="cancel_event set", category="wall_clock")
        if time.monotonic() >= deadline:
            try:
                container.kill()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"container.kill failed: {exc}")
            waiter.join(timeout=5)
            return _Termination(
                reason=f"wall clock expired ({max_wall}s)",
                category="wall_clock",
            )
        time.sleep(0.1)

    if "error" in done:
        return _Termination(
            reason=f"wait error: {done['error']}",
            category="harness_error",
        )
    status = int(done.get("status_code", 0))
    if status == 0:
        return _Termination(reason="exited cleanly", category="success")
    return _Termination(
        reason=f"container exited with status {status}",
        category="agent_failure",
    )


def _read_container_logs(container: Any) -> str:
    try:
        raw = container.logs(stdout=True, stderr=True, tail="all")
    except Exception as exc:  # noqa: BLE001
        return f"<log read failed: {exc}>"
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _tail_logs(container: Any, *, max_bytes: int) -> str:
    return _truncate_tail(_read_container_logs(container), max_bytes=max_bytes)


def _truncate_tail(text: str, *, max_bytes: int) -> str:
    if len(text) <= max_bytes:
        return text
    return text[-max_bytes:]


def _scrape_nsl_events(log_text: str) -> list[dict[str, Any]]:
    """Extract structured [nsl-event] {...json...} lines from container logs.

    Emitted by docker/ms-agent/run.py::_emit so the host can mirror the
    in-image workflow lifecycle onto the recorder. Malformed lines are
    skipped silently; a partial scrape is better than aborting the
    whole episode summary.
    """
    events: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        idx = line.find(_NSL_EVENT_PREFIX)
        if idx < 0:
            continue
        payload = line[idx + len(_NSL_EVENT_PREFIX) :].strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("event"), str):
            events.append(parsed)
    return events


def _replay_nsl_events(
    events: Iterable[dict[str, Any]],
    recorder: EventRecorder,
) -> None:
    """Mirror in-image workflow telemetry onto the host recorder.

    Event names align with src/harness/workflow_driver.py so consumers
    of the recorder stream see the same shapes for native-workflow
    profiles (ms-agent) and framework-driven profiles.
    """
    for event in events:
        name = event.get("event")
        payload = {k: v for k, v in event.items() if k != "event"}
        if name == "workflow_step_enter":
            step = payload.get("step")
            if isinstance(step, str):
                recorder.set_label("last_workflow_step", step)
            recorder.log_decision("workflow_step_enter", payload)
        elif name == "workflow_step_exit":
            recorder.log_decision("workflow_step_exit", payload)
        elif name == "workflow_finished":
            recorder.log_state("workflow_finished", payload)
        else:
            recorder.log_state("nsl_event", {"event": name, **payload})


class ContainerHarness(HarnessSpec):
    """Drives an external-container harness (ms-agent first; hermes,
    others via their own :class:`HarnessProfile`).

    Exactly one harness container per episode; the shim + bridge are also
    per-episode. Nothing is shared across episodes.
    """

    name = "container"
    description = "Drives a Docker-packaged external harness over OpenAI + MCP"

    def __init__(self, harness_config: dict[str, Any]) -> None:
        super().__init__(harness_config)
        self.config = ContainerHarnessConfig.model_validate(harness_config)
        self.profile = resolve_profile(self.config.profile, self.config.profile_config)
        self._ctx: HarnessContext | None = None

    def telemetry(self) -> dict[str, str]:
        if self._ctx is None:
            return {}
        counters = self._ctx.recorder.snapshot_counters()
        labels = self._ctx.recorder.snapshot_labels()
        return self.telemetry_from_recorder_snapshot(counters, labels)

    def telemetry_from_recorder_snapshot(
        self,
        counters: dict[str, Any],
        labels: dict[str, str],
    ) -> dict[str, str]:
        telemetry: dict[str, str] = {}
        for key in self.telemetry_columns():
            if key in counters:
                telemetry[key] = str(counters[key])
            elif key in labels:
                telemetry[key] = labels[key]
        return telemetry

    def telemetry_columns(self) -> list[str]:
        return ["turns", "last_tool", "tool_calling_no_calls"]

    def failure_extras(self) -> dict[str, Any]:
        telemetry = self.telemetry()
        return {"telemetry": telemetry} if telemetry else {}

    # --- HarnessSpec ---

    def run_workflow(
        self, workflow: Workflow, ctx: HarnessContext
    ) -> HarnessTranscript:
        if self.profile.supports_native_workflow(workflow):
            return self.run_episode(ctx=replace(ctx, workflow=workflow))
        return super().run_workflow(workflow, ctx)

    def run_episode(self, *, ctx: HarnessContext) -> HarnessTranscript:
        self._ctx = ctx
        try:
            with ExitStack() as stack:
                # Docker SDK requires absolute paths for bind-mount keys — a
                # relative key like "./code/.../harness-ep_..." is parsed as a
                # named-volume reference, which then fails validation on "/".
                # host_cache_folder comes from config (code_host_cache_path) and
                # may be relative; resolve() makes it absolute against CWD.
                scratch = (
                    Path(ctx.host_cache_folder) / f"harness-{ctx.episode_id}"
                ).resolve()
                scratch.mkdir(parents=True, exist_ok=True)

                token = secrets.token_urlsafe(24)

                # Servers first — the container must never boot against a
                # not-yet-listening endpoint. Each handle is registered with
                # the ExitStack BEFORE the next resource is acquired so
                # reverse-order teardown holds under any exception.
                shim = OpenAiShim(
                    ctx.genner,
                    token=token,
                    episode_id=ctx.episode_id,
                    recorder=ctx.recorder,
                    context_compaction=ContextCompactionSettings(
                        enabled=self.config.context_compaction_enabled,
                        trigger_tokens=self.config.context_compaction_trigger_tokens,
                        target_tokens=self.config.context_compaction_target_tokens,
                        keep_recent_tool_outputs=(
                            self.config.context_compaction_keep_recent_tool_outputs
                        ),
                        keep_recent_assistant_reasoning=(
                            self.config.context_compaction_keep_recent_assistant_reasoning
                        ),
                        chars_per_token=self.config.context_compaction_chars_per_token,
                    ),
                )
                shim_handle = _serve_host_shim(shim.app)
                stack.callback(shim_handle.stop)

                bridge = CapabilityMcpBridge(ctx, token=token)
                bridge_handle: BridgeHandle = bridge.serve_http()
                stack.callback(bridge_handle.stop)

                host_gateway = _docker_host_gateway(
                    ctx.docker_client, network_mode=self.config.network_mode
                )
                inference_url = f"http://{host_gateway}:{shim_handle.port}/v1"
                mcp_url = f"http://{host_gateway}:{bridge_handle.port}/mcp"
                native_workflow = (
                    ctx.workflow
                    if ctx.workflow is not None
                    and self.profile.supports_native_workflow(ctx.workflow)
                    else None
                )

                _validate_no_tool_retry_supported(
                    ctx,
                    self.profile,
                    profile_name=self.config.profile,
                    workflow=native_workflow,
                )

                query = self.profile.render_query(
                    ctx.prompt_spec,
                    constraints=ctx.constraints,
                )
                self.profile.render_config(
                    scratch=scratch,
                    query=query,
                    capabilities=list(ctx.prompt_spec.capabilities),
                    inference_url=inference_url,
                    mcp_url=mcp_url,
                    token=token,
                    prompt_spec=ctx.prompt_spec,
                    workflow=native_workflow,
                    constraints=ctx.constraints,
                )

                ctx.recorder.log_state(
                    "container_harness_start",
                    {
                        "image": self.config.image,
                        "profile": self.config.profile,
                        "scratch_dir": str(scratch),
                        "shim_port": shim_handle.port,
                        "bridge_port": bridge_handle.port,
                    },
                )

                container = ctx.docker_client.containers.run(
                    image=self.config.image,
                    entrypoint=self.config.entrypoint,
                    command=(
                        self.config.args
                        if self.config.args is not None
                        else self.profile.default_args(scratch)
                    ),
                    environment={
                        **self.config.env,
                        **self.profile.env(scratch),
                    },
                    volumes={str(scratch): {"bind": "/work", "mode": "rw"}},
                    user=str(os.getuid()),
                    network_mode=self.config.network_mode,
                    extra_hosts={"host.docker.internal": "host-gateway"},
                    mem_limit=self.config.mem_limit,
                    detach=True,
                )
                stack.callback(_force_remove_quietly, container)

                termination = _wait_with_cancel(
                    container,
                    ctx.cancel_event,
                    max_wall=self.config.max_wall_seconds,
                )
                graceful_failure_detail = getattr(
                    shim, "graceful_agent_failure_detail", None
                )
                if not isinstance(graceful_failure_detail, str):
                    graceful_failure_detail = None

                # If the inference endpoint went away mid-episode, the
                # agent container would have exited non-zero from a 502 on
                # /v1/chat/completions — which _wait_with_cancel reports as
                # `agent_failure`. That's wrong: the agent didn't fail, the
                # backend did. Give inference outages their own category so
                # endpoint-level routing handles quarantine/failover instead
                # of tripping the per-slot harness breaker.
                if shim.inference_unavailable_detail is not None:
                    termination = _Termination(
                        reason=(
                            "inference endpoint unavailable: "
                            f"{shim.inference_unavailable_detail}"
                        ),
                        category="endpoint_unavailable",
                    )
                elif shim.inference_timeout_detail is not None:
                    # A request timeout (decode starvation) is the backend's
                    # fault, not the agent's — but it is NOT an outage. It gets
                    # its own benign, non-quarantining category so a single
                    # timeout cannot breach the single-endpoint capacity floor
                    # and abort the run (quarantine is keyed on
                    # endpoint_unavailable; see the parallel worker loop).
                    termination = _Termination(
                        reason=(
                            "inference request timed out: "
                            f"{shim.inference_timeout_detail}"
                        ),
                        category="inference_timeout",
                    )
                elif graceful_failure_detail is not None:
                    termination = _Termination(
                        reason=graceful_failure_detail,
                        category="agent_failure",
                    )

                budget_exhaustion = (
                    ctx.budget_ledger.exhausted()
                    if ctx.budget_ledger is not None
                    else None
                )
                if (
                    budget_exhaustion is not None
                    and graceful_failure_detail is None
                ):
                    termination = _Termination(
                        reason=f"budget exhausted: {budget_exhaustion.kind}",
                        category="budget_exhausted",
                    )

                # Read container logs once and split the work: replay any
                # [nsl-event] lines into the recorder (so native-workflow
                # profiles surface workflow_step_enter/exit like the
                # host-side WorkflowDriver does), and keep a truncated tail
                # for the episode transcript. Done BEFORE read_transcript so
                # the events survive even when the profile's transcript
                # backstop raises (e.g., inference outage with workflow.yaml
                # rendered but no per-step JSON written).
                full_logs = _read_container_logs(container)
                _replay_nsl_events(_scrape_nsl_events(full_logs), ctx.recorder)
                stderr_tail = _truncate_tail(full_logs, max_bytes=64_000)

                try:
                    transcript_raw = self.profile.read_transcript(scratch)
                except MissingTranscriptError:
                    if termination.category not in {
                        "endpoint_unavailable",
                        "inference_timeout",
                    } and graceful_failure_detail is None:
                        raise
                    transcript_raw = None
                self._record_tool_calling_no_calls(
                    ctx=ctx,
                    workflow=native_workflow,
                    transcript=transcript_raw,
                )

                try:
                    capability_pairs = ctx.recorder.capability_pairs()
                except HarnessError as exc:
                    # An unpaired recorder event is a bridge bug; surface it but
                    # still let teardown run. Degrading to an empty pairs list
                    # preserves the best-effort artifact reconstruction.
                    logger.warning(f"capability_pairs failed: {exc}")
                    capability_pairs = []

                artifacts = self.profile.to_artifacts(
                    transcript=transcript_raw,
                    capability_pairs=capability_pairs,
                )
                if not isinstance(artifacts, EpisodeArtifacts):
                    # Allow the mocked-profile tests to return a MagicMock that
                    # quacks like artifacts; translate when necessary.
                    invs = list(getattr(artifacts, "capability_invocations", []) or [])
                    results = list(getattr(artifacts, "capability_results", []) or [])
                    final = getattr(artifacts, "final_response", None)
                    artifacts = EpisodeArtifacts(
                        capability_invocations=invs,
                        capability_results=results,
                        final_response=final,
                    )

                ctx.recorder.log_state(
                    "container_harness_stop",
                    {
                        "termination_reason": termination.reason,
                        "termination_category": termination.category,
                    },
                )

                extra = {
                    "stderr_tail": stderr_tail,
                    "scratch_dir": str(scratch),
                }
                if isinstance(transcript_raw, dict):
                    last_step = transcript_raw.get("last_workflow_step")
                    if isinstance(last_step, str):
                        extra["last_workflow_step"] = last_step
                if "last_workflow_step" not in extra:
                    last_step = ctx.recorder.snapshot_labels().get("last_workflow_step")
                    if last_step:
                        extra["last_workflow_step"] = last_step
                if budget_exhaustion is not None:
                    extra["budget_exhausted_kind"] = budget_exhaustion.kind
                if graceful_failure_detail is not None:
                    extra["agent_failure_kind"] = "output_length_retries_exhausted"

                return HarnessTranscript(
                    artifacts=artifacts,
                    llm_turns=self.profile.count_llm_turns(transcript_raw),
                    termination_reason=termination.reason,
                    termination_category=termination.category,
                    extra=extra,
                )
            # Unreachable — ExitStack returns via the ``return`` above.
            raise RuntimeError("ContainerHarness: ExitStack exited without return")
        finally:
            self._ctx = None

    def _record_tool_calling_no_calls(
        self,
        *,
        ctx: HarnessContext,
        workflow: Workflow | None,
        transcript: dict[str, Any] | None,
    ) -> None:
        if workflow is None or not transcript:
            return
        if not ctx.prompt_spec.capabilities:
            return

        tool_capable = self.profile.tool_capable_step_names(workflow)
        try:
            tool_capable_names = set(tool_capable or ())
        except TypeError:
            return
        if not tool_capable_names:
            return

        by_step = transcript.get("workflow")
        if not isinstance(by_step, dict):
            return

        for step in workflow.topological_order():
            if step.name not in tool_capable_names:
                continue
            step_payload = by_step.get(step.name)
            if not isinstance(step_payload, dict):
                continue
            try:
                tool_end_count = int(step_payload.get("tool_end_count") or 0)
            except (TypeError, ValueError):
                tool_end_count = 0
            if tool_end_count > 0:
                continue

            first_assistant = step_payload.get("first_assistant_content")
            if not isinstance(first_assistant, str):
                first_assistant = ""
                messages = step_payload.get("messages") or []
                if isinstance(messages, list):
                    for message in messages:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        if message.get("role") == "assistant" and isinstance(
                            content, str
                        ):
                            first_assistant = content
                            break
            ctx.recorder.bump_counter("tool_calling_no_calls")
            ctx.recorder.log_decision(
                "tool_calling_no_calls",
                {"step": step.name, "first_chars": first_assistant[:120]},
            )


def _validate_no_tool_retry_supported(
    ctx: HarnessContext,
    profile: Any,
    *,
    profile_name: str,
    workflow: Workflow | None = None,
) -> None:
    if ctx.constraints is None or getattr(profile, "supports_no_tool_retry", False):
        return

    if workflow is None:
        if ctx.constraints.no_tool_reply.retry and ctx.prompt_spec.capabilities:
            raise HarnessError(
                f"profile {profile_name!r} does not support no-tool retry"
            )
        return

    for step in workflow.topological_order():
        has_capabilities = (
            bool(ctx.prompt_spec.capabilities)
            if step.inherit_all_capabilities
            else bool(step.capabilities)
        )
        effective = project_step_constraints(
            ctx.constraints,
            step.name,
            has_capabilities=has_capabilities,
        )
        if effective is not None and effective.no_tool_reply.retry and has_capabilities:
            raise HarnessError(
                f"profile {profile_name!r} does not support no-tool retry"
            )


__all__ = ["ContainerHarness"]
