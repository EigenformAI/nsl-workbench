"""HarnessContext — the per-episode handle passed to ``harness.run_episode``.

All handles the harness needs (inference, task, containers, recorder,
cancel signal, config, harness-managed session state) live here. This is
what makes HarnessSpec's ``run_episode`` self-contained.

**Security**: ``private_context`` is deliberately absent. Secrets
(attacker_private_key, upstream_rpc, etc.) never reach harness-authored
code — task methods that need them (``measure_*``, ``finalize_episode``)
receive ``private_context`` directly from the framework, outside the
harness scope.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docker.models.containers import Container

from src.task.types import (
    CapabilityExecutionContext,
    EpisodeConstraints,
    TrustedEpisodeIdentity,
)

if TYPE_CHECKING:
    from src.harness.budget import BudgetLedger
    from src.harness.recorder import EventRecorder
    from src.harness.traced_genner import TracedGenner
    from src.observability.collector import MetricsCollector
    from src.task.base import TaskSpec
    from src.task.types import (
        CapabilityInvocation,
        CapabilityResult,
        EpisodeConstraints,
        TaskPromptSpec,
        Variation,
        Workflow,
    )


def project_step_constraints(
    constraints: EpisodeConstraints | None,
    step_name: str,
    *,
    has_capabilities: bool,
) -> EpisodeConstraints | None:
    if constraints is None:
        return None

    override = constraints.step_overrides.get(step_name)
    if override is None:
        budgets = constraints.budgets
        no_tool_reply = constraints.no_tool_reply
        success = constraints.success
    else:
        budgets = override.budgets
        no_tool_reply = override.no_tool_reply
        success = override.success

    if not has_capabilities and no_tool_reply.retry:
        no_tool_reply = replace(no_tool_reply, retry=False)

    return EpisodeConstraints(
        budgets=budgets,
        no_tool_reply=no_tool_reply,
        success=success,
    )


class HarnessConfigView:
    """Narrowed read-only view of the app config exposed to harnesses.

    A harness sees only its own per-harness config (from
    ``AppConfig.harness.<name>``) plus a small whitelist of framework reads
    (model name for logging, etc.). This prevents harnesses from reaching
    into unrelated config subtrees by accident.
    """

    def __init__(
        self,
        harness_settings: dict[str, Any],
        *,
        model_name: str,
        train_data_save_folder: str,
        code_host_cache_path: str,
    ) -> None:
        self._settings = dict(harness_settings)
        self.model_name = model_name
        self.train_data_save_folder = train_data_save_folder
        self.code_host_cache_path = code_host_cache_path

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)


@dataclass
class HarnessContext:
    """Per-episode handle passed to ``harness.run_episode(ctx=...)``."""

    episode_id: str
    genner: "TracedGenner"
    task: "TaskSpec"
    variation: "Variation"
    prompt_spec: "TaskPromptSpec"
    episode_context: dict[str, Any]
    containers: list[Container]
    agent_container: Container
    host_cache_folder: Path
    config: HarnessConfigView
    metrics: "MetricsCollector | None"
    recorder: "EventRecorder"
    cancel_event: threading.Event
    identity: TrustedEpisodeIdentity | None = None
    constraints: "EpisodeConstraints | None" = None
    budget_ledger: "BudgetLedger | None" = None
    workflow: "Workflow | None" = None
    workflow_step: str | None = None
    harness_session: dict[str, Any] = field(default_factory=dict)
    # Per-worker-slot, harness-managed state. Persists across episodes when
    # the caller reuses the same dict (parallel slots do); single-shot runs
    # use the default fresh dict.
    docker_client: Any = None  # DockerClient — Any to avoid a hard import
    extra: dict[str, Any] = field(default_factory=dict)
    # NOTE: private_context is deliberately absent. See module docstring.

    @property
    def capabilities_view(self) -> tuple[str, ...]:
        return tuple(cap.name for cap in self.prompt_spec.capabilities)

    def with_capability_allowlist(
        self,
        names: set[str],
        *,
        workflow: "Workflow | None" = None,
    ) -> "HarnessContext":
        caps_by_name = {cap.name: cap for cap in self.prompt_spec.capabilities}
        unknown = sorted(names - set(caps_by_name))
        if unknown:
            raise ValueError(f"unknown capability allowlist entries: {unknown}")
        projected = [cap for cap in self.prompt_spec.capabilities if cap.name in names]
        prompt_spec = replace(self.prompt_spec, capabilities=projected)
        return replace(
            self,
            prompt_spec=prompt_spec,
            workflow=self.workflow if workflow is None else workflow,
        )

    def with_step_constraints(self, step_name: str) -> "HarnessContext":
        return replace(
            self,
            constraints=project_step_constraints(
                self.constraints,
                step_name,
                has_capabilities=bool(self.prompt_spec.capabilities),
            ),
        )

    def with_workflow_step(self, step_name: str | None) -> "HarnessContext":
        return replace(self, workflow_step=step_name)

    def execute_capability(
        self,
        invocation: "CapabilityInvocation",
    ) -> "CapabilityResult":
        """Execute a task capability through the framework telemetry seam."""
        self.recorder.bump_counter("tool_calls")
        self.recorder.bump_counter("task_tool_calls")
        self.recorder.bump_counter(f"task_tool_calls.{invocation.name}")
        self.recorder.set_label("last_tool", invocation.name)
        budget_exhaustion = (
            self.budget_ledger.exhausted() if self.budget_ledger is not None else None
        )
        return self.task.execute_capability(
            invocation,
            self.containers,
            self.variation,
            CapabilityExecutionContext(
                episode_id=self.episode_id,
                workflow_step=self.workflow_step,
                episode_context=self.episode_context,
                budget_exhaustion=budget_exhaustion,
                recorder=self.recorder,
                identity=self.identity,
            ),
        )
