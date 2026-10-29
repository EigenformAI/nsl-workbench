"""HarnessProfile ABC — per-harness adapter for :class:`ContainerHarness`.

A profile owns:

- **Config rendering**: write whatever files the harness container needs
  into the per-episode scratch dir (``render_config``).
- **Query rendering**: compose the harness-native input string from the
  task's :class:`TaskPromptSpec` (``render_query``).
- **Transcript recovery**: read the harness's own transcript format back
  from the scratch dir (``read_transcript``).
- **Artifact reconstruction**: turn ``(invocation, result)`` pairs + the
  transcript into an :class:`EpisodeArtifacts` the task can finalize
  against (``to_artifacts``).

Each profile declares a typed ``profile_config_class`` (Pydantic model)
so config typos surface at ``AppConfig`` load rather than at run-episode
time. The loader calls ``profile_cls.profile_config_class.model_validate``
when validating ``ContainerHarnessConfig.profile_config``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from src.harness.tool_contract import ToolCallContractProbe
from src.task.types import (
    Capability,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    TaskPromptSpec,
    Workflow,
)


class MissingTranscriptError(RuntimeError):
    """The harness should have published a transcript but did not."""


class HarnessProfile(ABC):
    name: ClassVar[str]
    profile_config_class: ClassVar[type[BaseModel]]

    def __init__(self, config: BaseModel) -> None:
        self.config = config

    @abstractmethod
    def default_args(self, scratch: Path) -> list[str]:
        """Container command when the config doesn't override it."""
        ...

    @abstractmethod
    def env(self, scratch: Path) -> dict[str, str]:
        """Container environment contributed by the profile."""
        ...

    @abstractmethod
    def render_config(
        self,
        *,
        scratch: Path,
        query: str,
        capabilities: list[Capability],
        inference_url: str,
        mcp_url: str,
        token: str,
        prompt_spec: TaskPromptSpec | None = None,
        workflow: Workflow | None = None,
        constraints: EpisodeConstraints | None = None,
    ) -> None:
        """Write harness-native config files into ``scratch`` (mounted at
        ``/work`` inside the container)."""
        ...

    @abstractmethod
    def read_transcript(self, scratch: Path) -> dict[str, Any] | None:
        """Read the harness's transcript from the scratch dir, or return
        ``None`` if the harness crashed before writing one.

        Raise :class:`MissingTranscriptError` when a transcript was required
        for a completed run but was not published.
        """
        ...

    @abstractmethod
    def to_artifacts(
        self,
        *,
        transcript: dict[str, Any] | None,
        capability_pairs: list[tuple[CapabilityInvocation, CapabilityResult]],
    ) -> EpisodeArtifacts:
        """Reconstruct :class:`EpisodeArtifacts` from recorder state +
        harness transcript."""
        ...

    # --- Overridable defaults ---

    def render_query(
        self,
        prompt_spec: TaskPromptSpec,
        *,
        constraints: EpisodeConstraints | None = None,
    ) -> str:
        """Compose the harness-native query string from the task prompt.

        Default: concatenate system_instruction + environment_context +
        a capability manifest. Profiles override for harness-specific
        preambles (tool-call grammar hints etc.).
        """
        parts: list[str] = []
        if prompt_spec.system_instruction:
            parts.append(prompt_spec.system_instruction)
        body = self.render_query_without_system(prompt_spec, constraints=constraints)
        if body:
            parts.append(body)
        return "\n\n".join(parts)

    def render_query_without_system(
        self,
        prompt_spec: TaskPromptSpec,
        *,
        constraints: EpisodeConstraints | None = None,
    ) -> str:
        """Compose episode context + capabilities without system text."""
        parts: list[str] = []
        if prompt_spec.environment_context:
            parts.append(prompt_spec.environment_context)
        # Advertise-by-default: every declared capability is exposed as MCP.
        if prompt_spec.capabilities:
            caps = "\n".join(
                f"- {c.name}: {c.description}" for c in prompt_spec.capabilities
            )
            parts.append(
                "Capabilities (exposed as MCP tools under the 'nsl' server):\n" + caps
            )
        constraints_block = self.render_constraints_block(constraints)
        if constraints_block:
            parts.append(constraints_block)
        return "\n\n".join(parts)

    def render_constraints_block(
        self,
        constraints: EpisodeConstraints | None,
    ) -> str:
        if constraints is None:
            return ""
        budgets = constraints.budgets
        success = constraints.success
        no_tool = constraints.no_tool_reply

        lines = ["Task constraints:"]
        if budgets.max_task_tool_calls is None:
            lines.append("- task tool calls: no explicit limit")
        else:
            lines.append(f"- task tool calls: at most {budgets.max_task_tool_calls}")
        for name, limit in budgets.max_task_tool_calls_by_name.items():
            lines.append(f"- {name}: at most {limit} task tool calls")
        if budgets.max_task_tool_calls_by_name:
            lines.append(
                "- tool budgets are independent: using up one tool's budget does "
                "not block other tools with remaining budget"
            )
        if budgets.max_llm_turns is None:
            lines.append("- llm turns: no advisory limit")
        else:
            lines.append(f"- llm turns: advisory limit {budgets.max_llm_turns}")

        min_calls = success.min_task_tool_calls_for_success
        if min_calls == 1:
            requirement = "success requires at least 1 task tool call"
        else:
            requirement = f"success requires at least {min_calls} task tool calls"
        if success.terminal_capability_for_success is not None:
            requirement += (
                f" and a {success.terminal_capability_for_success} task tool call"
            )
        lines.append(f"- {requirement}")

        if no_tool.retry:
            lines.append(f"- no-tool replies may be retried up to {no_tool.max_retries} times")
        else:
            lines.append("- no-tool reply retry: disabled")
        return "\n".join(lines)

    supports_no_tool_retry: ClassVar[bool] = False

    def count_llm_turns(self, transcript: dict[str, Any] | None) -> int:
        if not transcript:
            return 0
        return sum(
            1 for m in transcript.get("messages", []) if m.get("role") == "assistant"
        )

    def supports_native_workflow(self, workflow: Workflow) -> bool:
        """Return True when this profile can render and drive ``workflow`` itself."""
        return False

    def tool_capable_step_names(self, workflow: Workflow | None) -> set[str]:
        """Workflow steps where the profile expects tool-call activity.

        Reasoning-only steps explicitly opt out with
        ``inherit_all_capabilities=False`` and ``capabilities=()``.
        """
        if workflow is None:
            return set()
        return {
            step.name
            for step in workflow.steps
            if step.inherit_all_capabilities or bool(step.capabilities)
        }

    def tool_call_contract_probe(self) -> ToolCallContractProbe | None:
        return None


__all__ = ["HarnessProfile", "MissingTranscriptError"]
