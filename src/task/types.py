"""Data types for the task abstraction layer.

Tasks produce these; the framework (and harnesses) consume them. The types
are harness-neutral — a task declares `capabilities` and emits
`CapabilityInvocation`s without knowing whether the harness is mode-based,
ReAct-style, or a foreign binary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

if TYPE_CHECKING:
    from src.harness.budget import BudgetExhaustion


@dataclass
class Variation:
    """Base class for environment variations.

    The framework only reads ``name`` and ``description``. Tasks typically
    subclass this to carry task-specific payloads.
    """

    name: str
    description: str


@dataclass
class Capability:
    """A thing the agent can do, declared by the task in harness-neutral form.

    The harness translates capabilities into its native vocabulary (mode,
    ReAct tool, OpenAI function spec, etc.).

    Intent flags (``runs_code``, ``publishes_metric``) are framework-wide
    and read by harnesses that care. ``annotations`` remains available for
    future, harness-specific extension points, but the shipped harnesses no
    longer use it for mode or MCP advertisement plumbing. Working memory is a
    harness concern, not a framework concern.
    """

    name: str
    description: str
    schema: dict[str, Any] | None = None
    timeout_seconds: int | None = None
    runs_code: bool = False
    publishes_metric: bool = False
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetConstraints:
    """Task-declared budget limits for one episode.

    ``max_task_tool_calls`` is hard-enforced at ``CapabilityMcpBridge``.
    ``max_llm_turns`` is advisory; tracked in the ledger, surfaced to agents,
    but never used to block a model call.
    ``max_task_tool_calls_by_name`` sub-allocates the global budget per capability;
    a named call decrements BOTH the global counter and the named counter.
    """

    max_task_tool_calls: int | None = None
    max_task_tool_calls_by_name: Mapping[str, int] = field(default_factory=dict)
    max_llm_turns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_task_tool_calls_by_name",
            MappingProxyType(dict(self.max_task_tool_calls_by_name)),
        )


@dataclass(frozen=True)
class NoToolReplyPolicy:
    """Retry behavior for assistant responses that contain no tool calls.

    Success or failure of a no-tool reply is determined post-hoc by
    ``SuccessConstraints``, not by this policy. This policy only controls
    whether the harness injects a corrective prompt and retries.
    """

    retry: bool = False
    max_retries: int = 0
    retry_instruction: str | None = None


@dataclass(frozen=True)
class SuccessConstraints:
    """Post-hoc success classification rules applied in ``_finalize_reward()``.

    Default of 1 for ``min_task_tool_calls_for_success`` preserves the
    existing zero-tool success override in episode.py. Tasks with no required
    tools (e.g. a planning step) must set 0 explicitly.
    """

    min_task_tool_calls_for_success: int = 1
    terminal_capability_for_success: str | None = None


@dataclass(frozen=True)
class StepConstraints:
    """Constraints applied to a single ``WorkflowStep``.

    Step counters share the episode-level ledger and do not reset between
    steps. A step override changes which limit applies; the used counter
    keeps accumulating across the whole episode.
    """

    budgets: BudgetConstraints = field(default_factory=BudgetConstraints)
    no_tool_reply: NoToolReplyPolicy = field(default_factory=NoToolReplyPolicy)
    success: SuccessConstraints = field(default_factory=SuccessConstraints)


@dataclass(frozen=True)
class EpisodeConstraints:
    """Task-owned, harness-neutral episode constraints.

    Computed once per episode by ``TaskSpec.episode_constraints()`` and stored
    on ``HarnessContext``. Harnesses and the framework read constraints without
    parsing prompt strings; enforcement lives at bridge chokepoints.

    ``step_overrides`` replaces episode-level values for the named step.
    Override entries cannot themselves nest further step_overrides.
    """

    budgets: BudgetConstraints = field(default_factory=BudgetConstraints)
    no_tool_reply: NoToolReplyPolicy = field(default_factory=NoToolReplyPolicy)
    success: SuccessConstraints = field(default_factory=SuccessConstraints)
    step_overrides: Mapping[str, StepConstraints] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_overrides",
            MappingProxyType(dict(self.step_overrides)),
        )


@dataclass
class CapabilityInvocation:
    """An instance of the agent invoking a declared capability.

    Produced by :meth:`TaskSpec.parse_response` — the task extracts
    structured capability invocations from the agent's raw response.
    """

    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityExecutionContext:
    """Trusted framework metadata for one capability execution.

    ``CapabilityInvocation`` remains agent-supplied input only. The framework
    owns this object, so tasks can enforce workflow-step and live hand-off
    rules without trusting fields the agent controls.

    ``recorder`` is the live episode :class:`EventRecorder` (typed ``Any``
    here to avoid a hard import cycle). Tasks may use it to emit synthetic
    inference rows that would otherwise be invisible to the SFT export.
    Optional so unit tests can construct contexts without recorder plumbing.
    """

    episode_id: str
    workflow_step: str | None
    episode_context: dict[str, Any]
    budget_exhaustion: BudgetExhaustion | None = None
    recorder: Any = None
    identity: TrustedEpisodeIdentity | None = None


@dataclass
class CapabilityResult:
    """The outcome of executing a CapabilityInvocation.

    Produced by :meth:`TaskSpec.execute_capability`. For invocations that
    capture data without side effects (metrics readouts, etc.), the task
    typically returns a success result with the parsed payload echoed to
    ``output``.
    """

    name: str
    output: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a harness-neutral workflow.

    The task declares step intent only. Harnesses own intra-episode state such
    as the LLM conversation, container, filesystem, and MCP sessions.
    """

    name: str
    prompt: str
    description: str = ""
    capabilities: tuple[str, ...] = ()
    inherit_all_capabilities: bool = True
    next_steps: tuple[str, ...] = ()
    is_entry: bool = False
    context_mode: Literal["inherit", "isolated"] = "inherit"
    on_error: str | None = None
    terminator_capabilities: tuple[str, ...] = ()
    """Capabilities whose invocation signals step completion.

    Harness profiles that support native early-exit (e.g. NAT return_direct)
    map this to their mechanism. Profiles that do not support it ignore the
    field; it has no effect on execution.
    """


@dataclass(frozen=True)
class Workflow:
    """Forward-only workflow plan evaluated once at episode start.

    v1 deliberately restricts execution to chains plus declarative error
    recovery. Kahn's algorithm is still used so future fan-in relaxation does
    not require replacing the topology helper.
    """

    steps: tuple[WorkflowStep, ...]

    @property
    def step_names(self) -> set[str]:
        return {step.name for step in self.steps}

    @property
    def entry_step(self) -> WorkflowStep | None:
        if not self.steps:
            return None
        explicit = [step for step in self.steps if step.is_entry]
        if len(explicit) > 1:
            raise ValueError("workflow has multiple entry steps")
        if explicit:
            return explicit[0]
        roots = self._root_steps()
        if len(roots) != 1:
            raise ValueError(
                "workflow requires exactly one entry step; got "
                f"{[step.name for step in roots]}"
            )
        return roots[0]

    def topological_order(self) -> list[WorkflowStep]:
        steps_by_name = self._steps_by_name()
        all_indegree = {name: 0 for name in steps_by_name}
        all_outgoing: dict[str, list[str]] = {}
        for name, step in steps_by_name.items():
            targets = list(step.next_steps)
            if step.on_error is not None:
                targets.append(step.on_error)
            all_outgoing[name] = targets
            for target in targets:
                if target not in steps_by_name:
                    raise ValueError(
                        f"workflow step {name!r} references unknown step {target!r}"
                    )
                all_indegree[target] += 1

        all_queue = deque(name for name, degree in all_indegree.items() if degree == 0)
        visited_count = 0
        while all_queue:
            name = all_queue.popleft()
            visited_count += 1
            for child in all_outgoing[name]:
                all_indegree[child] -= 1
                if all_indegree[child] == 0:
                    all_queue.append(child)
        if visited_count != len(self.steps):
            raise ValueError("workflow contains a cycle")

        indegree = {name: 0 for name in steps_by_name}
        outgoing = {name: list(step.next_steps) for name, step in steps_by_name.items()}
        for step in self.steps:
            for next_name in step.next_steps:
                if next_name not in steps_by_name:
                    raise ValueError(
                        f"workflow step {step.name!r} references unknown "
                        f"next step {next_name!r}"
                    )
                indegree[next_name] += 1

        queue = deque(name for name, degree in indegree.items() if degree == 0)
        ordered: list[WorkflowStep] = []
        while queue:
            name = queue.popleft()
            ordered.append(steps_by_name[name])
            for child in outgoing[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(self.steps):
            raise ValueError("workflow contains a cycle")
        return ordered

    def validate(self, capability_names: set[str]) -> None:
        if not self.steps:
            raise ValueError("workflow must declare at least one step")
        steps_by_name = self._steps_by_name()
        incoming: dict[str, int] = {name: 0 for name in steps_by_name}

        for step in self.steps:
            if step.context_mode not in {"inherit", "isolated"}:
                raise ValueError(
                    f"workflow step {step.name!r} has unknown context_mode "
                    f"{step.context_mode!r}"
                )
            if step.inherit_all_capabilities and step.capabilities:
                raise ValueError(
                    f"workflow step {step.name!r}: inherit_all_capabilities=True "
                    "requires capabilities=()"
                )
            if not step.inherit_all_capabilities:
                unknown = sorted(set(step.capabilities) - capability_names)
                if unknown:
                    raise ValueError(
                        f"workflow step {step.name!r} references unknown "
                        f"capabilities: {unknown}"
                    )
            # A terminator must be invokable through the step's own allowlist.
            allow = (
                capability_names
                if step.inherit_all_capabilities
                else set(step.capabilities)
            )
            stray_terminators = sorted(set(step.terminator_capabilities) - allow)
            if stray_terminators:
                raise ValueError(
                    f"workflow step {step.name!r} declares terminator_capabilities "
                    f"outside its allowlist: {stray_terminators}"
                )
            if len(step.next_steps) > 1:
                raise ValueError(
                    f"workflow step {step.name!r} has fan-out; v1 supports "
                    "at most one next step"
                )
            for next_name in step.next_steps:
                if next_name not in steps_by_name:
                    raise ValueError(
                        f"workflow step {step.name!r} references unknown "
                        f"next step {next_name!r}"
                    )
                incoming[next_name] += 1
            if step.on_error is not None and step.on_error not in steps_by_name:
                raise ValueError(
                    f"workflow step {step.name!r} references unknown "
                    f"on_error step {step.on_error!r}"
                )

        fan_in = sorted(name for name, degree in incoming.items() if degree > 1)
        if fan_in:
            raise ValueError(f"workflow has fan-in at steps: {fan_in}")

        # Raises for ambiguous or duplicate entry declarations.
        entry = self.entry_step
        self.topological_order()
        if entry is not None:
            reachable = self._reachable_from(entry.name, include_on_error=True)
            missing = sorted(set(steps_by_name) - reachable)
            if missing:
                raise ValueError(f"workflow has unreachable steps: {missing}")

    def _steps_by_name(self) -> dict[str, WorkflowStep]:
        steps_by_name: dict[str, WorkflowStep] = {}
        duplicates: set[str] = set()
        for step in self.steps:
            if step.name in steps_by_name:
                duplicates.add(step.name)
            steps_by_name[step.name] = step
        if duplicates:
            raise ValueError(f"workflow has duplicate step names: {sorted(duplicates)}")
        return steps_by_name

    def _root_steps(self) -> list[WorkflowStep]:
        steps_by_name = self._steps_by_name()
        incoming = {name: 0 for name in steps_by_name}
        for step in self.steps:
            for next_name in step.next_steps:
                if next_name in incoming:
                    incoming[next_name] += 1
            if step.on_error in incoming:
                incoming[step.on_error] += 1
        return [step for step in self.steps if incoming[step.name] == 0]

    def _reachable_from(self, start: str, *, include_on_error: bool) -> set[str]:
        steps_by_name = self._steps_by_name()
        seen: set[str] = set()
        stack = [start]
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            step = steps_by_name[name]
            stack.extend(step.next_steps)
            if include_on_error and step.on_error is not None:
                stack.append(step.on_error)
        return seen


@dataclass
class TaskPromptSpec:
    """Harness-neutral prompt content produced by the task.

    Fields:
      system_instruction: Task-wide system prompt (agent persona, goal).
      environment_context: Episode-specific context (addresses, paths, etc.).
      capabilities: What the agent can do, harness-neutral.

    Phase 2 deletes ``extras``: per-harness prompt content lives in config
    under ``[harness.<name>.*]``, not in the task output.
    """

    system_instruction: str
    environment_context: str = ""
    capabilities: list[Capability] = field(default_factory=list)


@dataclass
class PopulationResult:
    """Result of populating a single container with a variation."""

    container_id: str
    variation_name: str
    description: str
    success: bool
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationOutcome:
    """Result of populate(): per-container results plus episode-scoped context."""

    results: list[PopulationResult]
    episode_context: dict[str, Any] = field(default_factory=dict)
    private_context: dict[str, Any] | None = None


@dataclass
class EpisodeArtifacts:
    """Agent-produced data collected by the framework during the episode.

    Invocations and results are aligned — ``capability_results[i]`` is the
    outcome of ``capability_invocations[i]``.
    """

    capability_invocations: list[CapabilityInvocation] = field(default_factory=list)
    capability_results: list[CapabilityResult] = field(default_factory=list)
    final_response: str | None = None


@dataclass(frozen=True)
class FinalizationContext:
    """Trusted framework metadata supplied to task finalizers."""

    budget_exhaustion: BudgetExhaustion | None
    tool_calls_count: int
    last_workflow_step: str | None
    identity: TrustedEpisodeIdentity | None = None


@dataclass
class TaskReward:
    """Result of compute_reward(). Single source of truth for episode outcome."""

    value: float
    success: bool
    breakdown: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrustedEpisodeIdentity:
    """Framework-owned identity that agent input cannot spoof."""

    run_id: str
    generation_id: int
    episode_index: int
    episode_id: str


@dataclass(frozen=True)
class TaskCommitPreparation:
    """Opaque task intent and reward prepared for framework publication."""

    token: str
    intent_digest: str
    reward: TaskReward
