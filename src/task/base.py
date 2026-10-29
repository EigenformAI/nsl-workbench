"""TaskSpec ABC — the contract between tasks and the framework.

Tasks subclass TaskSpec[StateT] to define environment setup, prompts,
measurement, and reward computation. The framework (via a harness) handles
the agent loop, trajectory capture, and training.

This contract is intentionally harness-neutral: tasks declare
``capabilities`` rather than ``modes`` so the same task definition works
behind orchestrator-mode, ReAct, tool-calling, or external-container
harnesses.
"""

import enum
import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from docker.models.containers import Container

from src.container import compose_services, resolve_compose_file
from src.task.types import (
    CapabilityExecutionContext,
    CapabilityInvocation,
    CapabilityResult,
    EpisodeArtifacts,
    EpisodeConstraints,
    FinalizationContext,
    PopulationOutcome,
    TaskCommitPreparation,
    TaskPromptSpec,
    TaskReward,
    TrustedEpisodeIdentity,
    Variation,
    Workflow,
)

if TYPE_CHECKING:
    from src.training_data.transforms import TrainingDataTransform
    from src.typing.trajectory import EpisodeTrajectory, GenerationData

StateT = TypeVar("StateT")

# Phase tags reserved by the framework for trajectory records produced at
# harness boundaries that are not capability-level agent actions. Tasks MUST
# NOT declare capabilities with these names.
FRAMEWORK_RESERVED_PHASES: frozenset[str] = frozenset(
    {
        "orchestrator",
        "cancelled",
        "system",
        "harness_error",
    }
)


class TaskEnvironmentError(Exception):
    """Raised by task methods when the environment is in an unrecoverable state.

    Distinct from task *reward* failure (encoded in TaskReward.success).
    The framework catches this and may trip the circuit breaker.

    When the task knows which container(s) are broken, it should pass
    ``container_ids`` so the framework can do a targeted rebuild instead
    of tearing down the entire stack.
    """

    def __init__(
        self,
        message: str,
        container_ids: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.container_ids: list[str] = container_ids or []


class SaturationDecision(enum.Enum):
    CONTINUE = "continue"
    ROLLOVER = "rollover"
    STOP = "stop"


@dataclass(frozen=True)
class SaturationOutcome:
    decision: SaturationDecision
    reason: str | None = None


@dataclass(frozen=True)
class EpisodeClassification:
    """Task-owned labels consumed by generic generation policies.

    Tasks may override :meth:`TaskSpec.classify_episode` to return tags from
    their own vocabulary. ``bootstrap_active`` and ``admitted`` are
    framework-reserved ``TaskReward.breakdown`` markers that tasks populate and
    generation accounting interprets; they are intentionally not represented here.
    """

    tags: frozenset[str] = frozenset()


class TaskSpec(ABC, Generic[StateT]):
    """Base class for all tasks.

    Subclass this to define a new task. Implement the abstract methods
    and optionally override the default methods.

    Framework guarantees (per episode):

        1. reset(containers)
        2. populate(containers, variation) -> PopulationOutcome
        3. verify_population(containers, variation, episode_context) -> bool
        4. prompt_spec(variation, episode_context) -> TaskPromptSpec
        5. measure_initial_state(containers, episode_context) -> StateT
        6. [harness-driven agent loop — calls parse_response /
           execute_capability per turn]
        7. finalize_episode(containers, initial, episode_context, artifacts)
           -> TaskReward  (default: measure_final_state + compute_reward)

    Thread-safety contract
    ----------------------
    A single TaskSpec instance is shared across parallel episode workers.
    Implementations MUST be stateless with respect to episodes — do NOT
    store episode-scoped data on ``self``.

    Construction contract
    ---------------------
    Subclasses must accept ``task_config: dict[str, Any]`` as their single
    constructor argument.
    """

    # --- Identity ---
    name: str
    description: str

    # --- Metric metadata ---
    metric_name: str
    metric_unit: str
    higher_is_better: bool

    # --- Environment ---
    docker_compose_dir: str
    agent_service_name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        required = [
            "name",
            "description",
            "metric_name",
            "metric_unit",
            "higher_is_better",
            "docker_compose_dir",
            "agent_service_name",
        ]
        missing = [attr for attr in required if not hasattr(cls, attr)]
        if missing:
            raise TypeError(f"Task {cls.__name__} must define: {', '.join(missing)}")

    # --- Variations (abstract) ---

    @abstractmethod
    def list_variations(self) -> list[Variation]:
        """Return environment variations for episode diversity. Must be non-empty."""
        ...

    # --- Environment lifecycle ---

    def reset(self, containers: list[Container]) -> None:
        """Return containers to a neutral state between episodes. Default: no-op."""
        pass

    @abstractmethod
    def populate(
        self,
        containers: list[Container],
        variation: Variation,
    ) -> PopulationOutcome:
        """Apply a variation to containers for an episode."""
        ...

    def verify_population(
        self,
        containers: list[Container],
        variation: Variation,
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> bool:
        """Verify containers are in the expected post-population state. Default: True."""
        return True

    # --- Prompts (abstract) ---

    @abstractmethod
    def prompt_spec(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> TaskPromptSpec:
        """Return all prompt content plus the capability set for this episode.

        Called once per episode. The harness translates ``capabilities`` into
        whatever vocabulary it uses (modes, tools, function specs, etc.).
        """
        ...

    def workflow(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> Workflow | None:
        """Return an optional harness-neutral workflow for this episode.

        Default ``None`` preserves the single-step path. Implementations must
        be independently computable from the method arguments and must not
        depend on side effects from ``prompt_spec()``.
        """
        return None

    def episode_constraints(
        self,
        variation: Variation,
        episode_context: dict[str, Any],
    ) -> EpisodeConstraints:
        """Return task-owned episode constraints (budget, success rules, retry policy).

        Default returns ``EpisodeConstraints()`` which preserves the existing
        behaviour: ``min_task_tool_calls_for_success=1`` (keeps the zero-tool
        success override), no hard budget caps, and no no-tool retry.

        Override to declare ``max_task_tool_calls``, ``max_llm_turns``,
        ``terminal_capability_for_success``, or per-step overrides.
        """
        return EpisodeConstraints()

    def training_data_transforms(self) -> Sequence["TrainingDataTransform"]:
        """Return the ordered deterministic SFT export transform pipeline."""
        return ()

    def training_data_observability(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Return task-owned report metrics for final exported rows."""
        del rows
        return {}

    def classify_episode(
        self,
        episode: "EpisodeTrajectory",
    ) -> EpisodeClassification:
        """Return task-owned policy tags for a persisted episode.

        Overrides must be deterministic from the serialized episode. Framework
        budgeting does not use this hook: tasks must persist the framework-reserved
        ``bootstrap_active`` and ``admitted`` markers in ``TaskReward.breakdown``
        for generation accounting to interpret directly.
        """
        del episode
        return EpisodeClassification()

    # --- Response parsing & capability execution ---

    def parse_response(
        self,
        raw_response: str,
        *,
        invoked_capability: str | None = None,
    ) -> list[CapabilityInvocation]:
        """Extract structured capability invocations from an agent response.

        Default: returns []. Override to extract task-specific payloads
        (contract source, metric readouts, addresses, etc.) from the agent's
        raw text.

        Harness neutrality — two modes
        ------------------------------
        1. **Self-identifying responses.** Tasks whose agents emit a tag that
           names the capability (e.g. ``<tool name="explorer">...``, XML
           tool-calls, OpenAI function-call JSON) can parse without any
           harness hint. Such tasks work identically under an orchestrator,
           a ReAct, or a tool-calling harness.
        2. **Orchestrator-routed responses.** When a harness pre-selects
           which capability a turn invokes (orchestrator-modes harness), it
           passes ``invoked_capability`` so the task can disambiguate a
           response that doesn't self-identify. Tasks that rely on this hint
           are orchestrator-shaped by design; pair them with an
           orchestrator-shaped harness.

        MUST NOT raise — return [] on parse failure.
        """
        return []

    def execute_capability(
        self,
        invocation: CapabilityInvocation,
        containers: list[Container],
        variation: Variation,
        ctx: CapabilityExecutionContext,
    ) -> CapabilityResult:
        """Execute task-side work for a capability invocation.

        Default: returns a pass-through CapabilityResult echoing the input —
        suitable for capabilities whose "execution" is data capture only
        (e.g. metric readouts that don't require container interaction).

        Override for capabilities with task-managed side effects
        (contract deployment, external attack lifecycle, etc.).

        Normal execution failures should be encoded in the returned
        CapabilityResult with success=False. Raise TaskEnvironmentError
        only for unrecoverable environment failures.
        """
        return CapabilityResult(
            name=invocation.name,
            output=dict(invocation.input),
            success=True,
        )

    # --- Measurement & reward ---

    @abstractmethod
    def measure_initial_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        *,
        private_context: dict[str, Any] | None = None,
    ) -> StateT:
        """Capture environment state before agent execution."""
        ...

    def measure_final_state(
        self,
        containers: list[Container],
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
    ) -> StateT:
        """Capture environment state after agent execution.

        Default: delegates to measure_initial_state (artifacts ignored).
        Override for asymmetric measurement.
        """
        return self.measure_initial_state(
            containers, episode_context, private_context=private_context
        )

    @abstractmethod
    def compute_reward(
        self,
        initial: StateT,
        final: StateT,
        artifacts: EpisodeArtifacts,
    ) -> TaskReward:
        """Pure reward computation — no containers, no I/O."""
        ...

    # --- Episode finalization ---

    def finalize_episode(
        self,
        containers: list[Container],
        initial: StateT,
        episode_context: dict[str, Any],
        artifacts: EpisodeArtifacts,
        *,
        private_context: dict[str, Any] | None = None,
        finalization_context: FinalizationContext | None = None,
    ) -> TaskReward:
        """Default: measure_final_state then compute_reward."""
        final = self.measure_final_state(
            containers, episode_context, artifacts, private_context=private_context
        )
        return self.compute_reward(initial, final, artifacts)

    # --- Framework-coordinated episode commits ---

    def episode_commit_enabled(self) -> bool:
        """Return whether this task uses the durable prepare/apply handshake."""
        return False

    def prepare_episode_commit(
        self,
        identity: TrustedEpisodeIdentity,
        reward: TaskReward,
    ) -> TaskCommitPreparation:
        """Prepare task-owned mutation after framework success constraints."""
        del identity, reward
        raise RuntimeError("task did not implement prepare_episode_commit")

    def apply_episode_commit(self, token: str, intent_digest: str) -> None:
        """Idempotently apply a task intent already in the episode journal."""
        del token, intent_digest
        raise RuntimeError("task did not implement apply_episode_commit")

    def recover_episode_commits(self, journaled: Mapping[str, str]) -> None:
        """Apply journaled intents and discard task preparations absent from it."""
        del journaled
        raise RuntimeError("task did not implement recover_episode_commits")

    def parallel_checkpoint_safe(self) -> bool:
        """Return whether task state supports quiescent parallel checkpoints."""
        return False

    # --- Run-level saturation policy ---

    def saturation_enabled(self) -> bool:
        """Return whether run-level saturation evaluation should be called."""
        return False

    def evaluate_saturation(
        self,
        generation_data: "GenerationData",
    ) -> SaturationOutcome:
        """Return task-owned run-saturation policy decision.

        Default is opt-out. Tasks that own a resettable long-lived store can
        override this and return ROLLOVER or STOP; the execution loop remains
        task-agnostic and only acts on the returned decision.
        """
        del generation_data
        return SaturationOutcome(SaturationDecision.CONTINUE)

    def on_generation_rollover(
        self,
        reason_tag: str,
        generation_data: "GenerationData",
        *,
        rolled_at: str | None = None,
    ) -> None:
        """Reset task-owned run state after the execution loop has drained."""
        del reason_tag, generation_data, rolled_at
        return None

    def generation_checkpoint_state(self) -> dict[str, Any] | None:
        """Return task-owned state to persist in generation checkpoints."""
        return None

    def load_generation_checkpoint_state(
        self,
        state: dict[str, Any] | None,
    ) -> None:
        """Restore task-owned state from a generation checkpoint."""
        del state
        return None

    # --- Validation ---

    def validate(self) -> None:
        """Run task-level validation after construction.

        Called by the loader. Validates:

        - list_variations() returns ≥1 variation.
        - prompt_spec() returns a TaskPromptSpec with at least one capability.
        - Declared capability names are unique.
        - No capability name collides with a framework-reserved phase tag.
        - ``agent_service_name`` is declared in the compose file.

        Subclasses may override but MUST call ``super().validate()``.
        """
        variations = self.list_variations()
        if not variations:
            raise ValueError(f"Task {type(self).__name__} returned zero variations")
        probe = variations[0]
        spec = self.prompt_spec(probe, {})

        if not spec.capabilities:
            raise ValueError(
                f"Task {type(self).__name__}: prompt_spec() declared zero "
                f"capabilities; declare at least one."
            )
        names = [cap.name for cap in spec.capabilities]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Task {type(self).__name__}: duplicate capability names: "
                f"{sorted(duplicates)}"
            )
        reserved = set(names) & FRAMEWORK_RESERVED_PHASES
        if reserved:
            raise ValueError(
                f"Task {type(self).__name__}: capability names collide with "
                f"framework-reserved phase tags: {sorted(reserved)}"
            )

        self._validate_workflows(variations)

        task_cls = type(self)
        compose_cache = getattr(task_cls, "_compose_services_cache", {})
        compose_file = resolve_compose_file(self.docker_compose_dir)
        cache_key = str(compose_file.resolve())
        declared = compose_cache.get(cache_key)
        if declared is None:
            declared = tuple(compose_services(compose_file))
            compose_cache = dict(compose_cache)
            compose_cache[cache_key] = declared
            setattr(task_cls, "_compose_services_cache", compose_cache)

        if self.agent_service_name not in declared:
            raise ValueError(
                f"Task {type(self).__name__}.agent_service_name="
                f"{self.agent_service_name!r} not declared in compose file. "
                f"Declared services: {sorted(declared)}"
            )

    def _validate_workflows(self, variations: list[Variation]) -> None:
        for variation in variations:
            spec = self.prompt_spec(variation, {})
            workflow = self.workflow(variation, {})
            if workflow is None:
                continue
            capability_names = {cap.name for cap in spec.capabilities}
            try:
                workflow.validate(capability_names)
            except ValueError as exc:
                raise ValueError(
                    f"Task {type(self).__name__} workflow invalid for "
                    f"variation {variation.name!r}: {exc}"
                ) from exc
