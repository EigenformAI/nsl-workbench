"""HarnessSpec ABC.

A harness drives the agent loop for one episode. The ABC is deliberately
minimal — a single ``run_episode(ctx) -> HarnessTranscript`` method. Users
extend behavior not by growing the ABC but by subclassing the framework's
TracedGenner / EventRecorder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.harness.context import HarnessContext
    from src.harness.transcript import HarnessTranscript, StepTranscript
    from src.task.types import TaskPromptSpec, Workflow, WorkflowStep


class HarnessError(Exception):
    """Raised by framework layers when harness behavior is broken.

    Distinct from normal agent failure (which is encoded in the transcript's
    termination_category). A single HarnessError occurrence does NOT trip
    the circuit breaker; N consecutive occurrences across episodes do (see
    ``HarnessConfig.consecutive_harness_error_limit``).
    """


class HarnessTelemetry(Protocol):
    """Optional harness-owned UI telemetry seam.

    Implement on concrete harnesses that want the framework to surface
    lightweight per-episode state without growing ``HarnessSpec``.
    """

    def telemetry(self) -> dict[str, str]: ...

    def telemetry_columns(self) -> list[str]: ...


class HarnessRecorderTelemetry(Protocol):
    """Optional telemetry seam for harnesses backed directly by recorder data."""

    def telemetry_from_recorder_snapshot(
        self,
        counters: dict[str, Any],
        labels: dict[str, str],
    ) -> dict[str, str]: ...


class HarnessFailureExtras(Protocol):
    """Optional harness-owned failure-path diagnostics seam."""

    def failure_extras(self) -> dict[str, Any]: ...


class HarnessSpec(ABC):
    """Drives the agent loop for one episode.

    The framework owns inference (via TracedGenner), container lifecycle,
    and trajectory persistence (via EventRecorder). The harness owns
    loop shape: what counts as a step, how tools are invoked, how the
    scratchpad (if any) works.

    Optional UI telemetry / failure-path diagnostics belong on the harness
    via ``HarnessTelemetry`` / ``HarnessRecorderTelemetry`` /
    ``HarnessFailureExtras``, not on this ABC.

    Lifetime contract
    -----------------
    **Per episode**, not shared. Unlike TaskSpec (singleton across workers),
    each harness instance is constructed fresh for one episode. Harnesses
    naturally hold mutable episode state (budget, scratchpad pointers,
    repetition counters); per-episode construction avoids reset-dance.
    """

    name: str = "harness"
    description: str = ""

    def __init__(self, harness_config: dict[str, Any]) -> None:
        self.harness_config = harness_config

    @abstractmethod
    def run_episode(self, *, ctx: "HarnessContext") -> "HarnessTranscript":
        """Run exactly one episode and return a transcript.

        The transcript carries task-observable artifacts
        (capability invocations + results + final response) plus episode
        metadata (steps taken, termination reason/category). Prompt/response
        pairs for fine-tuning are captured automatically by
        ``ctx.genner`` (TracedGenner) and live in ``ctx.recorder`` — not
        in the transcript.
        """
        ...

    def run_workflow(
        self,
        workflow: "Workflow",
        ctx: "HarnessContext",
    ) -> "HarnessTranscript":
        """Run a task-declared workflow.

        Default execution is framework-driven through ``WorkflowDriver`` and
        ``run_step``. Harnesses with native workflow support override this.
        """
        from src.harness.workflow_driver import WorkflowDriver

        return WorkflowDriver(self).run(workflow, ctx)

    def run_step(
        self,
        step: "WorkflowStep",
        prompt_spec: "TaskPromptSpec",
        ctx: "HarnessContext",
        *,
        fresh_conversation: bool,
        is_first: bool,
        is_last: bool,
    ) -> "StepTranscript":
        """Dispatch one workflow step into a live harness session."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support framework-driven "
            "workflow steps. Implement run_step or override run_workflow."
        )

    def validate(self) -> None:
        """Optional self-check invoked by the loader. Default: no-op."""
        return None
