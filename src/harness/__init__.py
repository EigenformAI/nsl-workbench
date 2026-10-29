"""Harness abstraction.

A harness drives the agent loop for one episode. The framework owns
inference, containers, and trajectory persistence; the harness owns loop
shape (orchestrator, ReAct, graph, external binary, etc.).

"""

from src.harness.base import (
    HarnessError,
    HarnessFailureExtras,
    HarnessSpec,
    HarnessTelemetry,
)
from src.harness.context import HarnessConfigView, HarnessContext
from src.harness.recorder import Event, EventRecorder, TrajectoryRecord
from src.harness.traced_genner import TracedGenner
from src.harness.transcript import HarnessTranscript, StepTranscript, TerminationCategory

__all__ = [
    "Event",
    "EventRecorder",
    "HarnessConfigView",
    "HarnessContext",
    "HarnessError",
    "HarnessFailureExtras",
    "HarnessSpec",
    "HarnessTelemetry",
    "HarnessTranscript",
    "StepTranscript",
    "TerminationCategory",
    "TracedGenner",
    "TrajectoryRecord",
]
