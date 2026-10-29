"""EventRecorder — unified, append-only episode event log.

Inference calls land here automatically via :class:`TracedGenner`. Harnesses
synchronously log every other meaningful event (actions, observations, state
mutations, decisions, warnings). Each entry flushes a JSONL line to disk
immediately so partial transcripts survive crashes.

``EventRecorder`` is the **single source of truth** for training-data
trajectory capture. The harness's returned transcript is task-observable
output for reward computation; it is NOT used to build fine-tuning rows.

Ordering contract: ``entries`` preserves the arrival order of ALL entries
(inference records interleaved with non-inference events) so downstream
consumers can reconstruct the episode timeline faithfully.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.observability.types import UsageInfo
from src.typing.message import Message


class RecorderCancelledError(RuntimeError):
    """Raised by log_* / record_inference when ``cancel_event`` is set.

    Signals the harness should unwind cooperatively; equivalent to the
    signal that ``TracedGenner`` emits when cancellation is observed
    before an inference call.
    """


@dataclass
class TrajectoryRecord:
    """One recorded inference call.

    Framework-authoritative: every LLM call routed through TracedGenner
    produces one of these. ``phase`` is the harness-supplied tag used
    downstream to filter / partition training data (orchestrator,
    react_step, reflection, etc.). The framework does not interpret it.
    """

    episode_id: str
    phase: str
    messages: list[Message]
    response: str
    usage: UsageInfo | None
    timestamp: str
    success: bool
    error_message: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    reasoning_content: str | None = None


@dataclass
class Event:
    """One non-inference event emitted by the harness.

    The harness contract: synchronously log every action, observation,
    state mutation, decision, or warning at meaningful boundaries.
    """

    kind: str  # "action" | "observation" | "state" | "decision" | "warning"
    category: str
    payload: dict[str, Any]
    timestamp: str
    episode_id: str | None = None


def _usage_to_jsonable(usage: UsageInfo | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return asdict(usage)


def _record_to_jsonable(record: TrajectoryRecord) -> dict[str, Any]:
    return {
        "type": "inference",
        "episode_id": record.episode_id,
        "phase": record.phase,
        "messages": list(record.messages),
        "response": record.response,
        "reasoning_content": record.reasoning_content,
        "usage": _usage_to_jsonable(record.usage),
        "timestamp": record.timestamp,
        "success": record.success,
        "error_message": record.error_message,
        "meta": dict(record.meta),
    }


def _event_to_jsonable(event: Event) -> dict[str, Any]:
    return {
        "type": event.kind,
        "category": event.category,
        "payload": dict(event.payload),
        "timestamp": event.timestamp,
        "episode_id": event.episode_id,
    }


class EventRecorder:
    """Unified, append-only, ordered event log for one episode.

    Subclassable — user-authored wrappers can add OpenTelemetry export,
    custom sinks, filtering, or redaction by overriding ``record_inference``
    or the ``log_*`` methods. The loader resolves the concrete class from
    ``harness.event_recorder_class`` dotted-path config.
    """

    def __init__(
        self,
        episode_id: str,
        output_path: Path,
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.episode_id = episode_id
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cancel_event = cancel_event
        # Single ordered stream — inference records and non-inference events
        # in arrival order. Partitioned views (`inference_records`, `events`)
        # are filters over this list.
        self._entries: list[TrajectoryRecord | Event] = []
        self._telemetry_lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._labels: dict[str, str] = {}

    # --- Properties ---

    @property
    def inference_records(self) -> list[TrajectoryRecord]:
        """Training-data source. Read-only snapshot."""
        with self._lock:
            return [e for e in self._entries if isinstance(e, TrajectoryRecord)]

    @property
    def events(self) -> list[Event]:
        """Non-inference events (actions, observations, state, decisions, warnings)."""
        with self._lock:
            return [e for e in self._entries if isinstance(e, Event)]

    @property
    def entries(self) -> list[TrajectoryRecord | Event]:
        """All entries — inference + non-inference — in arrival order."""
        with self._lock:
            return list(self._entries)

    @property
    def output_path(self) -> Path:
        return self._output_path

    def set_label(self, key: str, value: str) -> None:
        with self._telemetry_lock:
            self._labels[key] = value

    def bump_counter(self, key: str, by: int = 1) -> None:
        with self._telemetry_lock:
            try:
                current = int(self._counters.get(key, 0))
            except (TypeError, ValueError):
                current = 0
            self._counters[key] = current + by

    def snapshot_counters(self) -> dict[str, int]:
        with self._telemetry_lock:
            return dict(self._counters)

    def snapshot_labels(self) -> dict[str, str]:
        with self._telemetry_lock:
            return dict(self._labels)

    # --- Recording ---

    def record_inference(self, record: TrajectoryRecord) -> None:
        """Called by :class:`TracedGenner` — do not call from user code."""
        self._check_cancelled()
        with self._telemetry_lock:
            workflow_step = self._labels.get("last_workflow_step")
        if workflow_step is not None and "workflow_step" not in record.meta:
            record.meta["workflow_step"] = workflow_step
        elif "workflow_step" not in record.meta:
            record.meta["workflow_step"] = None
        with self._lock:
            self._entries.append(record)
            self._flush_line(_record_to_jsonable(record))

    def log_action(self, category: str, payload: dict[str, Any]) -> None:
        """Before executing any side-effecting op."""
        self._log("action", category, payload)

    def log_observation(self, category: str, payload: dict[str, Any]) -> None:
        """After tool output is received."""
        self._log("observation", category, payload)

    def log_state(self, category: str, payload: dict[str, Any]) -> None:
        """On any harness-internal state mutation worth tracking."""
        self._log("state", category, payload)

    def log_decision(self, category: str, payload: dict[str, Any]) -> None:
        """On orchestration branch points."""
        self._log("decision", category, payload)

    def log_warning(self, category: str, payload: dict[str, Any]) -> None:
        """On recoverable oddities (parse ambiguity etc.)."""
        self._log("warning", category, payload)

    # --- Internals ---

    def _check_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise RecorderCancelledError(
                f"EventRecorder({self.episode_id}): cancel_event set"
            )

    def _log(self, kind: str, category: str, payload: dict[str, Any]) -> None:
        self._check_cancelled()
        event = Event(
            kind=kind,
            category=category,
            payload=dict(payload),
            timestamp=datetime.now().isoformat(),
            episode_id=self.episode_id,
        )
        with self._lock:
            self._entries.append(event)
            self._flush_line(_event_to_jsonable(event))

    def _flush_line(self, record: dict[str, Any]) -> None:
        """Append one JSONL line. Caller MUST hold ``self._lock``.

        OSError is NOT swallowed — the "partial transcript survives a crash"
        guarantee depends on the flush actually reaching disk. If it fails,
        surfacing the error lets the harness record the failure via its
        normal path (termination_category="harness_error") instead of
        silently diverging in-memory state from on-disk state.
        """
        with open(self._output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    # --- Introspection helpers ---

    def phases(self) -> Iterable[str]:
        with self._lock:
            return [
                entry.phase
                for entry in self._entries
                if isinstance(entry, TrajectoryRecord)
            ]

    def capability_pairs(self):
        """Join ``mcp_capability_call`` actions with ``mcp_capability_result``
        observations by shared ``correlation_id``.

        Returns ordered ``(invocation, result)`` tuples — the reconstruction
        profiles use to rebuild ``EpisodeArtifacts`` without reaching into
        event categories. Unpaired events raise ``HarnessError``: the
        bridge's action/observation pairing invariant should never leak
        through.
        """
        from src.harness.base import HarnessError
        from src.task.types import CapabilityInvocation, CapabilityResult

        actions: dict[str, tuple[int, Event]] = {}
        observations: dict[str, Event] = {}
        with self._lock:
            order = 0
            for entry in self._entries:
                if not isinstance(entry, Event):
                    continue
                cid = entry.payload.get("correlation_id")
                if cid is None:
                    continue
                if entry.kind == "action" and entry.category == "mcp_capability_call":
                    actions[cid] = (order, entry)
                    order += 1
                elif (
                    entry.kind == "observation"
                    and entry.category == "mcp_capability_result"
                ):
                    observations[cid] = entry

        orphans = set(actions).symmetric_difference(observations)
        if orphans:
            raise HarnessError(
                "capability_pairs: unpaired MCP correlation_ids "
                f"{sorted(orphans)} — action/observation pairing invariant "
                "broken."
            )

        def _invocation(payload: dict[str, Any]) -> "CapabilityInvocation":
            raw = payload.get("invocation") or {}
            return CapabilityInvocation(
                name=raw.get("name", payload.get("capability", "")),
                input=dict(raw.get("input") or {}),
            )

        def _result(payload: dict[str, Any]) -> "CapabilityResult":
            raw = payload.get("result") or {}
            return CapabilityResult(
                name=raw.get("name", payload.get("capability", "")),
                output=dict(raw.get("output") or {}),
                success=bool(raw.get("success", payload.get("success", True))),
                error=raw.get("error"),
            )

        ordered = sorted(actions.items(), key=lambda kv: kv[1][0])
        return [
            (
                _invocation(action_event.payload),
                _result(observations[cid].payload),
            )
            for cid, (_, action_event) in ordered
        ]
