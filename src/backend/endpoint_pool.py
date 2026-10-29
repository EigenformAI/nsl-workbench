from __future__ import annotations

import hashlib
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger
from result import Err, Result

from src.genner.Base import Genner, INFERENCE_UNAVAILABLE_PREFIX
from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message


class EndpointPoolUnavailable(RuntimeError):
    """Raised when no healthy endpoint capacity can be leased."""


@dataclass
class EndpointState:
    endpoint_id: str
    base_url: str
    models_url: str
    metrics_url: str | None
    capacity: int
    genner: Genner
    api_key: str | None = None
    healthy: bool = True
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    in_flight: int = 0
    recovery_probe: Callable[[], bool] | None = None


@dataclass
class EndpointLease:
    endpoint_id: str
    base_url: str
    genner: Genner
    metrics_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _released: bool = field(default=False, init=False, repr=False)


class EndpointAwareGenner(Genner):
    """Marks an endpoint unhealthy as soon as its genner reports an outage."""

    def __init__(
        self,
        inner: Genner,
        *,
        pool: "EndpointPool",
        endpoint_id: str,
        base_url: str,
    ) -> None:
        identifier = getattr(inner, "identifier", "vllm")
        super().__init__(identifier if isinstance(identifier, str) else str(identifier))
        self.inner = inner
        self.pool = pool
        self.endpoint_id = endpoint_id
        self.base_url = base_url

    def plist_completion(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Result[InferenceResult, str]:
        kwargs: dict[str, Any] = {"tools": tools, "tool_choice": tool_choice}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        result = self.inner.plist_completion(messages, **kwargs)
        match result:
            case Err(error_message) if str(error_message).startswith(
                INFERENCE_UNAVAILABLE_PREFIX
            ):
                self.pool.mark_unhealthy(self.endpoint_id, str(error_message))
        return result

    @staticmethod
    def get_usage_info(response: object) -> UsageInfo:
        return UsageInfo()


class EndpointPool:
    """Capacity-bounded endpoint leases with slot-home affinity.

    Home assignment is deterministic and capacity-proportional over the sum of
    endpoint capacities. Leasing prefers the slot's home endpoint, spilling only
    when that endpoint is unhealthy or currently at its configured cap.
    """

    def __init__(
        self,
        endpoints: list[EndpointState],
        *,
        min_healthy_capacity: int = 1,
    ) -> None:
        if not endpoints:
            raise ValueError("EndpointPool requires at least one endpoint")
        if min_healthy_capacity < 0:
            raise ValueError("min_healthy_capacity must be non-negative")

        self._condition = threading.Condition(threading.RLock())
        self._health_monitor_lock = threading.Lock()
        self._health_monitor_stop_event: threading.Event | None = None
        self._health_monitor_thread: threading.Thread | None = None
        self._states: dict[str, EndpointState] = {}
        self._order: list[str] = []
        self.min_healthy_capacity = min_healthy_capacity

        for state in endpoints:
            if not state.endpoint_id:
                raise ValueError("endpoint_id must be non-empty")
            if state.endpoint_id in self._states:
                raise ValueError(f"duplicate endpoint_id: {state.endpoint_id}")
            if state.capacity <= 0:
                raise ValueError(
                    f"endpoint {state.endpoint_id!r} capacity must be positive"
                )
            state.genner = EndpointAwareGenner(
                state.genner,
                pool=self,
                endpoint_id=state.endpoint_id,
                base_url=state.base_url,
            )
            state.metadata = {
                **dict(state.metadata),
                "endpoint_id": state.endpoint_id,
                "base_url": state.base_url,
            }
            self._states[state.endpoint_id] = state
            self._order.append(state.endpoint_id)

        self._home_cycle = self._build_home_cycle()

    def _build_home_cycle(self) -> list[str]:
        total = sum(self._states[endpoint_id].capacity for endpoint_id in self._order)
        current = {endpoint_id: 0 for endpoint_id in self._order}
        cycle: list[str] = []
        order_index = {endpoint_id: idx for idx, endpoint_id in enumerate(self._order)}

        for _ in range(total):
            for endpoint_id in self._order:
                current[endpoint_id] += self._states[endpoint_id].capacity
            selected = max(
                self._order,
                key=lambda endpoint_id: (
                    current[endpoint_id],
                    -order_index[endpoint_id],
                ),
            )
            current[selected] -= total
            cycle.append(selected)
        return cycle

    def home_endpoint_id(self, home_key: int | str) -> str:
        if isinstance(home_key, int):
            index = home_key
        else:
            digest = hashlib.sha256(str(home_key).encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big")
        return self._home_cycle[index % len(self._home_cycle)]

    @property
    def default_genner(self) -> Genner:
        return self._default_state().genner

    @property
    def default_metrics_url(self) -> str | None:
        return self._default_state().metrics_url

    @property
    def default_api_key(self) -> str | None:
        return self._default_state().api_key

    def _default_state(self) -> EndpointState:
        with self._condition:
            endpoint_id = next(
                (
                    candidate
                    for candidate in self._order
                    if self._states[candidate].healthy
                ),
                self._order[0],
            )
            return self._states[endpoint_id]

    def endpoint_ids(self) -> list[str]:
        with self._condition:
            return list(self._order)

    def is_healthy(self, endpoint_id: str) -> bool:
        with self._condition:
            return self._states[endpoint_id].healthy

    def in_flight(self, endpoint_id: str) -> int:
        with self._condition:
            return self._states[endpoint_id].in_flight

    def total_in_flight(self) -> int:
        with self._condition:
            return sum(state.in_flight for state in self._states.values())

    def healthy_capacity(self) -> int:
        with self._condition:
            return self._healthy_capacity_locked()

    def below_capacity_floor(self) -> bool:
        with self._condition:
            return self._healthy_capacity_locked() < self.min_healthy_capacity

    def endpoint_metadata(self, endpoint_id: str) -> dict[str, Any]:
        with self._condition:
            return dict(self._states[endpoint_id].metadata)

    def wrap_genners(self, wrapper: Callable[[Genner], Genner]) -> None:
        with self._condition:
            for state in self._states.values():
                wrapped = wrapper(state.genner)
                setattr(wrapped, "endpoint_id", state.endpoint_id)
                setattr(wrapped, "base_url", state.base_url)
                state.genner = wrapped
            self._condition.notify_all()

    def lease(
        self,
        home_key: int | str,
        *,
        timeout: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> EndpointLease:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise EndpointPoolUnavailable(
                        "stop requested before endpoint lease"
                    )
                if self._healthy_capacity_locked() < self.min_healthy_capacity:
                    raise EndpointPoolUnavailable(
                        "healthy endpoint capacity below configured floor "
                        f"({self._healthy_capacity_locked()} < {self.min_healthy_capacity})"
                    )

                state = self._select_endpoint_locked(home_key)
                if state is not None:
                    state.in_flight += 1
                    return EndpointLease(
                        endpoint_id=state.endpoint_id,
                        base_url=state.base_url,
                        genner=state.genner,
                        metrics_url=state.metrics_url,
                        metadata=dict(state.metadata),
                    )

                wait_for = 0.2
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise EndpointPoolUnavailable(
                            "timed out waiting for endpoint capacity"
                        )
                    wait_for = min(wait_for, remaining)
                self._condition.wait(wait_for)

    def release(self, lease: EndpointLease | None) -> None:
        if lease is None:
            return
        with self._condition:
            if lease._released:
                return
            lease._released = True
            state = self._states.get(lease.endpoint_id)
            if state is None:
                return
            state.in_flight = max(0, state.in_flight - 1)
            self._condition.notify_all()

    def mark_unhealthy(self, endpoint_id: str, detail: str | None = None) -> None:
        with self._condition:
            state = self._states.get(endpoint_id)
            if state is None:
                return
            state.healthy = False
            state.last_error = detail
            self._condition.notify_all()
        logger.warning(f"Endpoint {endpoint_id} quarantined: {detail or 'unhealthy'}")

    def mark_healthy(self, endpoint_id: str) -> None:
        with self._condition:
            state = self._states[endpoint_id]
            state.healthy = True
            state.last_error = None
            self._condition.notify_all()

    def probe(self, endpoint_id: str, *, timeout: float = 2.0) -> bool:
        with self._condition:
            state = self._states[endpoint_id]
            models_url = state.models_url
            api_key = state.api_key
            recovery_probe = state.recovery_probe
            requires_lora_attestation = bool(
                state.metadata.get("requires_lora_attestation")
            )

        if requires_lora_attestation:
            if recovery_probe is None:
                logger.debug(
                    f"Endpoint {endpoint_id} recovery deferred because fresh LoRA "
                    "attestation is required"
                )
                return False
            try:
                healthy = recovery_probe()
            except Exception as exc:
                logger.debug(f"Endpoint {endpoint_id} recovery probe failed: {exc}")
                healthy = False
            with self._condition:
                self._states[endpoint_id].metadata["lora_attested"] = healthy
            if healthy:
                self.mark_healthy(endpoint_id)
            return healthy

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(models_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                healthy = 200 <= response.status < 300
        except Exception as exc:
            logger.debug(f"Endpoint {endpoint_id} probe failed: {exc}")
            return False
        if healthy:
            self.mark_healthy(endpoint_id)
        return healthy

    def _recover_unhealthy_once(
        self, stop_event: threading.Event | None = None
    ) -> None:
        """Re-probe every currently-quarantined endpoint once; probe() restores
        any that answer 200. The unhealthy snapshot is taken under the lock, but
        the probes run WITHOUT it (probe does its own brief locking) so a slow
        network call never blocks leasing."""
        with self._condition:
            unhealthy = [
                endpoint_id
                for endpoint_id in self._order
                if not self._states[endpoint_id].healthy
            ]
        for endpoint_id in unhealthy:
            if stop_event is not None and stop_event.is_set():
                return
            self.probe(endpoint_id)

    def start_health_monitor(self, *, interval_s: float = 30.0) -> None:
        """Start one restartable monitor for quarantined endpoints."""
        with self._health_monitor_lock:
            if (
                self._health_monitor_thread is not None
                and self._health_monitor_thread.is_alive()
            ):
                return

            stop_event = threading.Event()

            def _loop() -> None:
                while not stop_event.wait(interval_s):
                    try:
                        self._recover_unhealthy_once(stop_event)
                    except Exception:  # pragma: no cover - monitor must survive
                        logger.debug(
                            "endpoint health monitor pass failed", exc_info=True
                        )

            thread = threading.Thread(
                target=_loop, name="endpoint-health-monitor", daemon=True
            )
            self._health_monitor_stop_event = stop_event
            self._health_monitor_thread = thread
            thread.start()
        logger.debug(f"Started endpoint health monitor for {self.endpoint_ids()}")

    def stop_health_monitor(self) -> None:
        """Signal the active monitor and wait for its current finite probe."""
        with self._health_monitor_lock:
            thread = self._health_monitor_thread
            stop_event = self._health_monitor_stop_event
            if thread is None or stop_event is None:
                return
            stop_event.set()

        thread.join()
        with self._health_monitor_lock:
            if self._health_monitor_thread is thread:
                self._health_monitor_thread = None
                self._health_monitor_stop_event = None
        logger.debug(f"Stopped endpoint health monitor for {self.endpoint_ids()}")

    def _healthy_capacity_locked(self) -> int:
        return sum(state.capacity for state in self._states.values() if state.healthy)

    def _select_endpoint_locked(self, home_key: int | str) -> EndpointState | None:
        home_id = self.home_endpoint_id(home_key)
        home = self._states[home_id]
        if home.healthy and home.in_flight < home.capacity:
            return home

        candidates = [
            state
            for state in self._states.values()
            if state.healthy and state.in_flight < state.capacity
        ]
        if not candidates:
            return None

        order_index = {endpoint_id: idx for idx, endpoint_id in enumerate(self._order)}
        return min(
            candidates,
            key=lambda state: (
                state.in_flight / state.capacity,
                order_index[state.endpoint_id],
            ),
        )


__all__ = [
    "EndpointAwareGenner",
    "EndpointLease",
    "EndpointPool",
    "EndpointPoolUnavailable",
    "EndpointState",
]
