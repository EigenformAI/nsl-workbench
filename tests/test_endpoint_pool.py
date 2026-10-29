from __future__ import annotations

import threading
from typing import Any

import pytest
from result import Err, Ok

from src.backend.endpoint_pool import (
    EndpointPool,
    EndpointPoolUnavailable,
    EndpointState,
)
from src.genner.Base import (
    Genner,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.observability.types import InferenceResult
from src.typing.message import Message


class _FakeGenner(Genner):
    def __init__(self, name: str, result: Any | None = None) -> None:
        super().__init__("vllm")
        self.name = name
        self.result = (
            result if result is not None else Ok(InferenceResult(content=name))
        )
        self.calls: list[list[Message]] = []

    def plist_completion(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ):
        del tools, tool_choice
        self.calls.append(messages)
        return self.result

    @staticmethod
    def get_usage_info(response: object):
        return None


def _state(endpoint_id: str, capacity: int, result: Any | None = None) -> EndpointState:
    return EndpointState(
        endpoint_id=endpoint_id,
        base_url=f"http://{endpoint_id}:8000/v1",
        models_url=f"http://{endpoint_id}:8000/v1/models",
        metrics_url=f"http://{endpoint_id}:8000/metrics",
        capacity=capacity,
        genner=_FakeGenner(endpoint_id, result),
    )


def test_capacity_proportional_home_assignment_is_deterministic() -> None:
    pool = EndpointPool([_state("ep0", 8), _state("ep1", 16)])

    homes = [pool.home_endpoint_id(slot_id) for slot_id in range(24)]

    assert homes.count("ep0") == 8
    assert homes.count("ep1") == 16
    assert homes == [pool.home_endpoint_id(slot_id) for slot_id in range(24)]
    assert homes == [pool.home_endpoint_id(slot_id + 24) for slot_id in range(24)]


def test_endpoint_metadata_cannot_override_canonical_provenance() -> None:
    state = _state("ep0", 1)
    state.metadata = {
        "endpoint_id": "stale-endpoint",
        "base_url": "http://stale.invalid/v1",
        "region": "local",
    }

    pool = EndpointPool([state])

    assert pool.endpoint_metadata("ep0") == {
        "endpoint_id": "ep0",
        "base_url": "http://ep0:8000/v1",
        "region": "local",
    }


def test_capacity_cap_never_exceeds_endpoint_limit() -> None:
    pool = EndpointPool([_state("ep0", 1)])

    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        with pytest.raises(EndpointPoolUnavailable):
            pool.lease(home_key=0, timeout=0.01)
        assert pool.in_flight("ep0") == 1
    finally:
        pool.release(lease)

    assert pool.in_flight("ep0") == 0


def test_reactive_quarantine_skips_unhealthy_home_until_probe_marks_healthy() -> None:
    pool = EndpointPool([_state("ep0", 1), _state("ep1", 1)])
    home = pool.home_endpoint_id(0)
    fallback = "ep1" if home == "ep0" else "ep0"

    pool.mark_unhealthy(home, "connection refused")
    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        assert lease.endpoint_id == fallback
    finally:
        pool.release(lease)

    pool.mark_healthy(home)
    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        assert lease.endpoint_id == home
    finally:
        pool.release(lease)


def test_health_monitor_recovery_pass_reprobes_only_unhealthy_endpoints() -> None:
    # A transient APIConnectionError quarantines an endpoint. Without a re-probe
    # nothing ever restores it (probe() had zero callers), so a single blip
    # permanently loses capacity for the whole run. _recover_unhealthy_once must
    # re-probe ONLY the quarantined endpoints and restore the ones that pass.
    pool = EndpointPool([_state("ep0", 4), _state("ep1", 22)])
    pool.mark_unhealthy("ep1", "APIConnectionError: connection reset")
    assert not pool.is_healthy("ep1")

    probed: list[str] = []

    def _fake_probe(endpoint_id: str, *, timeout: float = 2.0) -> bool:
        probed.append(endpoint_id)
        pool.mark_healthy(endpoint_id)  # mirrors real probe()'s success side effect
        return True

    pool.probe = _fake_probe  # type: ignore[assignment]
    pool._recover_unhealthy_once()

    assert probed == ["ep1"]  # healthy ep0 is never re-probed
    assert pool.is_healthy("ep1")
    assert pool.healthy_capacity() == 26


def test_health_monitor_recovery_pass_leaves_endpoint_quarantined_when_probe_fails() -> (
    None
):
    pool = EndpointPool([_state("ep0", 4), _state("ep1", 22)])
    pool.mark_unhealthy("ep1", "still down")

    pool.probe = lambda endpoint_id, *, timeout=2.0: False  # type: ignore[assignment]
    pool._recover_unhealthy_once()

    assert not pool.is_healthy("ep1")
    assert pool.healthy_capacity() == 4


def test_lora_endpoint_recovery_requires_fresh_attestation() -> None:
    state = _state("ep0", 1)
    attestation_results = iter([False, True])
    state.recovery_probe = lambda: next(attestation_results)
    state.metadata = {
        "requires_lora_attestation": True,
        "lora_attested": True,
    }
    pool = EndpointPool([state])
    pool.mark_unhealthy("ep0", "connection reset")

    pool._recover_unhealthy_once()
    assert not pool.is_healthy("ep0")
    assert pool.endpoint_metadata("ep0")["lora_attested"] is False

    pool._recover_unhealthy_once()
    assert pool.is_healthy("ep0")
    assert pool.endpoint_metadata("ep0")["lora_attested"] is True


def test_health_monitor_start_stop_is_idempotent_and_restartable() -> None:
    pool = EndpointPool([_state("ep0", 1)])

    pool.stop_health_monitor()
    pool.start_health_monitor(interval_s=60.0)
    first_thread = pool._health_monitor_thread
    assert first_thread is not None
    assert first_thread.is_alive()

    pool.start_health_monitor(interval_s=60.0)
    assert pool._health_monitor_thread is first_thread

    pool.stop_health_monitor()
    pool.stop_health_monitor()
    assert not first_thread.is_alive()
    assert pool._health_monitor_thread is None

    pool.start_health_monitor(interval_s=60.0)
    second_thread = pool._health_monitor_thread
    assert second_thread is not None
    assert second_thread is not first_thread
    pool.stop_health_monitor()
    assert not second_thread.is_alive()


def test_health_monitor_stop_waits_for_active_probe_and_skips_remaining() -> None:
    pool = EndpointPool([_state("ep0", 1), _state("ep1", 1)])
    pool.mark_unhealthy("ep0", "down")
    pool.mark_unhealthy("ep1", "down")
    probe_started = threading.Event()
    finish_probe = threading.Event()
    probed: list[str] = []

    def _blocking_probe(endpoint_id: str, *, timeout: float = 2.0) -> bool:
        del timeout
        probed.append(endpoint_id)
        probe_started.set()
        assert finish_probe.wait(2.0)
        return False

    pool.probe = _blocking_probe  # type: ignore[assignment]
    pool.start_health_monitor(interval_s=0)
    assert probe_started.wait(2.0)

    stopped = threading.Event()

    def _stop() -> None:
        pool.stop_health_monitor()
        stopped.set()

    stopper = threading.Thread(target=_stop)
    stopper.start()
    assert not stopped.wait(0.05)
    finish_probe.set()
    stopper.join(timeout=2.0)

    assert not stopper.is_alive()
    assert stopped.is_set()
    assert probed == ["ep0"]


def test_endpoint_unavailable_result_marks_endpoint_unhealthy_immediately() -> None:
    # A genuine endpoint outage (connection refused / unreachable) carries the
    # inference_unavailable prefix and DOES quarantine the endpoint, so the
    # pool can fail over to healthy endpoints.
    pool = EndpointPool(
        [
            _state(
                "ep0",
                1,
                Err(
                    f"{INFERENCE_UNAVAILABLE_PREFIX} APIConnectionError: connection refused"
                ),
            )
        ]
    )

    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        result = lease.genner.plist_completion([{"role": "user", "content": "hi"}])
    finally:
        pool.release(lease)

    assert isinstance(result, Err)
    assert not pool.is_healthy("ep0")


def test_inference_timeout_result_does_not_quarantine_endpoint() -> None:
    # A request timeout is a retryable episode failure, not an endpoint outage.
    # With a single endpoint, quarantining on a timeout would drop healthy
    # capacity below the floor and abort the whole run. The endpoint must stay
    # healthy: quarantine is keyed on the inference_unavailable prefix, which
    # the timeout prefix deliberately does not match.
    pool = EndpointPool(
        [
            _state(
                "ep0",
                1,
                Err(f"{INFERENCE_TIMEOUT_PREFIX} APITimeoutError: Request timed out."),
            )
        ]
    )

    lease = pool.lease(home_key=0, timeout=0.01)
    try:
        result = lease.genner.plist_completion([{"role": "user", "content": "hi"}])
    finally:
        pool.release(lease)

    assert isinstance(result, Err)
    assert pool.is_healthy("ep0")
    assert not pool.below_capacity_floor()
