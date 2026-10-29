"""Phase-tag resolution inside ``OpenAiShim._resolve_phase``.

Hierarchy:

1. Explicit ``X-NSL-Phase`` header — used verbatim.
2. Auto-generated ``external::{profile_name}::step_{N}`` where ``profile_name``
   falls back from ``X-NSL-Profile`` header to ``"external"`` and ``N`` is
   monotonic per shim instance (lock-protected).

These are the narrow contract tests — HTTP-layer handling is covered by
``test_openai_shim.py``; this file exercises ``_resolve_phase`` directly.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

from src.harness.openai_shim import OpenAiShim
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner


def _make_shim(tmp_path: Path, episode_id: str) -> OpenAiShim:
    recorder = EventRecorder(
        episode_id=episode_id, output_path=tmp_path / f"{episode_id}.jsonl"
    )
    traced = TracedGenner(
        inner=MagicMock(),
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id=episode_id,
    )
    return OpenAiShim(traced, token="t", episode_id=episode_id, recorder=recorder)


def _req() -> MagicMock:
    r = MagicMock()
    r.stream = False
    r.model = "nsl-test"
    r.messages = []
    r.tools = None
    return r


def test_explicit_x_nsl_phase_header_wins(tmp_path: Path) -> None:
    shim = _make_shim(tmp_path, "ep-1")
    phase = shim._resolve_phase(_req(), {"x-nsl-phase": "orchestrator"})
    assert phase == "orchestrator"


def test_auto_generated_phase_uses_external_prefix(tmp_path: Path) -> None:
    shim = _make_shim(tmp_path, "ep-1")
    p1 = shim._resolve_phase(_req(), {})
    p2 = shim._resolve_phase(_req(), {})
    assert p1 == "external::external::step_1"
    assert p2 == "external::external::step_2"


def test_auto_generated_phase_uses_profile_header(tmp_path: Path) -> None:
    shim = _make_shim(tmp_path, "ep-1")
    phase = shim._resolve_phase(_req(), {"x-nsl-profile": "ms_agent"})
    assert phase == "external::ms_agent::step_1"


def test_phase_counter_is_per_instance(tmp_path: Path) -> None:
    """Two shim instances → two independent counters; no shared state."""
    shim_a = _make_shim(tmp_path, "ep-a")
    shim_b = _make_shim(tmp_path, "ep-b")
    _ = shim_a._resolve_phase(_req(), {"x-nsl-profile": "ms_agent"})
    _ = shim_a._resolve_phase(_req(), {"x-nsl-profile": "ms_agent"})
    # shim_b should start at step_1 even though shim_a is at step_2.
    p_b = shim_b._resolve_phase(_req(), {"x-nsl-profile": "ms_agent"})
    assert p_b == "external::ms_agent::step_1"


def test_phase_counter_is_thread_safe(tmp_path: Path) -> None:
    """100 concurrent calls → 100 unique ``step_N`` tags, no duplicates."""
    shim = _make_shim(tmp_path, "ep-c")
    phases: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        p = shim._resolve_phase(_req(), {"x-nsl-profile": "ms_agent"})
        with lock:
            phases.append(p)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(phases) == 100
    assert len(set(phases)) == 100  # no duplicates → lock works
