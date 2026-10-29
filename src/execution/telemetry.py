from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger


_FRAMEWORK_COUNTER_TELEMETRY: dict[str, str] = {"tool_calls": "tool_calls"}
FRAMEWORK_TELEMETRY_COLUMNS: tuple[str, ...] = tuple(
    _FRAMEWORK_COUNTER_TELEMETRY
)


def _coerce_counter(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def snapshot_recorder_counters(recorder: Any) -> dict[str, Any]:
    snapshot_counters = getattr(recorder, "snapshot_counters", None)
    if not callable(snapshot_counters):
        return {}
    try:
        counters = snapshot_counters()
    except Exception as exc:
        logger.debug(f"recorder counter telemetry raised: {exc}")
        return {}
    if not isinstance(counters, Mapping):
        return {}
    return dict(counters)


def snapshot_recorder_labels(recorder: Any) -> dict[str, str]:
    snapshot_labels = getattr(recorder, "snapshot_labels", None)
    if not callable(snapshot_labels):
        return {}
    try:
        labels = snapshot_labels()
    except Exception as exc:
        logger.debug(f"recorder label telemetry raised: {exc}")
        return {}
    if not isinstance(labels, Mapping):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def snapshot_recorder_telemetry_sources(
    recorder: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    return snapshot_recorder_counters(recorder), snapshot_recorder_labels(recorder)


def framework_telemetry_from_counters(
    counters: Mapping[str, Any] | None,
) -> dict[str, str]:
    counters = counters or {}
    return {
        column: str(_coerce_counter(counters.get(counter_name, 0)))
        for column, counter_name in _FRAMEWORK_COUNTER_TELEMETRY.items()
    }


def snapshot_framework_telemetry(recorder: Any) -> dict[str, str]:
    return framework_telemetry_from_counters(snapshot_recorder_counters(recorder))


def initial_framework_telemetry() -> dict[str, str]:
    return framework_telemetry_from_counters({})


def telemetry_columns_for_harness(harness: Any) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for column in FRAMEWORK_TELEMETRY_COLUMNS:
        columns.append(column)
        seen.add(column)

    telemetry_columns_fn = getattr(harness, "telemetry_columns", None)
    if not callable(telemetry_columns_fn):
        return columns
    harness_columns = telemetry_columns_fn()
    if not isinstance(harness_columns, list):
        return columns
    for column in harness_columns:
        key = str(column)
        if key in seen:
            continue
        columns.append(key)
        seen.add(key)
    return columns
