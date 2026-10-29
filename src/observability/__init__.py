from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.observability.collector import MetricsCollector
    from src.observability.genner_wrapper import MetricsGenner
    from src.observability.types import (
        InferenceMetric,
        InferenceResult,
        PhaseMetric,
        ResourceSnapshot,
        UsageInfo,
        UtilizationSummary,
    )

__all__ = [
    "InferenceMetric",
    "InferenceResult",
    "MetricsCollector",
    "MetricsGenner",
    "PhaseMetric",
    "ResourceSnapshot",
    "UsageInfo",
    "UtilizationSummary",
]


def __getattr__(name: str) -> Any:
    if name == "MetricsCollector":
        from src.observability.collector import MetricsCollector

        return MetricsCollector
    if name == "MetricsGenner":
        from src.observability.genner_wrapper import MetricsGenner

        return MetricsGenner
    if name in {
        "InferenceMetric",
        "InferenceResult",
        "PhaseMetric",
        "ResourceSnapshot",
        "UsageInfo",
        "UtilizationSummary",
    }:
        from src.observability import types as observability_types

        return getattr(observability_types, name)
    raise AttributeError(f"module 'src.observability' has no attribute {name!r}")
