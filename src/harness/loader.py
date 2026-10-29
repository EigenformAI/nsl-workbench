"""Harness loader — resolves harness, TracedGenner, and EventRecorder classes.

Users can plug in custom subclasses via dotted-path config:

    [harness]
    harness_class = "my_project.harness.MyOrchestratorSubclass"
    traced_genner_class = "my_project.harness.RedactingTracedGenner"
    event_recorder_class = "my_project.harness.OtelEventRecorder"

Unset overrides use the framework defaults.
"""

from __future__ import annotations

import importlib
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from src.harness.base import HarnessSpec
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.typing.config import HarnessConfig


T = TypeVar("T")


def _lazy_container_harness() -> Type[HarnessSpec]:
    # Lazy import: ContainerHarness pulls in FastAPI / Docker / MCP which we
    # do not want on the critical path for in-process harness users.
    from src.harness.container import ContainerHarness

    return ContainerHarness


_BUILTIN_HARNESSES: dict[str, Any] = {
    "orchestrator_modes": OrchestratorModeHarness,
    "container": _lazy_container_harness,
}


def _resolve_dotted(path: str) -> Any:
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {path!r}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module {module_path!r} has no attribute {class_name!r}"
        ) from exc


def resolve_harness_class(config: HarnessConfig) -> Type[HarnessSpec]:
    """Return the harness class indicated by ``config``.

    Priority: ``config.harness_class`` dotted path overrides the registered
    builtin resolved via ``config.name``.
    """
    if config.harness_class:
        cls = _resolve_dotted(config.harness_class)
        if not (isinstance(cls, type) and issubclass(cls, HarnessSpec)):
            raise TypeError(
                f"{config.harness_class!r} is not a HarnessSpec subclass"
            )
        return cls
    builtin = _BUILTIN_HARNESSES.get(config.name)
    if builtin is None:
        raise KeyError(
            f"Unknown harness name {config.name!r}. Builtins: "
            f"{sorted(_BUILTIN_HARNESSES)}. Set harness.harness_class "
            f"to a dotted path for custom harnesses."
        )
    if callable(builtin) and not isinstance(builtin, type):
        return builtin()
    return builtin


def resolve_traced_genner_class(
    config: HarnessConfig,
) -> Type[TracedGenner]:
    if config.traced_genner_class:
        cls = _resolve_dotted(config.traced_genner_class)
        if not (isinstance(cls, type) and issubclass(cls, TracedGenner)):
            raise TypeError(
                f"{config.traced_genner_class!r} is not a TracedGenner subclass"
            )
        return cls
    return TracedGenner


def resolve_event_recorder_class(
    config: HarnessConfig,
) -> Type[EventRecorder]:
    if config.event_recorder_class:
        cls = _resolve_dotted(config.event_recorder_class)
        if not (isinstance(cls, type) and issubclass(cls, EventRecorder)):
            raise TypeError(
                f"{config.event_recorder_class!r} is not an EventRecorder subclass"
            )
        return cls
    return EventRecorder


def construct_harness(config: HarnessConfig) -> HarnessSpec:
    """Construct a fresh harness instance for a single episode."""
    cls = resolve_harness_class(config)
    section = getattr(config, config.name, None)
    if isinstance(section, BaseModel):
        settings: dict[str, Any] = section.model_dump()
    else:
        settings = dict(section or {})
    harness = cls(harness_config=settings)
    harness.validate()
    return harness
