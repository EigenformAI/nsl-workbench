"""Tests for the harness loader.

Covers:
- Default harness name "orchestrator_modes" resolves to OrchestratorModeHarness.
- Class-override dotted paths resolve and are validated.
- Unknown harness names raise a clear error.

Phase 2: ``HarnessConfig`` is typed and requires a populated section
matching ``name`` — the validator catches ``name=container`` with no
``[harness.container]`` section (a common TOML typo). Loader tests that
don't care about section content pass a minimal valid
``OrchestratorModesConfig``.
"""

from __future__ import annotations

import pytest

from src.harness.base import HarnessSpec
from src.harness.loader import (
    construct_harness,
    resolve_harness_class,
)
from src.harness.orchestrator_modes import OrchestratorModeHarness
from src.typing.config import HarnessConfig, OrchestratorModesConfig


class _CustomOrchestrator(OrchestratorModeHarness):
    name = "custom"


def _default_config(**overrides) -> HarnessConfig:
    """Build a minimal valid HarnessConfig with the orchestrator_modes
    section populated. Overrides merge on top."""
    overrides.setdefault(
        "orchestrator_modes",
        OrchestratorModesConfig(orchestrator_prompt="test"),
    )
    return HarnessConfig(**overrides)


def test_dotted_path_overrides_builtin():
    config = _default_config(harness_class=f"{__name__}._CustomOrchestrator")
    cls = resolve_harness_class(config)
    assert cls is _CustomOrchestrator
    assert issubclass(cls, HarnessSpec)


def test_unknown_name_raises():
    """A ``name`` not in the builtin registry AND not overridden via
    ``harness_class`` must raise at load time. Phase 2's exactly-one-of
    validator fires first (name doesn't match a known section), so the
    failure is a ``ValidationError`` rather than ``KeyError`` — either
    flavor proves the typo is caught at load."""
    from pydantic import ValidationError

    with pytest.raises((ValidationError, KeyError)):
        HarnessConfig(name="definitely_not_a_harness")


def test_construct_harness_returns_instance():
    config = _default_config()
    harness = construct_harness(config)
    assert isinstance(harness, OrchestratorModeHarness)


def test_non_harness_class_rejected():
    config = _default_config(harness_class="builtins.dict")
    with pytest.raises(TypeError):
        resolve_harness_class(config)
