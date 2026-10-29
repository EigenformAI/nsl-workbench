"""OrchestratorModeHarness must wire ``TaskPromptSpec.environment_context``
into its outgoing system prompt. Earlier it silently ignored the field —
tasks that split episode-specific context out of ``system_instruction``
would lose it entirely.
"""

from __future__ import annotations

import pytest

from src.harness.orchestrator_modes import _compose_system_prompt
from src.task.types import Capability, TaskPromptSpec


@pytest.mark.parametrize(
    ("system", "environment", "expected"),
    [
        pytest.param(
            "You are X.",
            "Target: 0xABC\nFork: 18000000",
            "You are X.\n\nTarget: 0xABC\nFork: 18000000",
            id="system-plus-environment",
        ),
        pytest.param("only system", None, "only system", id="system-only"),
        pytest.param("", "env only", "env only", id="environment-only"),
    ],
)
def test_compose_system_prompt(
    system: str, environment: str | None, expected: str
) -> None:
    spec = TaskPromptSpec(
        system_instruction=system,
        environment_context=environment,
        capabilities=[Capability(name="analyzer", description="look")],
    )

    assert _compose_system_prompt(spec) == expected
