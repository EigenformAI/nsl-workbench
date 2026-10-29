"""``ContainerHarnessConfig`` is typed + per-profile config is late-validated.

Phase 2 introduces ``HarnessConfig.container: ContainerHarnessConfig | None``
with ``extra="forbid"``. The nested ``profile_config: dict[str, Any]`` stays
generic to keep the discriminated-union footprint small, but the loader
validates it against ``profile.profile_config_class`` at ``AppConfig`` load
so typos surface at the same layer as the outer pydantic model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.typing.config import (
    AppConfig,
    ContainerHarnessConfig,
)


def _min_app_payload() -> dict:
    return {
        "model_name": "test",
        "code_host_cache_path": "./code/",
        "container_ids": ["svc-1"],
        "train_data_save_folder": "./data/",
        "task": {"class": "tests.fixtures.stub_task.ValidStubTask"},
    }


def test_container_defaults_and_context_compaction_overrides() -> None:
    cfg = ContainerHarnessConfig(
        profile="ms_agent",
        image="nsl/ms-agent:0.1.0",
    )
    assert cfg.max_wall_seconds == 600
    assert cfg.mem_limit == "2g"
    assert cfg.network_mode == "bridge"
    assert cfg.inference_transport == "tcp"
    assert cfg.env == {}
    assert cfg.profile_config == {}
    assert cfg.context_compaction_enabled is False
    assert cfg.context_compaction_trigger_tokens == 52_000
    assert cfg.context_compaction_target_tokens == 45_000
    assert cfg.context_compaction_keep_recent_tool_outputs == 3
    assert cfg.context_compaction_keep_recent_assistant_reasoning == 3
    assert cfg.context_compaction_chars_per_token == 3.0

    configured = ContainerHarnessConfig(
        profile="ms_agent",
        image="nsl/ms-agent:0.1.0",
        context_compaction_enabled=True,
        context_compaction_trigger_tokens=1_000,
        context_compaction_target_tokens=700,
        context_compaction_keep_recent_tool_outputs=2,
        context_compaction_keep_recent_assistant_reasoning=1,
        context_compaction_chars_per_token=2.5,
    )

    assert configured.context_compaction_enabled is True
    assert configured.context_compaction_trigger_tokens == 1_000
    assert configured.context_compaction_target_tokens == 700
    assert configured.context_compaction_keep_recent_tool_outputs == 2
    assert configured.context_compaction_keep_recent_assistant_reasoning == 1
    assert configured.context_compaction_chars_per_token == 2.5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"context_compaction_trigger_tokens": 0},
            "trigger_tokens must be positive",
            id="nonpositive-trigger",
        ),
        pytest.param(
            {"context_compaction_target_tokens": 0},
            "target_tokens must be positive",
            id="nonpositive-target",
        ),
        pytest.param(
            {
                "context_compaction_enabled": True,
                "context_compaction_trigger_tokens": 1_000,
                "context_compaction_target_tokens": 1_000,
            },
            "target_tokens must be below",
            id="target-not-below-trigger",
        ),
        pytest.param(
            {"context_compaction_keep_recent_tool_outputs": -1},
            "keep_recent_tool_outputs must be non-negative",
            id="negative-recent-tools",
        ),
        pytest.param(
            {"context_compaction_keep_recent_assistant_reasoning": -1},
            "keep_recent_assistant_reasoning must be non-negative",
            id="negative-recent-reasoning",
        ),
        pytest.param(
            {"context_compaction_chars_per_token": 0},
            "chars_per_token must be positive",
            id="nonpositive-chars-per-token",
        ),
    ],
)
def test_container_rejects_invalid_context_compaction_settings(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ContainerHarnessConfig(
            profile="ms_agent",
            image="nsl/ms-agent:0.1.0",
            **overrides,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"network_mode": "overlay"}, id="unknown-network-mode"),
        pytest.param({"inference_transport": "uds"}, id="non-tcp-transport"),
        pytest.param({"max_wall_secnds": 900}, id="extra-key"),
    ],
)
def test_container_rejects_invalid_field_shapes(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        ContainerHarnessConfig(
            profile="ms_agent",
            image="x",
            **overrides,
        )


def test_container_config_valid_loads_through_app_config() -> None:
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "ms_agent",
            "image": "nsl/ms-agent:0.1.0",
            "profile_config": {
                "model": "claude-sonnet-4-6",
                "max_chat_round": 60,
                "tool_call_timeout": 90,
                "transcript_tag": "episode",
            },
        },
    }
    app = AppConfig.model_validate(payload)
    assert app.harness.name == "container"
    assert app.harness.container is not None
    assert app.harness.container.profile == "ms_agent"


def test_app_config_surfaces_profile_config_typo_at_load() -> None:
    """Typo in ``profile_config`` trips the profile's model validation at
    ``AppConfig`` load — not at run_episode time."""
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "ms_agent",
            "image": "nsl/ms-agent:0.1.0",
            "profile_config": {
                "model": "claude-sonnet-4-6",
                "max_chat_roud": 60,  # typo
            },
        },
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)


def test_app_config_rejects_unknown_profile_name() -> None:
    payload = _min_app_payload()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "does_not_exist",
            "image": "x",
            "profile_config": {},
        },
    }
    with pytest.raises(ValidationError):
        AppConfig.model_validate(payload)
