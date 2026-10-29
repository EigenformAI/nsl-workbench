from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.typing.config import AppConfig


@pytest.mark.parametrize("task", [None, {}, {"config": {"option": True}}])
def test_task_selection_must_be_explicit(task: dict[str, object] | None) -> None:
    payload = _base_config()
    payload.pop("task", None)
    if task is not None:
        payload["task"] = task

    with pytest.raises(ValidationError) as error:
        AppConfig.model_validate(payload)

    expected = ("task",) if task is None else ("task", "class")
    assert any(
        item["loc"] == expected and item["type"] == "missing"
        for item in error.value.errors()
    )


def _base_config() -> dict[str, object]:
    return {
        "model_name": "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "code_host_cache_path": "/tmp/code",
        "container_ids": [],
        "train_data_save_folder": "/tmp/train-data",
        "task": {"class": "tests.fixtures.stub_task.ValidStubTask"},
    }


@pytest.mark.parametrize(
    ("section", "bad_key"),
    [
        (None, "model_nam"),
        ("task", "clas"),
        ("inference", "timeot"),
        ("vllm", "served_model_nam"),
        ("sglang", "served_model_nam"),
        ("llama", "startup_timeot"),
        ("observability", "record_inferenc"),
        ("training", "learning_rte"),
        ("orchestration", "num_generation"),
    ],
)
def test_framework_config_models_reject_unknown_keys(
    section: str | None,
    bad_key: str,
) -> None:
    payload = _base_config()
    if section is None:
        payload[bad_key] = "ignored"
    else:
        payload[section] = {bad_key: "ignored"}

    with pytest.raises(ValidationError, match=bad_key):
        AppConfig.model_validate(payload)


def test_task_config_remains_an_explicit_open_mapping() -> None:
    payload = _base_config()
    payload["task"] = {
        "class": "tests.fixtures.stub_task.ValidStubTask",
        "config": {"task_specific_option": {"nested": True}},
    }

    config = AppConfig.model_validate(payload)

    assert config.task.config["task_specific_option"] == {"nested": True}


def test_full_app_config_rejects_incomplete_harness_profile() -> None:
    payload = _base_config()
    payload["harness"] = {
        "name": "container",
        "container": {
            "profile": "ms_agent",
            "image": "nsl/ms-agent:test",
            "profile_config": {},
        },
    }

    with pytest.raises(ValidationError, match="model"):
        AppConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("orchestration", "num_generations", 0),
        ("orchestration", "training_window_size", 0),
        ("training", "lora_rank", 0),
        ("training", "lora_alpha", 0),
        ("training", "lora_dropout", -0.1),
        ("training", "lora_dropout", 1.0),
        ("training", "gpu_wait_min_free_memory_fraction", 1.1),
        ("generation", "generation_timeout_s", 0),
    ],
)
def test_unsafe_numeric_configuration_fails_validation(
    section: str,
    field: str,
    value: object,
) -> None:
    payload = _base_config()
    payload[section] = {field: value}

    with pytest.raises(ValidationError, match=field):
        AppConfig.model_validate(payload)


def test_recursive_external_vllm_lora_is_allowed_for_runtime_attestation() -> None:
    payload = _base_config()
    payload["vllm"] = {"endpoints": [{"base_url": "https://vllm.example"}]}
    payload["training"] = {"export_format": "lora"}
    payload["orchestration"] = {"num_generations": 2}

    config = AppConfig.model_validate(payload)

    assert config.orchestration is not None
    assert config.orchestration.num_generations == 2


def test_recursive_external_vllm_merged_export_is_rejected() -> None:
    payload = _base_config()
    payload["vllm"] = {"endpoints": [{"base_url": "https://vllm.example"}]}
    payload["training"] = {"export_format": "merged_16bit"}
    payload["orchestration"] = {"num_generations": 2}

    with pytest.raises(ValidationError, match="external vLLM.*LoRA"):
        AppConfig.model_validate(payload)


def test_recursive_loopback_endpoint_pool_still_rejects_merged_export() -> None:
    payload = _base_config()
    payload["vllm"] = {"endpoints": [{"base_url": "http://localhost:8000"}]}
    payload["training"] = {"export_format": "merged_16bit"}
    payload["orchestration"] = {"num_generations": 2}

    with pytest.raises(ValidationError, match="external vLLM.*LoRA"):
        AppConfig.model_validate(payload)


def test_recursive_single_external_vllm_merged_export_is_rejected() -> None:
    payload = _base_config()
    payload["vllm"] = {"endpoint": "https://vllm.example"}
    payload["training"] = {"export_format": "merged_16bit"}
    payload["orchestration"] = {"num_generations": 2}

    with pytest.raises(ValidationError, match="external vLLM.*LoRA"):
        AppConfig.model_validate(payload)


def test_recursive_wildcard_managed_vllm_allows_merged_export() -> None:
    payload = _base_config()
    payload["vllm"] = {"endpoint": "http://0.0.0.0:8000"}
    payload["training"] = {"export_format": "merged_16bit"}
    payload["orchestration"] = {"num_generations": 2}

    config = AppConfig.model_validate(payload)

    assert config.vllm is not None
    assert config.vllm.endpoint == "http://0.0.0.0:8000"


@pytest.mark.parametrize("backend", ["sglang:model", "claude:model", "openrouter:model"])
def test_recursive_training_rejects_unsupported_backends(backend: str) -> None:
    payload = deepcopy(_base_config())
    payload["model_name"] = backend
    payload["orchestration"] = {"num_generations": 2}

    with pytest.raises(ValidationError, match="recursive training"):
        AppConfig.model_validate(payload)


def test_training_auto_export_resolves_to_backend_consumable_format() -> None:
    from src.train.artifacts import resolve_training_export_format

    assert resolve_training_export_format("vllm:model", "auto") == "lora"
    assert resolve_training_export_format("llama:model", "auto") == "gguf"


def test_recursive_llama_rejects_peft_lora_export() -> None:
    payload = _base_config()
    payload["model_name"] = "llama:repo/model-GGUF"
    payload["training"] = {"export_format": "lora"}
    payload["orchestration"] = {"num_generations": 2}

    with pytest.raises(ValidationError, match="PEFT LoRA"):
        AppConfig.model_validate(payload)
