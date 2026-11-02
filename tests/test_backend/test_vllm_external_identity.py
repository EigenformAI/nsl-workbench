import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.backend.vllm import _build_vllm_config, setup_vllm
from src.execution.generation_persistence import (
    serving_lineage_from_config,
    validate_manifest_serving_lineage,
)
from src.typing.config import AppConfig


IDENTITY = {
    "external_request_model": "trained-alias",
    "expected_model_root": "/server/adapters/trained",
    "expected_model_parent": "base-model",
}


def card(**changes):
    return SimpleNamespace(
        **{
            "id": "trained-alias",
            "root": "/server/adapters/trained",
            "parent": "base-model",
            **changes,
        }
    )


def external_config(tmp_path, pooled, seed=0):
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="vllm:Qwen/demo",
        code_host_cache_path=str(tmp_path / "code"),
        container_ids=[],
        train_data_save_folder=str(tmp_path / "train"),
        inference={} if seed is None else {"seed": seed},
        vllm={
            "lora_adapter_path": str(adapter),
            **IDENTITY,
            **(
                {"endpoints": [{"base_url": "https://inference.example", "capacity": 2}]}
                if pooled
                else {"endpoint": "https://inference.example"}
            ),
        },
    )


def fake_endpoint(monkeypatch, cards):
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(data=cards)
    monkeypatch.setattr("src.backend.vllm.OpenAI", lambda **_: client)
    monkeypatch.setattr("src.backend.vllm.is_http_ready", lambda *_: True)
    monkeypatch.setattr("src.backend.vllm._is_http_ready_with_auth", lambda *_, **__: True)
    return client


@pytest.mark.parametrize("missing", list(IDENTITY))
def test_exact_inventory_configuration_requires_all_three_fields(missing):
    with pytest.raises(ValidationError, match="configured together"):
        AppConfig.VllmConfig(
            lora_adapter_path="/local/adapter",
            **{key: value for key, value in IDENTITY.items() if key != missing},
        )


def test_exact_inventory_configuration_requires_lora():
    with pytest.raises(ValidationError, match="requires lora_adapter_path"):
        AppConfig.VllmConfig(**IDENTITY)


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("seed", [None, 0, 123])
def test_external_requests_and_resume_prediction_share_alias_and_seed(
    tmp_path, monkeypatch, pooled, seed
):
    client = fake_endpoint(monkeypatch, [card()])
    config = external_config(tmp_path, pooled, seed)
    predicted = serving_lineage_from_config(config, 1)
    with setup_vllm(config) as session:
        session.genner.plist_completion([{"role": "user", "content": "hello", "meta": {}}])
        request = client.chat.completions.create.call_args.kwargs
        assert request["model"] == "trained-alias"
        assert predicted["request_model"] == request["model"]
        assert session.config.model == request["model"]
        if seed is None:
            assert "seed" not in request
        else:
            assert request["seed"] == seed


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "root", "parent", "empty_parent", "alias"]
)
def test_external_startup_and_recovery_reject_bad_inventory(
    tmp_path, monkeypatch, pooled, defect
):
    bad_cards = {
        "missing": [],
        "duplicate": [card(), card()],
        "root": [card(root="/server/adapters/wrong")],
        "parent": [card(parent="wrong-base")],
        "empty_parent": [card(parent="")],
        "alias": [card(id="adapter")],
    }[defect]
    client = fake_endpoint(monkeypatch, bad_cards)
    config = external_config(tmp_path, pooled)
    with pytest.raises(RuntimeError, match="LoRA attestation"):
        with setup_vllm(config):
            pytest.fail("mismatched inventory was admitted")

    client.models.list.return_value = SimpleNamespace(data=[card()])
    with setup_vllm(config) as session:
        pool = session.extras["endpoint_pool"]
        pool.mark_unhealthy("vllm-0", "outage")
        client.models.list.return_value = SimpleNamespace(data=bad_cards)
        pool._recover_unhealthy_once()
        assert not pool.is_healthy("vllm-0")
        assert pool.healthy_capacity() == 0
        client.models.list.return_value = SimpleNamespace(data=[card()])
        pool._recover_unhealthy_once()
        assert pool.is_healthy("vllm-0")


def test_resume_rejects_changed_request_alias_without_mutating_manifest(tmp_path):
    config = external_config(tmp_path, pooled=True)
    path = tmp_path / "generation_manifest.json"
    path.write_text(json.dumps(serving_lineage_from_config(config, 1)))
    original = path.read_bytes()
    validate_manifest_serving_lineage(path, serving_lineage_from_config(config, 1))
    config.vllm.external_request_model = "other-alias"
    assert (
        _build_vllm_config(config, endpoint="https://inference.example").model
        == "other-alias"
    )
    with pytest.raises(RuntimeError, match="serving lineage mismatch"):
        validate_manifest_serving_lineage(path, serving_lineage_from_config(config, 1))
    assert path.read_bytes() == original


def test_pool_quarantines_only_mismatched_endpoint_and_rechecks_each_inventory(
    tmp_path, monkeypatch
):
    first = fake_endpoint(monkeypatch, [card()])
    second = MagicMock()
    second.models.list.return_value = SimpleNamespace(data=[card(root="wrong")])
    clients = iter([first, second])
    monkeypatch.setattr("src.backend.vllm.OpenAI", lambda **_: next(clients))
    config = external_config(tmp_path, pooled=True)
    config.vllm.endpoints.append(
        AppConfig.VllmConfig.EndpointConfig(base_url="https://second.example", capacity=3)
    )
    with setup_vllm(config) as session:
        pool = session.extras["endpoint_pool"]
        assert pool.healthy_capacity() == 2
        assert pool.is_healthy("vllm-0")
        assert not pool.is_healthy("vllm-1")
        pool._recover_unhealthy_once()
        assert not pool.is_healthy("vllm-1")
        second.models.list.return_value = SimpleNamespace(data=[card()])
        pool._recover_unhealthy_once()
        assert pool.healthy_capacity() == 5

        pool.mark_unhealthy("vllm-0", "outage")
        first.models.list.return_value = SimpleNamespace(data=[card(parent="wrong")])
        pool._recover_unhealthy_once()
        assert not pool.is_healthy("vllm-0")
        assert pool.healthy_capacity() == 3
