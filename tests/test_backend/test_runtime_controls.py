from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.backend.llama import _build_llama_config
from src.backend.openrouter import setup_openrouter
from src.backend.sglang import _build_sglang_config
from src.backend.vllm import _build_vllm_config
from src.genner.OAI import OAIGenner
from src.parallel import GlobalCircuitBreaker, SlotCircuitBreaker
from src.typing.config import AppConfig


def app_config(backend: str, **kwargs) -> AppConfig:
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name=f"{backend}:Qwen/demo",
        code_host_cache_path="/tmp/code",
        container_ids=[],
        train_data_save_folder="/tmp/train",
        **kwargs,
    )


@pytest.mark.parametrize("seed", [None, 0, 123])
@pytest.mark.parametrize("backend", ["llama", "vllm", "sglang", "openrouter"])
def test_seed_reaches_requests_through_each_backend(backend, seed, monkeypatch):
    app = app_config(backend, inference={} if seed is None else {"seed": seed})
    client = MagicMock()
    if backend == "openrouter":
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr("src.backend.openrouter.OpenAI", lambda **_: client)
        with setup_openrouter(app) as session:
            genner = session.genner
    else:
        builder = {
            "llama": _build_llama_config,
            "vllm": _build_vllm_config,
            "sglang": _build_sglang_config,
        }[backend]
        server = builder(app, endpoint="http://127.0.0.1:8000")
        genner = OAIGenner(client, server)

    genner.plist_completion([{"role": "user", "content": "hello", "meta": {}}])
    request = client.chat.completions.create.call_args.kwargs
    if seed is None:
        assert "seed" not in request
    else:
        assert request["seed"] == seed


@pytest.mark.parametrize("value", [0, -0.1, float("inf"), float("-inf"), float("nan")])
def test_global_breaker_threshold_must_be_positive_and_finite(value):
    with pytest.raises(ValidationError):
        AppConfig.GenerationConfig(global_circuit_breaker_threshold=value)


@pytest.mark.parametrize(
    ("threshold", "tripped", "expected"),
    [(None, 1, False), (None, 2, True), (1.0, 3, False), (1.0, 4, True), (1.1, 4, False)],
)
def test_global_breaker_threshold_boundaries(threshold, tripped, expected):
    config = AppConfig.GenerationConfig(
        **({} if threshold is None else {"global_circuit_breaker_threshold": threshold})
    )
    breakers = [SlotCircuitBreaker(max_consecutive_failures=1) for _ in range(4)]
    for breaker in breakers[:tripped]:
        breaker.record_failure()
    circuit = GlobalCircuitBreaker(
        breakers, threshold=config.global_circuit_breaker_threshold
    )
    assert circuit.is_tripped() is expected
    assert sum(breaker.is_tripped() for breaker in breakers) == tripped
