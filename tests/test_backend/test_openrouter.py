from unittest.mock import MagicMock

import src.backend.openrouter as openrouter_module
from src.backend.openrouter import setup_openrouter
from src.backend.resolver import get_backend_context_factory
from src.typing.config import AppConfig


def test_openrouter_backend_resolves_exact_name_and_model_prefix() -> None:
    assert get_backend_context_factory("openrouter") is setup_openrouter
    assert get_backend_context_factory("openrouter:provider/model") is setup_openrouter


def test_openrouter_setup_uses_server_top_p_default(monkeypatch) -> None:
    config = AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="openrouter:provider/model",
        code_host_cache_path="/tmp/code-host-cache",
        container_ids=[],
        main_container_idx=0,
        dynamic_container=False,
        docker_compose_dir="",
        train_data_save_folder="/tmp/train-data",
    )
    client = MagicMock()
    genner = MagicMock()
    openai_cls = MagicMock(return_value=client)
    get_genner = MagicMock(return_value=genner)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter_module, "OpenAI", openai_cls)
    monkeypatch.setattr(openrouter_module, "get_genner", get_genner)

    with setup_openrouter(config) as session:
        assert session.genner is genner
        assert session.config.top_p == 0.9

    get_genner.assert_called_once_with(
        "vllm",
        server_config=session.config,
        oai_client=client,
    )
