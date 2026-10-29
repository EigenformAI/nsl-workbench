"""Operators can select automatic, CPU-only, or partial GGUF offload."""
import pytest

from src.backend.llama import _build_llama_config
from src.typing.config import AppConfig


@pytest.mark.parametrize("layers", [-1, 0, 12])
def test_llama_gpu_layer_setting_reaches_server_config(layers: int) -> None:
    config = AppConfig(
        model_name="llama:models/example.gguf",
        code_host_cache_path="code/example",
        container_ids=[],
        train_data_save_folder="data/example",
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        llama={"n_gpu_layers": layers},
    )
    server = _build_llama_config(config, endpoint="http://localhost:8000")
    assert server.n_gpu_layers == layers
