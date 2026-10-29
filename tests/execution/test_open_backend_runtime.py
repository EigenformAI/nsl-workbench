from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.execution.backend_runtime import open_backend_runtime
from src.typing.config import AppConfig


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="claude",
        code_host_cache_path=str(tmp_path / "cache"),
        container_ids=["container-a"],
        train_data_save_folder=str(tmp_path / "train-data"),
        harness={
            "name": "orchestrator_modes",
            "orchestrator_modes": {
                "orchestrator_prompt": "prompt {scratchpad_content}",
            },
        },
        observability={"enabled": True},
    )


def _make_container_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        task={"class": "tests.fixtures.stub_task.ValidStubTask"},
        model_name="vllm:Qwen/Qwen2.5-Coder-7B-Instruct",
        code_host_cache_path=str(tmp_path / "cache"),
        container_ids=[],
        train_data_save_folder=str(tmp_path / "train-data"),
        vllm=AppConfig.VllmConfig(tool_call_parser="hermes"),
        harness={
            "name": "container",
            "container": {
                "profile": "ms_agent",
                "image": "nsl/test:latest",
                "profile_config": {"model": "nsl-model"},
            },
        },
        observability={"enabled": True},
    )


def test_open_backend_runtime_propagates_run_id_and_flushes(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    backend_session = MagicMock(genner=MagicMock(), metrics_url="http://metrics")
    backend_session.smoke_test.return_value = "ok"
    wrapped_genner = MagicMock()
    collector = MagicMock()
    enter_count = 0
    exit_count = 0

    @contextmanager
    def backend_context():
        nonlocal enter_count, exit_count
        enter_count += 1
        try:
            yield backend_session
        finally:
            exit_count += 1

    with (
        patch(
            "src.execution.backend_runtime.resolve_backend_context",
            return_value=backend_context(),
        ),
        patch(
            "src.execution.backend_runtime.MetricsGenner",
            return_value=wrapped_genner,
        ),
    ):
        with open_backend_runtime(
            config,
            run_id="run-123",
            docker_client=MagicMock(),
            task=MagicMock(),
            metrics_collector=collector,
        ) as runtime:
            assert runtime.run_id == "run-123"
            assert runtime.genner is wrapped_genner
            assert runtime.metrics is collector

    assert enter_count == 1
    assert exit_count == 1
    assert collector.vllm_metrics_url == "http://metrics"
    collector.flush.assert_called_once_with()
    backend_session.smoke_test.assert_called_once_with()


def test_open_backend_runtime_rejects_unsafe_supplied_run_id(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    with (
        patch("src.execution.backend_runtime.resolve_backend_context") as resolve,
        pytest.raises(ValueError, match="safe path component"),
    ):
        with open_backend_runtime(config, run_id="../escape"):
            pass

    resolve.assert_not_called()


def test_open_backend_runtime_validates_generated_run_id(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    with (
        patch(
            "src.execution.backend_runtime.generate_readable_run_id",
            return_value="nested/run",
        ),
        patch("src.execution.backend_runtime.resolve_backend_context") as resolve,
        pytest.raises(ValueError, match="safe path component"),
    ):
        with open_backend_runtime(config):
            pass

    resolve.assert_not_called()


def test_open_backend_runtime_flushes_metrics_on_exception(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    backend_session = MagicMock(genner=MagicMock(), metrics_url=None)
    collector = MagicMock()

    @contextmanager
    def backend_context():
        yield backend_session

    with patch(
        "src.execution.backend_runtime.resolve_backend_context",
        return_value=backend_context(),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            with open_backend_runtime(
                config,
                docker_client=MagicMock(),
                task=MagicMock(),
                metrics_collector=collector,
            ):
                raise RuntimeError("boom")

    collector.flush.assert_called_once_with()


def test_open_backend_runtime_runs_static_warnings_before_backend_context(
    tmp_path: Path,
) -> None:
    config = _make_container_config(tmp_path)
    backend_session = MagicMock(genner=MagicMock(), metrics_url=None)
    order: list[str] = []

    @contextmanager
    def backend_context():
        order.append("backend_enter")
        yield backend_session

    def _static(config):
        order.append("static")
        return []

    with (
        patch(
            "src.execution.backend_runtime.emit_static_tool_contract_warnings",
            side_effect=_static,
        ),
        patch(
            "src.execution.backend_runtime.resolve_backend_context",
            return_value=backend_context(),
        ),
        patch("src.execution.backend_runtime.run_tool_call_contract_probe"),
    ):
        with open_backend_runtime(
            config,
            docker_client=MagicMock(),
            task=MagicMock(),
            metrics_collector=MagicMock(),
        ):
            pass

    assert order[:2] == ["static", "backend_enter"]


def test_open_backend_runtime_runs_dynamic_probe_after_smoke_before_yield(
    tmp_path: Path,
) -> None:
    config = _make_container_config(tmp_path)
    order: list[str] = []
    backend_session = MagicMock(genner=MagicMock(), metrics_url=None)
    backend_session.smoke_test.side_effect = lambda: order.append("smoke") or "ok"

    @contextmanager
    def backend_context():
        order.append("backend_enter")
        yield backend_session

    def _probe(**kwargs):
        order.append("probe")
        return MagicMock(status="passed")

    with (
        patch("src.execution.backend_runtime.emit_static_tool_contract_warnings"),
        patch(
            "src.execution.backend_runtime.resolve_backend_context",
            return_value=backend_context(),
        ),
        patch(
            "src.execution.backend_runtime.run_tool_call_contract_probe",
            side_effect=_probe,
        ),
    ):
        with open_backend_runtime(
            config,
            docker_client=MagicMock(),
            task=MagicMock(),
            metrics_collector=MagicMock(),
        ):
            order.append("yield")

    assert order == ["backend_enter", "smoke", "probe", "yield"]
