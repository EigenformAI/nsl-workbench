"""Shell and admission tests for memory-cleanup measurement."""

import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.task.base import TaskEnvironmentError
from tasks.memory_cleanup import MemoryCleanupTask


def _container(cid: str = "target") -> MagicMock:
    container = MagicMock()
    container.id = cid
    container.name = cid
    container.attrs = {
        "Config": {"Labels": {"com.docker.compose.service": "service-a"}},
        "State": {"Running": True},
    }
    return container


def test_measurement_shell_is_scoped_and_posix_compatible(tmp_path):
    task = MemoryCleanupTask({})
    root = tmp_path / "dir with space"
    root.mkdir()
    (root / "file").write_bytes(b"x" * 8192)
    container = _container()
    captured = {}

    def fake_exec(args, **_kwargs):
        captured["script"] = args[-1]
        result = subprocess.run(
            ["sh", "-c", args[-1]], capture_output=True, timeout=5
        )
        return result.returncode, result.stdout + result.stderr

    container.exec_run.side_effect = fake_exec
    measured = task._measure_allocated_kb(container, (str(root),))

    assert measured > 0
    assert re.search(r"__MEASURE_KB__=\d+", captured["script"])
    assert "/home/alice" not in captured["script"]


def test_verify_population_requires_manifest_and_healthy_box():
    task = MemoryCleanupTask({})
    container = _container()
    variation = task.list_variations()[0]
    output = task._population_success_output(variation, "nonce", 0, 100)
    with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, output.encode()),
        ):
            outcome = task.populate([container], variation)

    with patch.object(task, "_run_probe_script", return_value=(True, (), True, ())):
        with patch.object(task, "_count_absent", return_value=0):
            with patch.object(task, "_measure_allocated_kb", return_value=100):
                assert task.verify_population(
                    [container],
                    variation,
                    outcome.episode_context,
                    private_context=outcome.private_context,
                )


def test_verify_population_failure_is_typed_infrastructure_error():
    task = MemoryCleanupTask({})
    container = _container()
    variation = task.list_variations()[0]
    output = task._population_success_output(variation, "nonce", 0, 100)
    with patch("tasks.memory_cleanup.secrets.token_hex", return_value="nonce"):
        with patch(
            "tasks.memory_cleanup._exec_run_with_timeout",
            return_value=(0, output.encode()),
        ):
            outcome = task.populate([container], variation)

    with patch.object(task, "_run_probe_script", return_value=(False, (), False, ())):
        with patch.object(task, "_count_absent", return_value=0):
            with patch.object(task, "_measure_allocated_kb", return_value=100):
                with pytest.raises(TaskEnvironmentError) as exc_info:
                    task.verify_population(
                        [container],
                        variation,
                        outcome.episode_context,
                        private_context=outcome.private_context,
                    )
    assert exc_info.value.container_ids == ["target"]
