from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.observability.gpu import (
    VisibleGpuDetails,
    detect_hardware_tags,
    list_visible_gpu_indices,
    read_gpu_memory_info,
    read_gpu_utilization_pct,
)


def _mock_pynvml_memory(used_mb: float, free_mb: float, total_mb: float) -> MagicMock:
    mock_pynvml = MagicMock()
    mem_info = MagicMock()
    mem_info.used = int(used_mb * 1024 * 1024)
    mem_info.free = int(free_mb * 1024 * 1024)
    mem_info.total = int(total_mb * 1024 * 1024)
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    return mock_pynvml


_real_import_module = __import__("importlib").import_module


def _patch_pynvml(mock_pynvml):
    """Patch importlib.import_module to return mock_pynvml only for 'pynvml'."""

    def selective_import(name, *args, **kwargs):
        if name == "pynvml":
            if isinstance(mock_pynvml, type) and issubclass(mock_pynvml, Exception):
                raise mock_pynvml()
            return mock_pynvml
        return _real_import_module(name, *args, **kwargs)

    return patch("src.observability.gpu.importlib.import_module", side_effect=selective_import)


def _patch_pynvml_unavailable():
    """Patch importlib.import_module so 'pynvml' import raises ImportError."""
    return _patch_pynvml(ImportError)


class TestReadGpuMemoryInfo(unittest.TestCase):
    def test_returns_tuple_via_pynvml(self) -> None:
        mock = _mock_pynvml_memory(4096.0, 2048.0, 6144.0)
        with _patch_pynvml(mock):
            result = read_gpu_memory_info()

        self.assertIsNotNone(result)
        used, free, total = result
        self.assertAlmostEqual(used, 4096.0)
        self.assertAlmostEqual(free, 2048.0)
        self.assertAlmostEqual(total, 6144.0)

    def test_calls_nvml_shutdown_on_success(self) -> None:
        mock = _mock_pynvml_memory(1024.0, 1024.0, 2048.0)
        with _patch_pynvml(mock):
            read_gpu_memory_info()

        mock.nvmlShutdown.assert_called_once()

    def test_calls_nvml_shutdown_on_failure(self) -> None:
        mock = _mock_pynvml_memory(0, 0, 0)
        mock.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("GPU error")

        with _patch_pynvml(mock):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                read_gpu_memory_info()

        mock.nvmlShutdown.assert_called_once()

    def test_accepts_device_index(self) -> None:
        mock = _mock_pynvml_memory(1024.0, 1024.0, 2048.0)
        with _patch_pynvml(mock):
            read_gpu_memory_info(device_index=1)

        mock.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)

    def test_falls_back_to_nvidia_smi(self) -> None:
        with _patch_pynvml_unavailable():
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "2048, 4096, 6144\n"
            with patch("subprocess.run", return_value=mock_result):
                result = read_gpu_memory_info()

        self.assertIsNotNone(result)
        used, free, total = result
        self.assertAlmostEqual(used, 2048.0)
        self.assertAlmostEqual(free, 4096.0)
        self.assertAlmostEqual(total, 6144.0)

    def test_returns_none_when_both_unavailable(self) -> None:
        with _patch_pynvml_unavailable():
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = read_gpu_memory_info()

        self.assertIsNone(result)



class TestReadGpuUtilizationPct(unittest.TestCase):
    def test_returns_utilization_via_pynvml(self) -> None:
        mock_pynvml = MagicMock()
        utilization = MagicMock()
        utilization.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = utilization

        with _patch_pynvml(mock_pynvml):
            result = read_gpu_utilization_pct()

        self.assertAlmostEqual(result, 75.0)

    def test_calls_nvml_shutdown(self) -> None:
        mock_pynvml = MagicMock()
        utilization = MagicMock()
        utilization.gpu = 50
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = utilization

        with _patch_pynvml(mock_pynvml):
            read_gpu_utilization_pct()

        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_falls_back_to_nvidia_smi(self) -> None:
        with _patch_pynvml_unavailable():
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "88\n"
            with patch("subprocess.run", return_value=mock_result):
                result = read_gpu_utilization_pct()

        self.assertAlmostEqual(result, 88.0)

    def test_returns_none_when_both_unavailable(self) -> None:
        with _patch_pynvml_unavailable():
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = read_gpu_utilization_pct()

        self.assertIsNone(result)


class TestListVisibleGpuIndices(unittest.TestCase):
    def test_pynvml_returns_range_of_count(self) -> None:
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        with _patch_pynvml(mock_pynvml):
            result = list_visible_gpu_indices()

        self.assertEqual(result, [0, 1])
        mock_pynvml.nvmlShutdown.assert_called_once()

    def test_nvidia_smi_fallback_parses_indices(self) -> None:
        with _patch_pynvml_unavailable():
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "0\n1\n"
            with patch("subprocess.run", return_value=mock_result):
                result = list_visible_gpu_indices()

        self.assertEqual(result, [0, 1])

    def test_returns_empty_when_both_unavailable(self) -> None:
        with _patch_pynvml_unavailable():
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = list_visible_gpu_indices()

        self.assertEqual(result, [])


class TestDetectHardwareTags(unittest.TestCase):
    def test_handles_no_visible_gpus(self) -> None:
        with patch("src.observability.gpu.list_visible_gpu_details", return_value=[]):
            result = detect_hardware_tags()

        self.assertEqual(result, ["cpu-only"])

    def test_handles_probe_failure(self) -> None:
        with patch("src.observability.gpu.list_visible_gpu_details", return_value=None):
            result = detect_hardware_tags()

        self.assertEqual(result, ["unknown-hardware"])

    def test_handles_mixed_visible_gpus(self) -> None:
        devices = [
            VisibleGpuDetails(index=0, name="NVIDIA GeForce RTX 4090", total_memory_mb=24564),
            VisibleGpuDetails(index=1, name="NVIDIA A100-SXM4-80GB", total_memory_mb=81920),
        ]
        with patch("src.observability.gpu.list_visible_gpu_details", return_value=devices):
            result = detect_hardware_tags()

        self.assertEqual(
            result,
            [
                "2x",
                "rtx-4090",
                "24gb",
                "rtx-4090-24gb-1x",
                "a100-sxm4",
                "80gb",
                "a100-sxm4-80gb-1x",
            ],
        )


if __name__ == "__main__":
    unittest.main()
