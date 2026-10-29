"""Entry points require the operator to select a configuration explicitly."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("script", "argument"),
    [
        ("run_episode.py", "config"),
        ("generate_training_data.py", "--config"),
        ("run_train_loop.py", "--config"),
    ],
)
def test_missing_config_is_a_usage_error(script: str, argument: str) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "usage:" in result.stderr
    assert "required" in result.stderr
    assert argument in result.stderr
    assert "Traceback" not in result.stderr
