"""Generation checkpoints must remain valid across interrupted writes."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.typing.training as training
from src.typing.training import (
    load_generation_checkpoint,
    save_generation_checkpoint,
)


def test_roundtrip_creates_nested_dirs_and_no_temp_left(tmp_path: Path) -> None:
    path = tmp_path / "generations" / "generation_2" / "checkpoint.json"
    payload = {"episode": 7, "admitted": [1, 2, 3]}

    save_generation_checkpoint(payload, path)

    assert load_generation_checkpoint(path) == payload
    # No temp siblings left behind.
    assert sorted(p.name for p in path.parent.iterdir()) == ["checkpoint.json"]


def test_write_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.json"
    save_generation_checkpoint({"episode": 41}, path)

    def boom(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("disk full mid-write")

    monkeypatch.setattr(training.json, "dump", boom)

    with pytest.raises(RuntimeError):
        save_generation_checkpoint({"episode": 42}, path)

    # The live file is untouched (atomic): a non-atomic write would have
    # truncated checkpoint.json to empty/corrupt before json.dump failed.
    assert load_generation_checkpoint(path) == {"episode": 41}
    assert sorted(p.name for p in path.parent.iterdir()) == ["checkpoint.json"]
