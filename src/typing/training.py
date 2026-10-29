import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal, TypeAlias


TrainData: TypeAlias = list[dict[str, Any]]
TrainDataType: TypeAlias = Literal["episode_v2"]


def save_train_data(
    train_data_type: TrainDataType | str,
    train_data: TrainData,
    folder_path: str | Path,
) -> None:
    folder_path = Path(folder_path)

    for train_data_item in train_data:
        cur_datetime = datetime.datetime.now()
        formatted_datetime = cur_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        run_id = train_data_item["run_id"]
        version = train_data_item["version"]

        filename = f"{version}_{run_id}_{train_data_type}_{formatted_datetime}.json"
        file_path = folder_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(train_data_item, handle, indent=4)


def save_generation_training_data(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output_path.parent)


def append_episode_jsonl(
    episode: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output_path.parent)


def ensure_durable_file(output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return
    _fsync_directory(output_path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(output_path: Path, writer: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        _fsync_directory(output_path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_durable_json(payload: dict[str, Any], output_path: Path) -> None:
    def _write(handle: Any) -> None:
        json.dump(payload, handle, indent=2, default=str)

    _atomic_write(output_path, _write)


def save_generation_checkpoint(
    checkpoint: dict[str, Any],
    output_path: Path,
) -> None:
    _atomic_write(
        output_path,
        lambda handle: json.dump(checkpoint, handle, indent=2, default=str),
    )


def save_stop_reason_sentinel(reason: str, output_path: Path) -> None:
    _atomic_write(output_path, lambda handle: handle.write(reason))


def load_generation_checkpoint(output_path: Path) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
