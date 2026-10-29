from __future__ import annotations

import argparse
from pathlib import Path

import pydantic
import toml
from dotenv import load_dotenv

load_dotenv()

from src.execution import (  # noqa: E402
    open_backend_runtime,
    run_generation,
    save_generation_data,
)
from src.harness.provisioning import ensure_configured_harness  # noqa: E402
from src.helper import unflatten_toml_dict  # noqa: E402
from src.typing.config import AppConfig  # noqa: E402


def _load_config(config_file: str) -> AppConfig:
    with open(config_file, "r", encoding="utf-8") as handle:
        config_dict = toml.load(handle)
    try:
        return AppConfig(**unflatten_toml_dict(config_dict))
    except pydantic.ValidationError as exc:
        raise RuntimeError(f"Config validation error: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--generation-id", type=int, default=0)
    parser.add_argument("--target-rows", type=int, default=None)
    parser.add_argument(
        "--rebuild-harness",
        action="store_true",
        help="Force a no-cache harness image rebuild. Source-content drift also triggers automatic rebuilding. Without a [harness.container.build] block, the image is pulled rather than built.",
    )
    return parser.parse_args()


def main() -> Path:
    args = _parse_args()
    config = _load_config(args.config)
    ensure_configured_harness(config, rebuild=args.rebuild_harness)
    if config.generation is None:
        config.generation = AppConfig.GenerationConfig()
    if args.target_rows is not None:
        config.generation.target_training_rows = args.target_rows

    with open_backend_runtime(config) as rt:
        generation_data = run_generation(rt, generation_id=args.generation_id)
        return save_generation_data(
            generation_data,
            Path(config.generation.generation_output_dir),
            rt.run_id,
            rt.task,
            rt.config,
            request_model=rt.request_model,
            external_lora_attested=rt.external_lora_attested,
        )


if __name__ == "__main__":
    main()
