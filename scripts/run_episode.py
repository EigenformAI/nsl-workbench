from __future__ import annotations

import argparse

import pydantic
import toml
from dotenv import load_dotenv

load_dotenv()

from src.execution import (  # noqa: E402
    open_backend_runtime,
    run_generation_sequential,
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


def main(
    config_file: str,
    *,
    rebuild_harness: bool = False,
) -> None:
    config = _load_config(config_file)
    ensure_configured_harness(config, rebuild=rebuild_harness)
    with open_backend_runtime(config) as rt:
        run_generation_sequential(rt, generation_id=0, target_rows=1, max_episodes=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Path to the config TOML.",
    )
    parser.add_argument(
        "--rebuild-harness",
        action="store_true",
        help="Force a no-cache harness image rebuild. Source-content drift also triggers automatic rebuilding. Without a [harness.container.build] block, the image is pulled rather than built.",
    )
    cli_args = parser.parse_args()
    main(config_file=cli_args.config, rebuild_harness=cli_args.rebuild_harness)
