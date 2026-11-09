"""Export a static, human-only visualisation page for one materials adsorption run.

Reads the archive read-only, plus blobs, generation journals and the dataset
registry, then writes viz.json and a self-contained index.html. By default the
page lands in <generation_output_dir>/<run_id>/viz/. Never point --out at a
location an agent container or prompt can reach.
"""

from __future__ import annotations

import argparse
import sqlite3
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.viz import assert_safe_output_dir, write_viz_bundle
from tasks.materials_adsorption.viz import (
    INPUT_GROUPINGS,
    extract_run_viz,
    latest_run_id,
    run_directory,
)


_TASK_DEFAULTS = {
    "archive_path": "data/materials-adsorption/archive.sqlite",
    "blob_dir": "data/materials-adsorption/blobs",
    "dataset_path": "data/materials-adsorption/dataset.json",
    "docker_compose_dir": "docker/materials-adsorption-compose",
}
_GENERATION_OUTPUT_DEFAULT = "./data/generations"


@dataclass(frozen=True)
class ExportPaths:
    archive_path: Path
    blob_dir: Path
    dataset_path: Path
    generation_root: Path
    forbidden_roots: tuple[Path, ...]


def _section(config: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = config
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _configured(section: Mapping[str, Any], key: str) -> Path | None:
    value = section.get(key)
    return Path(value) if isinstance(value, str) and value else None


def resolve_export_paths(args: argparse.Namespace) -> ExportPaths:
    config: Mapping[str, Any] = {}
    if args.config is not None:
        with Path(args.config).open("rb") as stream:
            config = tomllib.load(stream)
    task_config = _section(config, "task", "config")
    generation = _section(config, "generation")

    def pick(explicit: str | None, key: str) -> Path:
        if explicit is not None:
            return Path(explicit)
        return _configured(task_config, key) or Path(_TASK_DEFAULTS[key])

    forbidden = [
        _configured(task_config, "docker_compose_dir")
        or Path(_TASK_DEFAULTS["docker_compose_dir"]),
        _configured(config, "docker_compose_dir"),
        _configured(config, "code_host_cache_path"),
    ]
    return ExportPaths(
        archive_path=pick(args.archive, "archive_path"),
        blob_dir=pick(args.blobs, "blob_dir"),
        dataset_path=pick(args.dataset, "dataset_path"),
        generation_root=(
            Path(args.generations)
            if args.generations is not None
            else _configured(generation, "generation_output_dir")
            or Path(_GENERATION_OUTPUT_DEFAULT)
        ),
        forbidden_roots=tuple(path for path in forbidden if path is not None),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="run config TOML to read paths from")
    parser.add_argument("--archive", help="archive SQLite path (overrides --config)")
    parser.add_argument("--blobs", help="blob directory (overrides --config)")
    parser.add_argument(
        "--generations",
        help="generation_output_dir holding run directories (overrides --config)",
    )
    parser.add_argument("--dataset", help="dataset.json path (overrides --config)")
    parser.add_argument("--run-id", help="run to export; default is the latest run")
    parser.add_argument("--out", help="output directory for viz.json and index.html")
    parser.add_argument(
        "--input-grouping",
        choices=INPUT_GROUPINGS,
        default="auto",
        help="group inputs by registry family or by data correlation",
    )
    parser.add_argument(
        "--frozen-groups",
        help="frozen input-group file; default <run dir>/viz-input-groups.json",
    )
    return parser


def _missing_archive(path: Path) -> str:
    return (
        f"export_viz: no archive at {path.absolute()}. Relative config paths "
        f"resolve against the current directory ({Path.cwd()}). Run this from the "
        "directory training ran in, pass --archive, or run training first."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        paths = resolve_export_paths(args)
        if not paths.archive_path.is_file():
            raise SystemExit(_missing_archive(paths.archive_path))
        run_id = (
            latest_run_id(paths.archive_path) if args.run_id is None else args.run_id
        )
        run_dir = run_directory(paths.generation_root, run_id)
        out_dir = run_dir / "viz" if args.out is None else Path(args.out).expanduser()
        frozen_groups = (
            None
            if args.frozen_groups is None
            else Path(args.frozen_groups).expanduser()
        )
        for directory in (
            out_dir,
            run_dir if frozen_groups is None else frozen_groups.parent,
        ):
            assert_safe_output_dir(directory, forbidden_roots=paths.forbidden_roots)
        document = extract_run_viz(
            archive_path=paths.archive_path,
            blob_dir=paths.blob_dir,
            generation_root=paths.generation_root,
            dataset_path=paths.dataset_path,
            run_id=run_id,
            input_grouping=args.input_grouping,
            frozen_groups_path=frozen_groups,
        )
        index = write_viz_bundle(document, out_dir)
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise SystemExit(f"export_viz: {exc}") from exc
    print(index)


if __name__ == "__main__":
    main()
