"""Task-agnostic writer for the static, human-only run visualisation bundle."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DATA_PLACEHOLDER = "__VIZ_DATA__"
DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("index.html")
FORBIDDEN_DIRECTORY_NAMES = frozenset({"public-evidence"})
ALWAYS_FORBIDDEN_ROOTS = (Path("/data"),)
_SCRIPT_SAFE = str.maketrans(
    {
        "<": "\\u003c",
        ">": "\\u003e",
        "&": "\\u0026",
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
    }
)


def serialize_document(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=None, sort_keys=True, allow_nan=False)


def script_safe_json(text: str) -> str:
    """Escape serialized JSON so no string inside it can end a script element."""
    return text.translate(_SCRIPT_SAFE)


def write_viz_bundle(
    document: Mapping[str, Any],
    out_dir: str | Path,
    *,
    template_path: str | Path | None = None,
) -> Path:
    """Write viz.json and a self-contained index.html; return the page path."""
    template = Path(template_path or DEFAULT_TEMPLATE_PATH).read_text(encoding="utf-8")
    if template.count(DATA_PLACEHOLDER) != 1:
        raise ValueError(
            f"visualisation template must contain {DATA_PLACEHOLDER} exactly once"
        )
    payload = serialize_document(document)
    before, after = template.split(DATA_PLACEHOLDER)
    page = before + script_safe_json(payload) + after

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    write_text_atomic(directory / "viz.json", payload + "\n")
    index = directory / "index.html"
    write_text_atomic(index, page)
    return index


def assert_safe_output_dir(
    out_dir: str | Path, *, forbidden_roots: Iterable[str | Path]
) -> Path:
    """Refuse output locations an agent container or prompt could reach."""
    candidate = Path(out_dir).expanduser()
    spellings = {Path(os.path.abspath(candidate)), candidate.resolve()}
    for path in spellings:
        if FORBIDDEN_DIRECTORY_NAMES.intersection(path.parts):
            raise ValueError(
                f"refusing to write the visualisation inside a public-evidence "
                f"directory: {out_dir}"
            )
    for root in (*ALWAYS_FORBIDDEN_ROOTS, *forbidden_roots):
        if not str(root):
            continue
        root_path = Path(root).expanduser()
        for forbidden in {Path(os.path.abspath(root_path)), root_path.resolve()}:
            if any(path.is_relative_to(forbidden) for path in spellings):
                raise ValueError(
                    f"refusing to write the visualisation inside {forbidden}: {out_dir}"
                )
    return candidate.resolve()


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
