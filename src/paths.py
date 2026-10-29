"""Shared path constants.

``PROJECT_ROOT`` is the repository root — the directory containing
``src/``, ``config/``, ``docker/``, ``tests/``, etc. Callers that need
to resolve a repo-relative path should import this rather than
recomputing ``Path(__file__).resolve().parents[N]`` per site, which
breaks silently when a file moves between nesting levels.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
