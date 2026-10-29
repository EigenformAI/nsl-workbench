"""Parse-time repetition detector for degenerate LLM decoding.

Disabled by default; users opt in via the harness config by setting
``repetition_guard.min_paragraphs`` inside the active harness's sub-config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class RepetitionGuardConfig(BaseModel):
    """Parse-time detector for degenerate decoding (repeated paragraphs).

    Disabled by default; enable by setting ``min_paragraphs`` to an integer
    ≥ 2. Lives on the harness config so different harnesses can tune it
    independently.
    """

    model_config = ConfigDict(extra="forbid")

    min_paragraphs: Optional[int] = Field(default=None, ge=2)
    similarity_threshold: float = Field(default=0.95, ge=0, le=1)
    min_paragraph_chars: int = Field(default=100, gt=0)
    window_size: int = Field(default=5, gt=0)
    first_hit_action: Literal["truncate", "warn_only"] = "warn_only"
    second_hit_action: Literal["end_episode", "warn_only"] = "end_episode"


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_FENCE_PATTERN = re.compile(r"```[^\n]*\n.*?\n?```", re.DOTALL)


class RepetitionAction(str, Enum):
    TRUNCATE = "truncate"
    END_EPISODE = "end_episode"
    WARN_ONLY = "warn_only"


@dataclass
class RepetitionCheck:
    """Outcome of running the detector on a single response."""

    triggered: bool
    action: RepetitionAction = RepetitionAction.WARN_ONLY
    truncated_response: Optional[str] = None
    repeated_segment_preview: Optional[str] = None
    match_index: Optional[int] = None  # paragraph index where the repeat starts


class RepetitionCollapseError(RuntimeError):
    """Raised when the detector fires with `end_episode` action."""


def _split_paragraphs(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, text) tuples. Fenced blocks count as one unit."""

    segments: list[tuple[int, int, str]] = []
    cursor = 0
    for m in _FENCE_PATTERN.finditer(text):
        if m.start() > cursor:
            _append_paragraph_segments(text, cursor, m.start(), segments)
        segments.append((m.start(), m.end(), m.group(0).strip()))
        cursor = m.end()
    if cursor < len(text):
        _append_paragraph_segments(text, cursor, len(text), segments)
    return segments


def _append_paragraph_segments(
    text: str,
    start: int,
    end: int,
    segments: list[tuple[int, int, str]],
) -> None:
    # re.split with a capturing group keeps separators, so we can walk a
    # local cursor forward without relying on str.find (which returns the
    # first occurrence and corrupts offsets when paragraphs repeat).
    local_cursor = start
    for part in re.split(r"(\n\s*\n)", text[start:end]):
        if not part:
            continue
        if _PARAGRAPH_SPLIT.fullmatch(part):
            local_cursor += len(part)
            continue
        stripped = part.strip()
        if stripped:
            segments.append((local_cursor, local_cursor + len(part), stripped))
        local_cursor += len(part)


def _similar(a: str, b: str, threshold: float) -> bool:
    if a == b:
        return True
    # Only bother with Levenshtein-ish comparison when lengths are comparable.
    short, long = sorted((len(a), len(b)))
    if short == 0 or short / long < threshold - 0.05:
        return False
    return SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= threshold


@dataclass
class RepetitionDetector:
    """Per-episode detector. Construct once per episode; call `check()` per response."""

    config: RepetitionGuardConfig
    hits: int = field(default=0, init=False)

    @property
    def enabled(self) -> bool:
        return self.config.min_paragraphs is not None

    def reset(self) -> None:
        self.hits = 0

    def check(self, response: str) -> RepetitionCheck:
        if not self.enabled or not response:
            return RepetitionCheck(triggered=False)

        min_paras = self.config.min_paragraphs
        assert min_paras is not None and min_paras >= 2

        segments = _split_paragraphs(response)
        candidates = [
            (start, end, body)
            for start, end, body in segments
            if len(body) >= self.config.min_paragraph_chars
        ]
        if len(candidates) < min_paras:
            return RepetitionCheck(triggered=False)

        match_start = self._find_repeat(candidates, min_paras)
        if match_start is None:
            return RepetitionCheck(triggered=False)

        self.hits += 1
        _, _, first_body = candidates[match_start]
        preview = first_body[:120]

        if self.hits == 1:
            action_name = self.config.first_hit_action
        else:
            action_name = self.config.second_hit_action
        action = RepetitionAction(action_name)

        logger.warning(
            f"repetition_collapse: hit={self.hits} action={action.value} "
            f"paragraphs_repeated={min_paras} preview={preview!r}"
        )

        truncated: Optional[str] = None
        if action is RepetitionAction.TRUNCATE:
            truncate_at = candidates[match_start][1]
            truncated = response[:truncate_at].rstrip() + (
                "\n\n[truncated: repetition_collapse detected]"
            )

        return RepetitionCheck(
            triggered=True,
            action=action,
            truncated_response=truncated,
            repeated_segment_preview=preview,
            match_index=match_start,
        )

    def _find_repeat(
        self,
        candidates: list[tuple[int, int, str]],
        min_paras: int,
    ) -> Optional[int]:
        threshold = self.config.similarity_threshold
        window = max(self.config.window_size, min_paras)

        # Adjacent-run detection: look for `min_paras` consecutive near-duplicates.
        run = 1
        run_start = 0
        for i in range(1, len(candidates)):
            if _similar(candidates[i - 1][2], candidates[i][2], threshold):
                run += 1
                if run >= min_paras:
                    return run_start
            else:
                run = 1
                run_start = i

        # Windowed detection: same paragraph re-emitted `min_paras` times within window.
        for i, (_, _, body) in enumerate(candidates):
            matches = [i]
            lo = max(0, i - window)
            for j in range(lo, i):
                if _similar(candidates[j][2], body, threshold):
                    matches.append(j)
                    if len(matches) >= min_paras:
                        return min(matches)
        return None
