from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from loguru import logger


CODE_BLOCK_PATTERN = re.compile(
    r"```([A-Za-z0-9_+-]*)\s*\n(.*?)\n```",
    re.DOTALL,
)


@dataclass(frozen=True)
class CodeBlock:
    """A fenced code block extracted from an LLM response."""

    lang: str
    body: str


class WhenNoMatch(Enum):
    """What to do when no fenced block matches the accepted languages."""

    NONE = "none"
    RETURN_RAW = "raw"
    PREFLIGHT = "preflight"


def extract_code_block(
    text: str,
    accepted_langs: Sequence[str] = ("python",),
    *,
    fallback: WhenNoMatch = WhenNoMatch.NONE,
) -> CodeBlock | None:
    """Extract the first fenced code block matching an accepted language."""

    normalized_langs = {lang.strip().lower() for lang in accepted_langs}
    first_match: CodeBlock | None = None
    matching_count = 0

    for match in CODE_BLOCK_PATTERN.finditer(text):
        lang = match.group(1).strip().lower()
        body = match.group(2).strip()

        if not body:
            continue

        if lang not in normalized_langs:
            logger.debug(
                "Skipping fenced code block with language '{}' (accepted: {})",
                lang,
                sorted(normalized_langs),
            )
            continue

        matching_count += 1
        if first_match is None:
            first_match = CodeBlock(lang=lang, body=body)

    if matching_count > 1:
        logger.warning(
            "Multiple fenced code blocks matched accepted languages {}; using first match",
            sorted(normalized_langs),
        )

    if first_match is not None:
        return first_match

    stripped_text = text.strip()
    if fallback is WhenNoMatch.RETURN_RAW:
        return CodeBlock(lang="", body=stripped_text)

    if fallback is WhenNoMatch.PREFLIGHT and "```" not in text:
        return CodeBlock(lang="", body=stripped_text)

    return None
