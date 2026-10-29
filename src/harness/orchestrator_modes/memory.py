"""Cross-episode scratchpad for the orchestrator-modes harness."""

import re
from datetime import datetime
from typing import Optional

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text


class CrossEpisodeScratchpad:
    def __init__(self, max_chars: int = 10000):
        self.max_chars = max_chars
        self.content: str = ""

    def append(self, text: str, episode_id: Optional[str] = None) -> None:
        """Add new content with optional episode marker.

        Reasoning-model <think>...</think> blocks are stripped before entry:
        they are internal deliberation, not observations worth persisting.
        """
        stripped = _strip_think_tags(text).strip()
        if not stripped:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        if episode_id:
            entry = f"\n[{timestamp} | Episode {episode_id}] {stripped}"
        else:
            entry = f"\n[{timestamp}] {stripped}"

        self.content += entry

        # Trim from start if over limit (FIFO)
        if len(self.content) > self.max_chars:
            # Find a good cut point (try to cut at episode boundaries)
            excess = len(self.content) - self.max_chars
            cut_point = excess

            # Look for episode boundary within reasonable range
            search_start = max(0, excess - 200)
            search_end = min(len(self.content), excess + 200)
            episode_marker = self.content.find("\n[", search_start)

            if episode_marker != -1 and episode_marker < search_end:
                cut_point = episode_marker

            self.content = self.content[cut_point:].lstrip()

    def get_content(self) -> str:
        """Get current scratchpad content."""
        return self.content
