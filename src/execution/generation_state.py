from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Iterable

from src.typing.trajectory import EpisodeTrajectory


def is_bootstrap_episode(episode: EpisodeTrajectory) -> bool:
    return bool((episode.task_breakdown or {}).get("bootstrap_active"))


def is_admitted_episode(episode: EpisodeTrajectory) -> bool:
    return bool((episode.task_breakdown or {}).get("admitted"))


@dataclass(frozen=True)
class CommittedEpisodeAccounting:
    committed_episode_indices: tuple[int, ...]
    next_episode_index: int
    bootstrap_episodes_run: int
    regular_episodes_run: int
    has_admission: bool


def committed_episode_accounting(
    committed: Iterable[EpisodeTrajectory],
) -> CommittedEpisodeAccounting:
    episodes = list(committed)
    indices = tuple(sorted(episode.episode_index for episode in episodes))
    bootstrap_episodes_run = sum(is_bootstrap_episode(ep) for ep in episodes)
    return CommittedEpisodeAccounting(
        committed_episode_indices=indices,
        next_episode_index=(indices[-1] + 1 if indices else 0),
        bootstrap_episodes_run=bootstrap_episodes_run,
        regular_episodes_run=len(episodes) - bootstrap_episodes_run,
        has_admission=any(is_admitted_episode(ep) for ep in episodes),
    )


@dataclass(frozen=True)
class BudgetSnapshot:
    bootstrap_done: int
    regular_done: int
    has_admission: bool
    in_flight: int
    next_episode_index: int


class IntrinsicBudgetController:
    """Serialize reservations and intrinsic committed-episode accounting.

    Thresholds close new issuance based on committed episodes; they do not count
    existing reservations. Parallel work therefore drains to at most the active
    threshold plus the number of active reservations minus one. Episode class and
    admission are reconstructed from persisted ``bootstrap_active`` and
    ``admitted`` task-breakdown values.
    """

    def __init__(
        self,
        *,
        max_episodes: int,
        max_bootstrap_episodes: int | None,
        committed: Iterable[EpisodeTrajectory],
    ) -> None:
        episodes = list(committed)
        accounting = committed_episode_accounting(episodes)
        self.max_episodes = max_episodes
        self.max_bootstrap_episodes = max_bootstrap_episodes
        self.bootstrap_done = accounting.bootstrap_episodes_run
        self.regular_done = accounting.regular_episodes_run
        self.has_admission = accounting.has_admission
        self._next_episode_index = accounting.next_episode_index
        self._reservations: set[int] = set()
        self._stop_reason: str | None = None
        self.lock = threading.RLock()

    @property
    def in_flight(self) -> int:
        with self.lock:
            return len(self._reservations)

    @property
    def stop_reason(self) -> str | None:
        with self.lock:
            return self._stop_reason

    @property
    def exhaustion_reason(self) -> str | None:
        with self.lock:
            return self._exhaustion_reason_locked()

    def _exhaustion_reason_locked(self) -> str | None:
        if self.max_bootstrap_episodes is None:
            # Without a separate bootstrap allowance, max_episodes is the
            # total-work committed threshold even if admission never occurs.
            if self.bootstrap_done + self.regular_done >= self.max_episodes:
                return "max_episodes"
            return None
        if not self.has_admission:
            if self.bootstrap_done >= self.max_bootstrap_episodes:
                return "max_bootstrap_episodes"
            if self.regular_done >= self.max_episodes:
                return "max_episodes"
            return None
        if self.regular_done >= self.max_episodes:
            return "max_episodes"
        return None

    def reserve(self) -> int | None:
        with self.lock:
            reason = self._exhaustion_reason_locked()
            if reason is not None:
                self._stop_reason = reason
                return None
            episode_index = self._next_episode_index
            self._next_episode_index += 1
            self._reservations.add(episode_index)
            return episode_index

    def abandon(self, episode_index: int) -> None:
        with self.lock:
            if episode_index not in self._reservations:
                raise RuntimeError(f"episode index {episode_index} is not reserved")
            self._reservations.remove(episode_index)

    def commit(self, episode: EpisodeTrajectory) -> None:
        with self.lock:
            episode_index = episode.episode_index
            if episode_index not in self._reservations:
                raise RuntimeError(f"episode index {episode_index} is not reserved")
            self._reservations.remove(episode_index)
            if is_bootstrap_episode(episode):
                self.bootstrap_done += 1
            else:
                self.regular_done += 1
            self.has_admission = self.has_admission or is_admitted_episode(episode)

    def snapshot(self) -> BudgetSnapshot:
        with self.lock:
            return BudgetSnapshot(
                bootstrap_done=self.bootstrap_done,
                regular_done=self.regular_done,
                has_admission=self.has_admission,
                in_flight=len(self._reservations),
                next_episode_index=self._next_episode_index,
            )


__all__ = [
    "BudgetSnapshot",
    "CommittedEpisodeAccounting",
    "IntrinsicBudgetController",
    "committed_episode_accounting",
    "is_admitted_episode",
    "is_bootstrap_episode",
]
