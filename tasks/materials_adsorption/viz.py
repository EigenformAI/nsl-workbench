"""Read-only extraction of the human-only materials run visualisation document.

Verifier numbers end up in this document, so it must never reach an agent. The
archive is opened read-only, blobs are read and verified directly from disk, and
journal records contribute only allowlisted fields.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.viz import write_text_atomic
from tasks.materials_adsorption.compiler import RuleCompiler
from tasks.materials_adsorption.dataset import LoadedDataset, load_dataset
from tasks.materials_adsorption.descriptors import DescriptorRegistry


SCHEMA_VERSION = 1
TASK_NAME = "materials_adsorption"
FROZEN_GROUPS_FILENAME = "viz-input-groups.json"
INPUT_GROUPINGS = ("auto", "family", "data")
GROUP_CUT_DISTANCE = 0.5

_SIEVE_STAGES = (
    ("explore", "Explore"),
    ("hypothesise", "Hypothesise"),
    ("compile", "Compile"),
    ("submit", "Submit"),
    ("guard", "Guard"),
    ("signal", "Signal"),
    ("marginal", "Marginal"),
    ("rewrite", "Rewrite"),
    ("archive_gate", "Unique"),
    ("admitted", "Admitted"),
)
_OFF_SIEVE_STAGES = (
    ("incomplete", "Pre-submit (old archive)"),
    ("error", "Finalize error"),
    ("crash", "No outcome row"),
)
_OBSERVED_STAGE_LABELS = {"pending": "Commit pending"}
_RECORD_FIELDS = (
    "hypothesis",
    "rationale",
    "applicability",
    "expression",
    "uncertainty",
    "discriminating_test",
)
_PROCESS_COUNTS = ("compile_attempts", "compile_successes", "probes")
_GAP_FIELDS = ("parents", "inputs", "process", "rewrite")
_GENERATION_DIRECTORY = re.compile(r"generation_(\d+)")
_UNMATERIALIZED_PREFIX = "unmaterialized-"
_HEX_DIGITS = frozenset("0123456789abcdef")


def extract_run_viz(
    *,
    archive_path: str | Path,
    blob_dir: str | Path,
    generation_root: str | Path,
    dataset_path: str | Path,
    run_id: str | None = None,
    input_grouping: str = "auto",
    frozen_groups_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the viz.json document for one run without writing the archive."""
    if input_grouping not in INPUT_GROUPINGS:
        raise ValueError(f"input_grouping must be one of {', '.join(INPUT_GROUPINGS)}")
    loaded = load_dataset(dataset_path)
    archive = Path(archive_path)
    if run_id is None:
        run_id = latest_run_id(archive)
    run_dir = run_directory(generation_root, run_id)
    journal_generations, journals = _read_journals(run_dir)
    snapshot = _read_archive(archive, run_id)
    if snapshot.run_id not in snapshot.runs and not run_dir.is_dir():
        raise ValueError(
            f"run {snapshot.run_id!r} has no outcomes and no run directory"
        )

    blobs = _BlobReader(Path(blob_dir))
    fallback = _InputFallback(loaded.registry)
    gaps = dict.fromkeys(_GAP_FIELDS, 0)
    recorded = [
        _outcome_episode(row, blobs, fallback, gaps, journals)
        for row in snapshot.outcomes
    ]
    seen = {(episode["generation"], episode["episode_index"]) for episode in recorded}
    journal_only = [
        _journal_only_episode(generation, index, entries[-1])
        for (generation, index), entries in sorted(journals.items())
        if (generation, index) not in seen
    ]
    episodes = sorted(
        recorded + journal_only, key=lambda episode: episode["generation"]
    )

    by_token = {
        row["token"]: episode for row, episode in zip(snapshot.outcomes, recorded)
    }
    rules = [
        _rule(revision, rule_hash, by_token[token])
        for revision, token, rule_hash in snapshot.records
        if token in by_token
    ]
    archived_rules = {rule_hash for _revision, _token, rule_hash in snapshot.records}

    frozen_path = (
        Path(frozen_groups_path)
        if frozen_groups_path is not None
        else run_dir / FROZEN_GROUPS_FILENAME
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_NAME,
        "run_id": snapshot.run_id,
        "runs_available": list(snapshot.runs),
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "archive_revision": snapshot.archive_revision,
        "generations": sorted(
            {episode["generation"] for episode in episodes} | journal_generations
        ),
        "stages": _stages(episode["stage"] for episode in episodes),
        "episodes": episodes,
        "rules": rules,
        "edges": _edges(episodes, archived_rules),
        "inputs": _inputs(loaded, input_grouping, frozen_path),
        "sources": {
            "outcomes": len(recorded),
            "journal_only": len(journal_only),
            "missing_blobs": blobs.missing,
            "fields_not_recorded": gaps,
        },
    }


@dataclass(frozen=True)
class _ArchiveSnapshot:
    archive_revision: int
    runs: tuple[str, ...]
    run_id: str
    outcomes: tuple[sqlite3.Row, ...]
    records: tuple[tuple[int, str, str], ...]


def _connect_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"archive database not found: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30.0)
    connection.execute("PRAGMA query_only = ON")
    return connection


def latest_run_id(archive_path: str | Path) -> str:
    """Return the run with the most recent outcomes row."""
    with closing(_connect_read_only(Path(archive_path))) as connection:
        latest = connection.execute(
            "SELECT run_id FROM outcomes ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    if latest is None:
        raise ValueError("the archive has no outcomes yet; pass a run id")
    return str(latest[0])


def _read_archive(path: Path, run_id: str) -> _ArchiveSnapshot:
    with closing(_connect_read_only(path)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        try:
            revision = connection.execute(
                "SELECT value FROM metadata WHERE key = 'archive_revision'"
            ).fetchone()
            runs = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT run_id FROM outcomes GROUP BY run_id ORDER BY MIN(rowid)"
                )
            )
            outcomes = tuple(
                connection.execute(
                    """
                    SELECT outcomes.generation_id, outcomes.episode_index,
                           outcomes.episode_id, outcomes.token,
                           outcomes.training_success, outcomes.archive_admission,
                           outcomes.rejection_stage, outcomes.canonical_rule_hash,
                           outcomes.receipt_hash, outcomes.outcome_json,
                           outcomes.committed_at, commit_intents.payload_json
                    FROM outcomes
                    LEFT JOIN commit_intents ON commit_intents.token = outcomes.token
                    WHERE outcomes.run_id = ?
                    ORDER BY outcomes.generation_id, outcomes.rowid
                    """,
                    (run_id,),
                )
            )
            records = tuple(
                (int(row[0]), str(row[1]), str(row[2]))
                for row in connection.execute(
                    """
                    SELECT revision, token, canonical_rule_hash
                    FROM archive_records ORDER BY revision
                    """
                )
            )
        finally:
            connection.rollback()
    return _ArchiveSnapshot(
        archive_revision=0 if revision is None else int(revision[0]),
        runs=runs,
        run_id=run_id,
        outcomes=outcomes,
        records=records,
    )


def run_directory(generation_root: str | Path, run_id: str) -> Path:
    """Return <generation_root>/<run_id>, refusing ids that are not plain names."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError(f"run id is not a plain directory name: {run_id!r}")
    return Path(generation_root) / run_id


class _BlobReader:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[str, dict[str, Any] | None] = {}

    @property
    def missing(self) -> int:
        return sum(value is None for value in self._cache.values())

    def read_object(self, digest: str) -> dict[str, Any] | None:
        if digest not in self._cache:
            self._cache[digest] = self._load(digest)
        return self._cache[digest]

    def _load(self, digest: str) -> dict[str, Any] | None:
        if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
            return None
        try:
            data = (self._root / digest[:2] / digest).read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != digest:
            return None
        try:
            value = json.loads(data)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None


class _InputFallback:
    """Recover rule inputs for archives written before they were persisted."""

    def __init__(self, registry: DescriptorRegistry) -> None:
        self._compiler = RuleCompiler(registry)
        self._cache: dict[tuple[str, str], list[str] | None] = {}

    def inputs(self, record: Mapping[str, Any] | None) -> list[str] | None:
        if record is None or record["expression"] is None:
            return None
        key = (record["expression"], record["uncertainty"] or "0.0")
        if key not in self._cache:
            try:
                compiled = self._compiler.compile(key[0], uncertainty=key[1])
            except (ValueError, TypeError, SyntaxError, RecursionError):
                self._cache[key] = None
            else:
                self._cache[key] = sorted(compiled.referenced_descriptors)
        cached = self._cache[key]
        return None if cached is None else list(cached)


@dataclass(frozen=True)
class _JournalEntry:
    fields: Mapping[str, Any]
    has_commit_token: bool


def _read_journals(
    run_dir: Path,
) -> tuple[set[int], dict[tuple[int, int], list[_JournalEntry]]]:
    generations: set[int] = set()
    journals: dict[tuple[int, int], list[_JournalEntry]] = {}
    try:
        directories = sorted(run_dir.iterdir())
    except OSError:
        return generations, journals
    for directory in directories:
        match = _GENERATION_DIRECTORY.fullmatch(directory.name)
        if match is None or not directory.is_dir():
            continue
        generations.add(int(match.group(1)))
        try:
            stream = (directory / "all_episodes.jsonl").open("rb")
        except OSError:
            continue
        with stream:
            for line in stream:
                try:
                    raw = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(raw, dict):
                    continue
                generation = _integer(raw.get("generation_id"))
                index = _integer(raw.get("episode_index"))
                if generation is not None and index is not None:
                    token = raw.get("task_commit_token")
                    journals.setdefault((generation, index), []).append(
                        _JournalEntry(
                            fields=_journal_fields(raw),
                            has_commit_token=isinstance(token, str) and bool(token),
                        )
                    )
    return generations, journals


def _journal_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = raw.get("trajectory")
    trajectory = trajectory if isinstance(trajectory, Mapping) else {}
    extra = trajectory.get("extra")
    extra = extra if isinstance(extra, Mapping) else {}
    return {
        "episode_id": _text(raw.get("episode_id")),
        "started_at": _text(raw.get("started_at")),
        "completed_at": _text(raw.get("completed_at")),
        "duration_seconds": _number(raw.get("duration_seconds")),
        "error_category": _text(raw.get("error_category")),
        "termination_category": _text(trajectory.get("termination_category")),
        "last_workflow_step": _text(extra.get("last_workflow_step")),
        "budget_exhausted_kind": _text(extra.get("budget_exhausted_kind")),
    }


def _outcome_episode(
    row: sqlite3.Row,
    blobs: _BlobReader,
    fallback: _InputFallback,
    gaps: dict[str, int],
    journals: Mapping[tuple[int, int], Sequence[_JournalEntry]],
) -> dict[str, Any]:
    generation = int(row["generation_id"])
    index = int(row["episode_index"])
    episode_id = str(row["episode_id"])
    outcome = _json_object(row["outcome_json"])
    payload = _json_object(row["payload_json"])
    archive_record = payload.get("archive_record")
    archive_record = archive_record if isinstance(archive_record, Mapping) else {}
    admitted = bool(row["archive_admission"])

    rule_hash = row["canonical_rule_hash"]
    materialized = outcome.get("materialized")
    if not isinstance(materialized, bool):
        materialized = isinstance(rule_hash, str) and not rule_hash.startswith(
            _UNMATERIALIZED_PREFIX
        )
    receipt_hash = row["receipt_hash"]
    receipt = (
        blobs.read_object(receipt_hash) if isinstance(receipt_hash, str) else None
    )
    score = None if receipt is None else _score(receipt)
    record = _public_record(archive_record)

    parents = None
    if "parent_rule_hashes" in outcome:
        parents = _string_list(outcome["parent_rule_hashes"])
    if parents is None:
        gaps["parents"] += 1

    if "referenced_descriptors" in outcome:
        raw_inputs = outcome["referenced_descriptors"]
        inputs = _string_list(raw_inputs)
        if inputs is None and raw_inputs is not None:
            gaps["inputs"] += 1
    else:
        inputs = fallback.inputs(record)
        gaps["inputs"] += 1

    process = _process(outcome.get("process"))
    if process is None:
        gaps["process"] += 1

    rewrite = _text(outcome.get("rewrite"))
    if "rewrite" not in outcome:
        gaps["rewrite"] += 1

    journal = next(
        (
            entry.fields
            for entry in reversed(journals.get((generation, index), ()))
            if entry.fields["episode_id"] == episode_id
        ),
        None,
    )
    return {
        "key": _episode_key(generation, index),
        "generation": generation,
        "episode_index": index,
        "episode_id": episode_id,
        "committed_at": _number(row["committed_at"]),
        "stage": _stage(admitted, row["rejection_stage"], score, receipt_hash),
        "admitted": admitted,
        "training_success": bool(row["training_success"]),
        "rule_id": rule_hash if materialized and isinstance(rule_hash, str) else None,
        "parents": parents,
        "claimed_parents": _claimed_parents(archive_record),
        "inputs": None if inputs is None else sorted(set(inputs)),
        "score": score,
        "process": process,
        "stop_cause": _text(outcome.get("stop_cause")),
        "record": record,
        "rewrite": rewrite,
        "journal": None if journal is None else dict(journal),
    }


def _journal_only_episode(
    generation: int, index: int, entry: _JournalEntry
) -> dict[str, Any]:
    return {
        "key": _episode_key(generation, index),
        "generation": generation,
        "episode_index": index,
        "episode_id": entry.fields["episode_id"],
        "committed_at": None,
        "stage": "pending" if entry.has_commit_token else "crash",
        "admitted": False,
        "training_success": False,
        "rule_id": None,
        "parents": None,
        "claimed_parents": [],
        "inputs": None,
        "score": None,
        "process": None,
        "stop_cause": None,
        "record": None,
        "rewrite": None,
        "journal": dict(entry.fields),
    }


def _episode_key(generation: int, index: int) -> str:
    return f"g{generation}-e{index}"


def _stage(
    admitted: bool,
    rejection_stage: Any,
    score: Mapping[str, Any] | None,
    receipt_hash: Any,
) -> str:
    if admitted:
        return "admitted"
    if isinstance(rejection_stage, str) and rejection_stage:
        return rejection_stage
    if score is not None:
        return "rewrite" if score["marginal_pass"] is True else "error"
    return "rewrite" if isinstance(receipt_hash, str) and receipt_hash else "error"


def _stages(observed: Any) -> list[dict[str, Any]]:
    stages = [
        {"id": stage_id, "label": label, "sieve": True}
        for stage_id, label in _SIEVE_STAGES
    ] + [
        {"id": stage_id, "label": label, "sieve": False}
        for stage_id, label in _OFF_SIEVE_STAGES
    ]
    known = {stage["id"] for stage in stages}
    for stage_id in sorted(set(observed) - known):
        label = _OBSERVED_STAGE_LABELS.get(stage_id, stage_id)
        stages.append({"id": stage_id, "label": label, "sieve": False})
    return stages


def _score(receipt: Mapping[str, Any]) -> dict[str, Any]:
    axes: dict[str, dict[str, Any]] = {}
    diagnostics = receipt.get("axis_diagnostics")
    for item in diagnostics if isinstance(diagnostics, list) else ():
        if isinstance(item, Mapping) and isinstance(item.get("axis"), str):
            axes[item["axis"]] = {
                "stage2_gain": _number(item.get("stage2_gain")),
                "stage2_lower_bound": _number(item.get("stage2_lower_bound")),
                "system_count": _integer(item.get("system_count")),
            }
    return {
        "stage1_gain": _number(receipt.get("stage1_gain")),
        "stage1_threshold": _number(receipt.get("stage1_threshold")),
        "stage2_gain": _number(receipt.get("stage2_gain")),
        "stage2_lower_bound": _number(receipt.get("stage2_lower_bound")),
        "complexity": _integer(receipt.get("complexity")),
        "complexity_charge": _number(receipt.get("complexity_charge")),
        "signal_pass": _boolean(receipt.get("signal_pass")),
        "marginal_pass": _boolean(receipt.get("marginal_pass")),
        "archive_revision": _integer(receipt.get("archive_revision")),
        "axes": axes,
    }


def _process(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    raw_calls = value.get("calls")
    calls = {
        name: count
        for name, count in sorted(
            raw_calls.items() if isinstance(raw_calls, Mapping) else ()
        )
        if isinstance(name, str) and _integer(count) is not None
    }
    percentiles = value.get("probe_percentiles")
    if not isinstance(percentiles, list):
        percentiles = []
    return {
        "calls": calls,
        **{name: _integer(value.get(name)) for name in _PROCESS_COUNTS},
        "first_compile_success_attempt": _integer(
            value.get("first_compile_success_attempt")
        ),
        "probe_percentiles": [
            number
            for number in map(_number, percentiles)
            if number is not None
        ],
        "admission_percentile": _number(value.get("admission_percentile")),
    }


def _public_record(archive_record: Mapping[str, Any]) -> dict[str, Any] | None:
    record = {name: _text(archive_record.get(name)) for name in _RECORD_FIELDS}
    return record if any(value is not None for value in record.values()) else None


def _claimed_parents(archive_record: Mapping[str, Any]) -> list[str]:
    claimed = _string_list(archive_record.get("parent_hypothesis_ids")) or []
    return list(dict.fromkeys(claimed))


def _rule(revision: int, rule_hash: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    score = episode["score"]
    return {
        "rule_id": rule_hash,
        "revision": revision,
        "generation": episode["generation"],
        "episode_key": episode["key"],
        "inputs": episode["inputs"],
        "parents": episode["parents"],
        "stage2_gain": None if score is None else score["stage2_gain"],
    }


def _edges(
    episodes: Sequence[Mapping[str, Any]], archived_rules: set[str]
) -> list[dict[str, str]]:
    edges = []
    for episode in episodes:
        for parent in episode["parents"] or ():
            edges.append({"parent": parent, "child": episode["key"], "kind": "trusted"})
        for parent in episode["claimed_parents"]:
            if parent in archived_rules and parent != episode["rule_id"]:
                edges.append(
                    {"parent": parent, "child": episode["key"], "kind": "claimed"}
                )
    return edges


def _inputs(
    loaded: LoadedDataset, requested: str, frozen_path: Path
) -> dict[str, Any]:
    registry = loaded.registry
    registry_hash = registry.registry_hash
    universe_hash = loaded.universe.universe_hash
    names = [descriptor.name for descriptor in registry.descriptors]
    frozen = _read_frozen_groups(frozen_path)
    if frozen is not None:
        if (
            frozen["registry_hash"] != registry_hash
            or frozen["universe_hash"] != universe_hash
        ):
            raise ValueError(
                f"input groups frozen in {frozen_path} belong to a different dataset "
                "(registry or universe hash mismatch); refusing to regroup mid-run"
            )
        if requested not in ("auto", frozen["grouping"]):
            raise ValueError(
                f"input groups in {frozen_path} are frozen as "
                f"{frozen['grouping']!r}, not {requested!r}"
            )
        frozen_names = [name for group in frozen["groups"] for name in group["inputs"]]
        if sorted(frozen_names) != sorted(names):
            raise ValueError(
                f"frozen input groups in {frozen_path} do not cover the registry"
            )
        grouping, groups = frozen["grouping"], frozen["groups"]
    else:
        grouping = requested
        if grouping == "auto":
            families = {descriptor.family for descriptor in registry.descriptors}
            grouping = "family" if len(families) >= 2 else "data"
        groups = (
            _family_groups(registry) if grouping == "family" else _data_groups(loaded)
        )
        _write_json_atomic(
            frozen_path,
            {
                "grouping": grouping,
                "registry_hash": registry_hash,
                "universe_hash": universe_hash,
                "groups": groups,
            },
        )
    return {
        "grouping": grouping,
        "registry_hash": registry_hash,
        "universe_hash": universe_hash,
        "groups": groups,
        "descriptors": [
            {
                "name": descriptor.name,
                "family": descriptor.family,
                "unit": descriptor.unit,
                "value_type": descriptor.value_type,
            }
            for descriptor in registry.descriptors
        ],
    }


def _read_frozen_groups(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(f"frozen input groups file {path} is unreadable") from exc
    malformed = ValueError(f"frozen input groups file {path} is malformed")
    if not isinstance(raw, dict) or raw.get("grouping") not in ("family", "data"):
        raise malformed
    hashes = (raw.get("registry_hash"), raw.get("universe_hash"))
    if not all(isinstance(value, str) for value in hashes):
        raise malformed
    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list):
        raise malformed
    groups = []
    for group in raw_groups:
        if not isinstance(group, dict):
            raise malformed
        inputs = _string_list(group.get("inputs"))
        labels = (group.get("id"), group.get("label"))
        if inputs is None or not all(isinstance(value, str) for value in labels):
            raise malformed
        groups.append({"id": group["id"], "label": group["label"], "inputs": inputs})
    return {
        "grouping": raw["grouping"],
        "registry_hash": raw["registry_hash"],
        "universe_hash": raw["universe_hash"],
        "groups": groups,
    }


def _family_groups(registry: DescriptorRegistry) -> list[dict[str, Any]]:
    families = sorted({descriptor.family for descriptor in registry.descriptors})
    return [
        {
            "id": family,
            "label": family,
            "inputs": sorted(
                descriptor.name
                for descriptor in registry.descriptors
                if descriptor.family == family
            ),
        }
        for family in families
    ]


def _data_groups(loaded: LoadedDataset) -> list[dict[str, Any]]:
    names = [descriptor.name for descriptor in loaded.registry.descriptors]
    if not names:
        return []
    candidates = loaded.universe.candidates
    matrix = np.array(
        [
            [float(candidate.descriptors[name]) for name in names]
            for candidate in candidates
        ],
        dtype=float,
    ).reshape(len(candidates), len(names))
    clusters = _average_linkage(
        names, 1.0 - _absolute_spearman(matrix), cut=GROUP_CUT_DISTANCE
    )
    return [
        {"id": f"cluster-{number}", "label": f"Cluster {number}", "inputs": members}
        for number, members in enumerate(clusters, start=1)
    ]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def _absolute_spearman(matrix: np.ndarray) -> np.ndarray:
    rows, columns = matrix.shape
    if rows < 2:
        return np.eye(columns)
    ranks = np.column_stack(
        [_average_ranks(matrix[:, column]) for column in range(columns)]
    )
    centered = ranks - ranks.mean(axis=0)
    scale = np.sqrt((centered**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (centered.T @ centered) / np.outer(scale, scale)
    rho = np.nan_to_num(np.abs(rho), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(rho, 1.0)
    return np.clip(rho, 0.0, 1.0)


def _average_linkage(
    names: Sequence[str], distance: np.ndarray, *, cut: float
) -> list[list[str]]:
    """Agglomerate deterministically; clusters are keyed by their minimum leaf."""
    position = {name: index for index, name in enumerate(names)}
    members = {name: [name] for name in names}
    between = {
        (first, second): float(distance[position[first], position[second]])
        for first, second in itertools.combinations(sorted(names), 2)
    }
    flat: list[list[str]] | None = None
    while len(members) > 1:
        (left, right), height = min(
            between.items(), key=lambda item: (round(item[1], 12), item[0])
        )
        if flat is None and round(height, 12) > cut:
            flat = [list(group) for group in members.values()]
        left_size, right_size = len(members[left]), len(members[right])
        del between[(left, right)]
        for other in members:
            if other in (left, right):
                continue
            joined = (
                left_size * between.pop(_pair(left, other))
                + right_size * between.pop(_pair(right, other))
            ) / (left_size + right_size)
            between[_pair(left, other)] = joined
        members[left] = members[left] + members.pop(right)
    order = next(iter(members.values()))
    rank = {name: index for index, name in enumerate(order)}
    return sorted(flat or [order], key=lambda group: rank[group[0]])


def _pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    write_text_atomic(path, text + "\n")


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
