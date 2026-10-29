"""Transactional archive storage for the materials adsorption task."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.task.types import (
    TaskCommitPreparation,
    TaskReward,
    TrustedEpisodeIdentity,
)


_SCHEMA_VERSION = 1
_CHECKPOINT_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    """Immutable, content-addressed byte storage backed by SHA-256 files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        if not isinstance(data, bytes):
            raise TypeError("blob data must be bytes")

        digest = _sha256(data)
        directory = self.root / digest[:2]
        path = directory / digest
        directory.mkdir(parents=True, exist_ok=True)

        if path.exists():
            self._verify_path(path, digest)
            return digest

        temporary = directory / f".{digest}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # A hard link publishes without ever replacing an existing blob.
                os.link(temporary, path)
                self._fsync_directory(directory)
            except FileExistsError:
                self._verify_path(path, digest)
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        data = path.read_bytes()
        if _sha256(data) != digest:
            raise RuntimeError(f"blob hash mismatch for {digest}")
        return data

    def _path(self, digest: str) -> Path:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX_DIGITS for character in digest)
        ):
            raise ValueError("blob digest must be a lowercase SHA-256 hex digest")
        return self.root / digest[:2] / digest

    @staticmethod
    def _verify_path(path: Path, digest: str) -> None:
        if _sha256(path.read_bytes()) != digest:
            raise RuntimeError(f"blob hash mismatch for {digest}")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class ArchiveCommitResult:
    """Durable outcome of applying one episode commit intent."""

    training_success: bool
    archive_admission: bool
    rejection_stage: str | None
    archive_revision: int


class ArchiveStore:
    """SQLite WAL archive with crash-recoverable episode commit intents."""

    def __init__(self, database_path: str | Path, blob_dir: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(blob_dir)
        self._initialize()

    def prepare_episode_commit(
        self,
        *,
        identity: TrustedEpisodeIdentity,
        training_success: bool,
        requested_admission: bool,
        canonical_rule_hash: str,
        behavior_hash: str | None,
        consequence_hash: str | None,
        receipt_hash: str | None,
        outcome: Mapping[str, Any],
        archive_record: Mapping[str, Any],
    ) -> TaskCommitPreparation:
        """Persist an invisible intent and return the reward to be journaled."""
        if not canonical_rule_hash:
            raise ValueError("canonical_rule_hash must not be empty")
        if requested_admission and not behavior_hash:
            raise ValueError("behavior_hash is required for archive admission")

        identity_value = self._identity_value(identity)
        request = {
            "identity": identity_value,
            "training_success": bool(training_success),
            "requested_admission": bool(requested_admission),
            "canonical_rule_hash": canonical_rule_hash,
            "behavior_hash": behavior_hash,
            "consequence_hash": consequence_hash,
            "receipt_hash": receipt_hash,
            "outcome": dict(outcome),
            "archive_record": dict(archive_record),
        }
        request_json = _canonical_json(request)
        request_digest = _sha256(request_json.encode("utf-8"))
        token = _sha256(
            ("materials-adsorption-episode\0" + _canonical_json(identity_value)).encode(
                "utf-8"
            )
        )

        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT token, request_digest, intent_digest, status, payload_json,
                       result_json
                FROM commit_intents
                WHERE run_id = ? AND generation_id = ? AND episode_index = ?
                """,
                (
                    identity_value["run_id"],
                    identity_value["generation_id"],
                    identity_value["episode_index"],
                ),
            ).fetchone()
            if existing is not None:
                if existing[3] == "discarded":
                    # A discarded unpublished attempt releases its deterministic
                    # slot when that slot is retried.
                    connection.execute(
                        "DELETE FROM commit_intents WHERE token = ?", (existing[0],)
                    )
                    existing = None
                elif existing[1] != request_digest:
                    raise RuntimeError(
                        "episode identity is already associated with a different intent"
                    )
                else:
                    return self._preparation_from_row(existing)

            effective_success = bool(training_success or requested_admission)
            proposed_admission = False
            if requested_admission:
                duplicate = connection.execute(
                    """
                    SELECT 1 FROM archive_records
                    WHERE canonical_rule_hash = ? OR behavior_hash = ?
                    UNION ALL
                    SELECT 1 FROM commit_intents
                    WHERE reserved_rule_hash = ? OR reserved_behavior_hash = ?
                    LIMIT 1
                    """,
                    (
                        canonical_rule_hash,
                        behavior_hash,
                        canonical_rule_hash,
                        behavior_hash,
                    ),
                ).fetchone()
                proposed_admission = duplicate is None

            rejection_stage = self._rejection_stage(
                requested_admission=bool(requested_admission),
                proposed_admission=proposed_admission,
                outcome=outcome,
            )
            payload = dict(request)
            payload.update(
                {
                    "effective_training_success": effective_success,
                    "proposed_admission": proposed_admission,
                    "rejection_stage": rejection_stage,
                }
            )
            payload_json = _canonical_json(payload)
            intent_digest = _sha256(payload_json.encode("utf-8"))
            now = time.time()
            connection.execute(
                """
                INSERT INTO commit_intents (
                    token, request_digest, intent_digest, run_id, generation_id,
                    episode_index, episode_id, payload_json, status,
                    reserved_rule_hash, reserved_behavior_hash, prepared_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?)
                """,
                (
                    token,
                    request_digest,
                    intent_digest,
                    identity_value["run_id"],
                    identity_value["generation_id"],
                    identity_value["episode_index"],
                    identity_value["episode_id"],
                    payload_json,
                    canonical_rule_hash if proposed_admission else None,
                    behavior_hash if proposed_admission else None,
                    now,
                ),
            )
            return self._make_preparation(
                token,
                intent_digest,
                effective_success,
                proposed_admission,
                rejection_stage,
            )

    def apply_episode_commit(
        self, token: str, intent_digest: str
    ) -> ArchiveCommitResult:
        """Atomically publish a prepared outcome and optional archive revision."""
        with self._transaction() as connection:
            row = self._validated_intent(connection, token, intent_digest)
            return self._apply_intent(connection, row)

    def recover_episode_commits(self, journaled: Mapping[str, str]) -> None:
        """Apply journaled preparations and discard all other preparations."""
        journaled_values = dict(journaled)
        with self._transaction() as connection:
            rows = {
                row[0]: row
                for row in connection.execute(
                    """
                    SELECT token, intent_digest, status, payload_json, result_json
                    FROM commit_intents
                    """
                )
            }
            for token, digest in journaled_values.items():
                row = rows.get(token)
                if row is None:
                    raise RuntimeError(f"unknown journaled episode commit token: {token}")
                if row[1] != digest:
                    raise RuntimeError(f"intent digest mismatch for token {token}")
                if row[2] == "discarded":
                    raise RuntimeError(f"journal references discarded intent: {token}")

            now = time.time()
            for token, row in rows.items():
                if row[2] == "prepared" and token not in journaled_values:
                    connection.execute(
                        """
                            UPDATE commit_intents
                            SET status = 'discarded', reserved_rule_hash = NULL,
                                reserved_behavior_hash = NULL,
                                discarded_at = ?
                        WHERE token = ?
                        """,
                        (now, token),
                    )
            for token, digest in journaled_values.items():
                row = self._validated_intent(connection, token, digest)
                self._apply_intent(connection, row)

    def intent_status(self, token: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM commit_intents WHERE token = ?", (token,)
            ).fetchone()
        return None if row is None else str(row[0])

    def archive_revision(self) -> int:
        with self._connect() as connection:
            return self._metadata_int(connection, "archive_revision")

    def outcome_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0])

    def scoring_snapshot(self) -> tuple[int, tuple[tuple[str, str], ...]]:
        """Read the revision and its immutable consequence references atomically."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            try:
                revision = self._metadata_int(connection, "archive_revision")
                rows = connection.execute(
                    """
                    SELECT canonical_rule_hash, consequence_hash
                    FROM archive_records
                    WHERE consequence_hash IS NOT NULL
                    ORDER BY revision
                    """
                ).fetchall()
            finally:
                connection.rollback()
        return revision, tuple(
            (str(rule_hash), str(blob_hash)) for rule_hash, blob_hash in rows
        )

    def admitted_records(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_json FROM archive_records ORDER BY revision"
            ).fetchall()
        return tuple(json.loads(str(row[0])) for row in rows)

    def register_evidence_sources(self, source_ids: Iterable[str]) -> None:
        values = list(dict.fromkeys(source_ids))
        if any(not isinstance(source_id, str) or not source_id for source_id in values):
            raise ValueError("evidence source IDs must be non-empty strings")
        with self._transaction() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO evidence_sources (source_id) VALUES (?)",
                ((source_id,) for source_id in values),
            )

    def assign_evidence_source(
        self, source_ids: Iterable[str] | None = None
    ) -> str:
        """Choose a least-attempted source and record an attempt, not a lease."""
        allowed = None if source_ids is None else list(dict.fromkeys(source_ids))
        if allowed is not None and not allowed:
            raise ValueError("at least one evidence source ID is required")
        with self._transaction() as connection:
            query = """
                SELECT source_id FROM evidence_sources
            """
            parameters: tuple[str, ...] = ()
            if allowed is not None:
                placeholders = ", ".join("?" for _ in allowed)
                query += f" WHERE source_id IN ({placeholders})"
                parameters = tuple(allowed)
            query += " ORDER BY attempt_count, registered_order LIMIT 1"
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                raise RuntimeError("no evidence sources are registered")
            source_id = str(row[0])
            connection.execute(
                """
                UPDATE evidence_sources
                SET attempt_count = attempt_count + 1
                WHERE source_id = ?
                """,
                (source_id,),
            )
            return source_id

    def evidence_attempt_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, attempt_count FROM evidence_sources
                ORDER BY registered_order
                """
            ).fetchall()
        return {str(source_id): int(count) for source_id, count in rows}

    def register_parent(self, parent_id: str, *, quality: float) -> None:
        if not parent_id:
            raise ValueError("parent_id must not be empty")
        quality_value = float(quality)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO parents (parent_id, quality) VALUES (?, ?)
                ON CONFLICT(parent_id) DO UPDATE SET quality = excluded.quality
                """,
                (parent_id, quality_value),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO parent_pair_attempts (parent_a, parent_b)
                SELECT first.parent_id, second.parent_id
                FROM parents AS first CROSS JOIN parents AS second
                WHERE first.parent_id != second.parent_id
                """
            )

    def assign_parent_pair(self) -> tuple[str, str]:
        """Choose and increment an ordered pair without reserving either parent."""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT pairs.parent_a, pairs.parent_b, pairs.attempt_count,
                       first.quality, second.quality,
                       first.registered_order, second.registered_order
                FROM parent_pair_attempts AS pairs
                JOIN parents AS first ON first.parent_id = pairs.parent_a
                JOIN parents AS second ON second.parent_id = pairs.parent_b
                ORDER BY first.registered_order, second.registered_order
                """
            ).fetchall()
            if not rows:
                raise RuntimeError("at least two parents must be registered")

            uses: dict[str, int] = {}
            for parent_a, parent_b, attempts, *_unused in rows:
                uses[str(parent_a)] = uses.get(str(parent_a), 0) + int(attempts)
                uses[str(parent_b)] = uses.get(str(parent_b), 0) + int(attempts)

            def priority(row: sqlite3.Row) -> float:
                parent_a, parent_b, attempts, first_quality, second_quality, *_ = row
                base = max(0.0, float(first_quality)) + max(
                    0.0, float(second_quality)
                )
                base = max(base, 1e-12)
                return (
                    base
                    / (1.0 + int(attempts))
                    / (1.0 + 0.05 * uses.get(str(parent_a), 0))
                    / (1.0 + 0.05 * uses.get(str(parent_b), 0))
                )

            chosen = max(rows, key=priority)
            parent_a, parent_b = str(chosen[0]), str(chosen[1])
            connection.execute(
                """
                UPDATE parent_pair_attempts
                SET attempt_count = attempt_count + 1
                WHERE parent_a = ? AND parent_b = ?
                """,
                (parent_a, parent_b),
            )
            return parent_a, parent_b

    def parent_pair_attempt_count(self, parent_a: str, parent_b: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attempt_count FROM parent_pair_attempts
                WHERE parent_a = ? AND parent_b = ?
                """,
                (parent_a, parent_b),
            ).fetchone()
        return 0 if row is None else int(row[0])

    def generation_checkpoint_state(self) -> dict[str, Any]:
        """Flush the WAL and return an integrity manifest for this database."""
        with self._connect() as connection:
            checkpoint = connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise RuntimeError("archive WAL checkpoint could not complete")
            manifest = self._checkpoint_manifest(connection)
        manifest["manifest_digest"] = _sha256(
            _canonical_json(manifest).encode("utf-8")
        )
        return manifest

    def load_generation_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            raise RuntimeError("invalid archive checkpoint state")
        with self._connect() as connection:
            current = self._checkpoint_manifest(connection)

        if state.get("archive_identity") != current["archive_identity"]:
            raise RuntimeError("archive identity does not match checkpoint")
        if state.get("checkpoint_version") != _CHECKPOINT_VERSION:
            raise RuntimeError("unsupported archive checkpoint version")
        digest = state.get("manifest_digest")
        unsigned = {key: value for key, value in state.items() if key != "manifest_digest"}
        if digest != _sha256(_canonical_json(unsigned).encode("utf-8")):
            raise RuntimeError("archive checkpoint manifest hash mismatch")
        monotonic_fields = (
            "archive_revision",
            "outcome_count",
            "applied_intent_count",
            "evidence_attempt_count",
            "parent_pair_attempt_count",
        )
        if any(int(current[key]) < int(unsigned[key]) for key in monotonic_fields):
            raise RuntimeError("archive checkpoint state is ahead of database")

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS commit_intents (
                    token TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    intent_digest TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    generation_id INTEGER NOT NULL,
                    episode_index INTEGER NOT NULL,
                    episode_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('prepared', 'applied', 'discarded')
                    ),
                    reserved_rule_hash TEXT UNIQUE,
                    reserved_behavior_hash TEXT UNIQUE,
                    result_json TEXT,
                    prepared_at REAL NOT NULL,
                    applied_at REAL,
                    discarded_at REAL,
                    UNIQUE (run_id, generation_id, episode_index)
                );

                CREATE TABLE IF NOT EXISTS archive_records (
                    revision INTEGER PRIMARY KEY,
                    token TEXT NOT NULL UNIQUE,
                    canonical_rule_hash TEXT NOT NULL UNIQUE,
                    behavior_hash TEXT NOT NULL UNIQUE,
                    consequence_hash TEXT,
                    receipt_hash TEXT,
                    record_json TEXT NOT NULL,
                    committed_at REAL NOT NULL,
                    FOREIGN KEY (token) REFERENCES commit_intents(token)
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    run_id TEXT NOT NULL,
                    generation_id INTEGER NOT NULL,
                    episode_index INTEGER NOT NULL,
                    episode_id TEXT NOT NULL,
                    token TEXT NOT NULL UNIQUE,
                    training_success INTEGER NOT NULL,
                    requested_admission INTEGER NOT NULL,
                    archive_admission INTEGER NOT NULL,
                    rejection_stage TEXT,
                    canonical_rule_hash TEXT NOT NULL,
                    behavior_hash TEXT,
                    consequence_hash TEXT,
                    receipt_hash TEXT,
                    outcome_json TEXT NOT NULL,
                    commit_revision INTEGER,
                    committed_at REAL NOT NULL,
                    PRIMARY KEY (run_id, generation_id, episode_index),
                    FOREIGN KEY (token) REFERENCES commit_intents(token)
                );

                CREATE TABLE IF NOT EXISTS evidence_sources (
                    registered_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL UNIQUE,
                    attempt_count INTEGER NOT NULL DEFAULT 0
                        CHECK (attempt_count >= 0)
                );

                CREATE TABLE IF NOT EXISTS parents (
                    registered_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id TEXT NOT NULL UNIQUE,
                    quality REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS parent_pair_attempts (
                    parent_a TEXT NOT NULL,
                    parent_b TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0
                        CHECK (attempt_count >= 0),
                    PRIMARY KEY (parent_a, parent_b),
                    CHECK (parent_a != parent_b),
                    FOREIGN KEY (parent_a) REFERENCES parents(parent_id),
                    FOREIGN KEY (parent_b) REFERENCES parents(parent_id)
                );
                """
            )
            with self._transaction(connection) as transaction:
                transaction.execute(
                    "INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
                transaction.execute(
                    "INSERT OR IGNORE INTO metadata (key, value) VALUES ('archive_revision', '0')"
                )
                transaction.execute(
                    "INSERT OR IGNORE INTO metadata (key, value) VALUES ('archive_identity', ?)",
                    (uuid.uuid4().hex,),
                )
                schema_version = self._metadata_int(transaction, "schema_version")
                if schema_version != _SCHEMA_VERSION:
                    raise RuntimeError(
                        f"unsupported archive schema version: {schema_version}"
                    )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.database_path,
            isolation_level=None,
            timeout=30.0,
        )
        try:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(
        self, connection: sqlite3.Connection | None = None
    ) -> Iterator[sqlite3.Connection]:
        if connection is None:
            with self._connect() as owned:
                with self._transaction(owned) as transaction:
                    yield transaction
            return

        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    @staticmethod
    def _identity_value(identity: TrustedEpisodeIdentity) -> dict[str, Any]:
        return {
            "run_id": identity.run_id,
            "generation_id": identity.generation_id,
            "episode_index": identity.episode_index,
            "episode_id": identity.episode_id,
        }

    @staticmethod
    def _rejection_stage(
        *,
        requested_admission: bool,
        proposed_admission: bool,
        outcome: Mapping[str, Any],
    ) -> str | None:
        if proposed_admission:
            return None
        if requested_admission:
            return "archive_gate"
        stage = outcome.get("rejection_stage")
        return stage if isinstance(stage, str) and stage else None

    @staticmethod
    def _make_preparation(
        token: str,
        intent_digest: str,
        training_success: bool,
        archive_admission: bool,
        rejection_stage: str | None,
    ) -> TaskCommitPreparation:
        reward = TaskReward(
            value=1.0 if training_success else 0.0,
            success=training_success,
            breakdown={
                "training_success": training_success,
                "archive_admission": archive_admission,
                "admitted": archive_admission,
                "rejection_stage": rejection_stage,
            },
        )
        return TaskCommitPreparation(
            token=token,
            intent_digest=intent_digest,
            reward=reward,
        )

    def _preparation_from_row(self, row: sqlite3.Row) -> TaskCommitPreparation:
        token, _request_digest, intent_digest, status, payload_json, result_json = row
        if status == "applied":
            result = self._result_from_json(str(result_json))
            return self._make_preparation(
                str(token),
                str(intent_digest),
                result.training_success,
                result.archive_admission,
                result.rejection_stage,
            )
        payload = json.loads(str(payload_json))
        return self._make_preparation(
            str(token),
            str(intent_digest),
            bool(payload["effective_training_success"]),
            bool(payload["proposed_admission"]),
            payload.get("rejection_stage"),
        )

    @staticmethod
    def _validated_intent(
        connection: sqlite3.Connection, token: str, intent_digest: str
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT token, intent_digest, status, payload_json, result_json
            FROM commit_intents WHERE token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"unknown episode commit token: {token}")
        if row[1] != intent_digest:
            raise RuntimeError(f"intent digest mismatch for token {token}")
        return row

    def _apply_intent(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ArchiveCommitResult:
        token, _intent_digest, status, payload_json, result_json = row
        if status == "applied":
            return self._result_from_json(str(result_json))
        if status == "discarded":
            raise RuntimeError(f"cannot apply discarded intent: {token}")

        payload = json.loads(str(payload_json))
        identity = payload["identity"]
        archive_admission = bool(payload["proposed_admission"])
        revision = self._metadata_int(connection, "archive_revision")
        commit_revision: int | None = None
        now = time.time()

        if archive_admission:
            commit_revision = revision + 1
            connection.execute(
                """
                INSERT INTO archive_records (
                    revision, token, canonical_rule_hash, behavior_hash, consequence_hash,
                    receipt_hash, record_json, committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commit_revision,
                    token,
                    payload["canonical_rule_hash"],
                    payload["behavior_hash"],
                    payload["consequence_hash"],
                    payload["receipt_hash"],
                    _canonical_json(payload["archive_record"]),
                    now,
                ),
            )
            connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'archive_revision'",
                (str(commit_revision),),
            )
            revision = commit_revision

        result = ArchiveCommitResult(
            training_success=bool(payload["effective_training_success"]),
            archive_admission=archive_admission,
            rejection_stage=payload.get("rejection_stage"),
            archive_revision=revision,
        )
        connection.execute(
            """
            INSERT INTO outcomes (
                run_id, generation_id, episode_index, episode_id, token,
                training_success, requested_admission, archive_admission,
                rejection_stage, canonical_rule_hash, consequence_hash,
                behavior_hash, receipt_hash, outcome_json, commit_revision, committed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity["run_id"],
                identity["generation_id"],
                identity["episode_index"],
                identity["episode_id"],
                token,
                int(result.training_success),
                int(payload["requested_admission"]),
                int(result.archive_admission),
                result.rejection_stage,
                payload["canonical_rule_hash"],
                payload["consequence_hash"],
                payload["behavior_hash"],
                payload["receipt_hash"],
                _canonical_json(payload["outcome"]),
                commit_revision,
                now,
            ),
        )
        result_json_value = _canonical_json(
            {
                "training_success": result.training_success,
                "archive_admission": result.archive_admission,
                "rejection_stage": result.rejection_stage,
                "archive_revision": result.archive_revision,
            }
        )
        connection.execute(
            """
            UPDATE commit_intents
            SET status = 'applied', result_json = ?, applied_at = ?
            WHERE token = ?
            """,
            (result_json_value, now, token),
        )
        return result

    @staticmethod
    def _result_from_json(result_json: str) -> ArchiveCommitResult:
        value = json.loads(result_json)
        return ArchiveCommitResult(
            training_success=bool(value["training_success"]),
            archive_admission=bool(value["archive_admission"]),
            rejection_stage=value.get("rejection_stage"),
            archive_revision=int(value["archive_revision"]),
        )

    @staticmethod
    def _metadata_int(connection: sqlite3.Connection, key: str) -> int:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"archive metadata is missing {key!r}")
        return int(row[0])

    def _checkpoint_manifest(self, connection: sqlite3.Connection) -> dict[str, Any]:
        identity_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'archive_identity'"
        ).fetchone()
        if identity_row is None:
            raise RuntimeError("archive metadata is missing archive identity")
        return {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "schema_version": self._metadata_int(connection, "schema_version"),
            "archive_identity": str(identity_row[0]),
            "archive_revision": self._metadata_int(connection, "archive_revision"),
            "outcome_count": int(
                connection.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            ),
            "applied_intent_count": int(
                connection.execute(
                    "SELECT COUNT(*) FROM commit_intents WHERE status = 'applied'"
                ).fetchone()[0]
            ),
            "prepared_intent_count": int(
                connection.execute(
                    "SELECT COUNT(*) FROM commit_intents WHERE status = 'prepared'"
                ).fetchone()[0]
            ),
            "evidence_attempt_count": int(
                connection.execute(
                    "SELECT COALESCE(SUM(attempt_count), 0) FROM evidence_sources"
                ).fetchone()[0]
            ),
            "parent_pair_attempt_count": int(
                connection.execute(
                    "SELECT COALESCE(SUM(attempt_count), 0) FROM parent_pair_attempts"
                ).fetchone()[0]
            ),
        }
