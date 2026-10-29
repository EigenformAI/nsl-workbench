from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.task.types import TrustedEpisodeIdentity
from tasks.materials_adsorption.archive import ArchiveStore, BlobStore


def _identity(index: int) -> TrustedEpisodeIdentity:
    return TrustedEpisodeIdentity("run-a", 3, index, f"ep-{index}")


def _intent(
    store: ArchiveStore,
    index: int,
    *,
    rule_hash: str,
    behavior_hash: str | None = None,
    admit: bool = True,
):
    return store.prepare_episode_commit(
        identity=_identity(index),
        training_success=True,
        requested_admission=admit,
        canonical_rule_hash=rule_hash,
        behavior_hash=behavior_hash or f"behavior-{index}",
        consequence_hash=f"consequence-{index}",
        receipt_hash=f"receipt-{index}",
        outcome={"public": "ok", "private_score": index},
        archive_record={"hypothesis_id": f"hyp-{index}"},
    )


def test_blob_store_is_content_addressed_and_verifies_reads(tmp_path) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    first = blobs.put(b"consequence")
    second = blobs.put(b"consequence")

    assert first == second
    assert blobs.get(first) == b"consequence"
    (tmp_path / "blobs" / first[:2] / first).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        blobs.get(first)


def test_prepared_intent_is_not_visible_until_framework_journals_and_applies(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    prepared = _intent(store, 0, rule_hash="rule-a")

    assert store.archive_revision() == 0
    assert store.outcome_count() == 0

    applied = store.apply_episode_commit(prepared.token, prepared.intent_digest)

    assert applied.archive_admission
    assert store.archive_revision() == 1
    assert store.outcome_count() == 1
    assert store.apply_episode_commit(prepared.token, prepared.intent_digest) == applied


def test_recovery_discards_unjournaled_and_applies_journaled_intents(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    journaled = _intent(store, 0, rule_hash="rule-a")
    unjournaled = _intent(store, 1, rule_hash="rule-b")

    store.recover_episode_commits({journaled.token: journaled.intent_digest})

    assert store.intent_status(journaled.token) == "applied"
    assert store.intent_status(unjournaled.token) == "discarded"
    assert store.archive_revision() == 1
    assert store.outcome_count() == 1


def test_discarded_episode_slot_can_be_prepared_again(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    first = _intent(store, 0, rule_hash="rule-a")
    store.recover_episode_commits({})

    retried = store.prepare_episode_commit(
        identity=TrustedEpisodeIdentity("run-a", 3, 0, "ep-retry"),
        training_success=True,
        requested_admission=True,
        canonical_rule_hash="rule-a",
        behavior_hash="behavior-0",
        consequence_hash="consequence-0",
        receipt_hash="receipt-0",
        outcome={"public": "ok", "private_score": 0},
        archive_record={"hypothesis_id": "hyp-0"},
    )

    assert retried.token != first.token
    assert store.intent_status(first.token) is None
    assert store.intent_status(retried.token) == "prepared"


def test_exact_duplicate_is_training_eligible_but_not_reserved_for_admission(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    first = _intent(store, 0, rule_hash="same-rule")
    store.apply_episode_commit(first.token, first.intent_digest)

    duplicate = _intent(store, 1, rule_hash="same-rule")
    result = store.apply_episode_commit(duplicate.token, duplicate.intent_digest)

    assert result.training_success
    assert not result.archive_admission
    assert result.rejection_stage == "archive_gate"
    assert store.archive_revision() == 1
    assert store.outcome_count() == 2


def test_behavioral_duplicate_is_blocked_transactionally(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    first = _intent(store, 0, rule_hash="rule-a", behavior_hash="same-behavior")
    store.apply_episode_commit(first.token, first.intent_digest)

    duplicate = _intent(store, 1, rule_hash="rule-b", behavior_hash="same-behavior")
    result = store.apply_episode_commit(duplicate.token, duplicate.intent_digest)

    assert result.training_success
    assert not result.archive_admission
    assert result.rejection_stage == "archive_gate"
    assert store.archive_revision() == 1


def test_scheduler_counts_attempts_without_exclusive_reservations(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    store.register_evidence_sources(["source-a", "source-b"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        sources = list(pool.map(lambda _: store.assign_evidence_source(), range(40)))

    assert set(sources) == {"source-a", "source-b"}
    assert store.evidence_attempt_counts() == {"source-a": 20, "source-b": 20}

    store.register_parent("p1", quality=1.0)
    store.register_parent("p2", quality=1.0)
    assert store.assign_parent_pair() == ("p1", "p2")
    assert store.assign_parent_pair() == ("p2", "p1")
    assert store.parent_pair_attempt_count("p1", "p2") == 1
    assert store.parent_pair_attempt_count("p2", "p1") == 1


def test_scheduler_can_ignore_sources_from_an_older_shard_scheme(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    store.register_evidence_sources(["old-source", "new-a", "new-b"])

    selected = [store.assign_evidence_source(["new-a", "new-b"]) for _ in range(4)]

    assert selected == ["new-a", "new-b", "new-a", "new-b"]
    assert store.evidence_attempt_counts() == {
        "old-source": 0,
        "new-a": 2,
        "new-b": 2,
    }


def test_checkpoint_manifest_round_trips_and_detects_wrong_database(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    prepared = _intent(store, 0, rule_hash="rule-a")
    store.apply_episode_commit(prepared.token, prepared.intent_digest)

    checkpoint = store.generation_checkpoint_state()
    json.dumps(checkpoint)
    store.load_generation_checkpoint_state(checkpoint)

    other = ArchiveStore(tmp_path / "other.sqlite", tmp_path / "other-blobs")
    with pytest.raises(RuntimeError, match="archive identity"):
        other.load_generation_checkpoint_state(checkpoint)


def test_checkpoint_restore_accepts_journal_ahead_database_state(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    checkpoint = store.generation_checkpoint_state()
    prepared = _intent(store, 0, rule_hash="rule-after-checkpoint")
    store.apply_episode_commit(prepared.token, prepared.intent_digest)

    # The framework journal is durable before apply. A crash before the next
    # checkpoint therefore leaves legitimate task state ahead of this token.
    store.load_generation_checkpoint_state(checkpoint)


def test_scoring_snapshot_returns_revision_and_archive_from_one_view(tmp_path) -> None:
    store = ArchiveStore(tmp_path / "archive.sqlite", tmp_path / "blobs")
    first = _intent(store, 0, rule_hash="rule-a")
    store.apply_episode_commit(first.token, first.intent_digest)

    revision, consequences = store.scoring_snapshot()

    assert revision == 1
    assert consequences == (("rule-a", "consequence-0"),)


def test_scoring_snapshot_stays_atomic_when_writer_commits_between_reads(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "archive.sqlite"
    blob_dir = tmp_path / "blobs"
    reader = ArchiveStore(database_path, blob_dir)
    first = _intent(reader, 0, rule_hash="rule-a")
    reader.apply_episode_commit(first.token, first.intent_digest)

    writer = ArchiveStore(database_path, blob_dir)
    second = _intent(writer, 1, rule_hash="rule-b")
    revision_read = threading.Event()
    writer_committed = threading.Event()
    original_metadata_int = reader._metadata_int

    def pause_after_revision_read(connection, key: str) -> int:
        value = original_metadata_int(connection, key)
        if key == "archive_revision":
            revision_read.set()
            assert writer_committed.wait(timeout=5), "writer did not commit"
        return value

    monkeypatch.setattr(reader, "_metadata_int", pause_after_revision_read)

    with ThreadPoolExecutor(max_workers=1) as pool:
        snapshot_future = pool.submit(reader.scoring_snapshot)
        assert revision_read.wait(timeout=5), "snapshot did not read its revision"
        try:
            committed = writer.apply_episode_commit(
                second.token, second.intent_digest
            )
        finally:
            writer_committed.set()
        snapshot = snapshot_future.result(timeout=5)

    assert committed.archive_revision == 2
    assert snapshot == (1, (("rule-a", "consequence-0"),))
    assert writer.scoring_snapshot() == (
        2,
        (
            ("rule-a", "consequence-0"),
            ("rule-b", "consequence-1"),
        ),
    )
