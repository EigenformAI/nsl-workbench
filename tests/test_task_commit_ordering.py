from __future__ import annotations

import pytest

from src.execution.generation_persistence import (
    JournaledTaskCommitError,
    journal_apply_and_account,
)


def test_task_commit_protocol_orders_journal_before_apply_before_accounting() -> None:
    events: list[str] = []

    journal_apply_and_account(
        append_journal=lambda: events.append("journal"),
        apply_task=lambda: events.append("apply"),
        account_episode=lambda: events.append("account"),
    )

    assert events == ["journal", "apply", "account"]


def test_apply_failure_still_accounts_durably_journaled_episode() -> None:
    events: list[str] = []

    def fail_apply() -> None:
        events.append("apply")
        raise RuntimeError("apply failed")

    with pytest.raises(JournaledTaskCommitError, match="apply failed"):
        journal_apply_and_account(
            append_journal=lambda: events.append("journal"),
            apply_task=fail_apply,
            account_episode=lambda: events.append("account"),
        )

    assert events == ["journal", "apply", "account"]


def test_journal_failure_does_not_apply_or_account() -> None:
    events: list[str] = []

    def fail_journal() -> None:
        events.append("journal")
        raise RuntimeError("journal failed")

    with pytest.raises(RuntimeError, match="journal failed"):
        journal_apply_and_account(
            append_journal=fail_journal,
            apply_task=lambda: events.append("apply"),
            account_episode=lambda: events.append("account"),
        )

    assert events == ["journal"]


def test_accounting_failure_is_classified_as_post_journal_failure() -> None:
    def fail_account() -> None:
        raise RuntimeError("account failed")

    with pytest.raises(JournaledTaskCommitError, match="account failed"):
        journal_apply_and_account(
            append_journal=lambda: None,
            apply_task=lambda: None,
            account_episode=fail_account,
        )
