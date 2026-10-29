from __future__ import annotations

import json

from src.harness.context_compaction import (
    ContextCompactionSettings,
    compact_messages,
    compact_messages_with_report,
    estimate_messages_tokens,
)
from src.typing.message import Message


def _tool_payload(output: object, *, success: bool = True) -> str:
    return json.dumps(
        {
            "output": output,
            "success": success,
            "error": None,
            "budget": {"task_tool_calls": {"remaining": 12}},
        }
    )


def _multi_tool_messages() -> list[Message]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nsl---run_python", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "nsl---inspect", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "nsl---run_python",
            "content": _tool_payload(
                {
                    "execution_id": "exec-1",
                    "artifact_files": ["/work/output/a.json"],
                    "stdout": "x" * 2000,
                    "dense_values": list(range(500)),
                }
            ),
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "nsl---inspect",
            "content": _tool_payload({"quality_score": 0.8, "summary": "recent"}),
        },
    ]


def test_compact_messages_noops_under_trigger_by_identity() -> None:
    messages = _multi_tool_messages()

    compacted = compact_messages(
        messages,
        settings=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=1_000_000,
            target_tokens=500_000,
        ),
    )

    assert compacted is messages


def test_compacts_old_tool_content_preserving_pairing_and_recent_tool() -> None:
    messages = _multi_tool_messages()
    recent_content = messages[-1]["content"]

    report = compact_messages_with_report(
        messages,
        settings=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=1,
            target_tokens=200,
            keep_recent_tool_outputs=1,
        ),
    )

    compacted = report.messages
    assert report.compacted
    assert report.compacted_tool_messages == 1
    assert compacted[2].get("tool_calls") == messages[2].get("tool_calls")
    assert compacted[3].get("tool_call_id") == "call_1"
    assert compacted[4].get("tool_call_id") == "call_2"
    assert compacted[4]["content"] == recent_content

    old_tool_payload = json.loads(compacted[3]["content"] or "{}")
    assert old_tool_payload["success"] is True
    assert old_tool_payload["error"] is None
    assert old_tool_payload["budget"] == {"task_tool_calls": {"remaining": 12}}
    assert old_tool_payload["output"]["execution_id"] == "exec-1"
    assert old_tool_payload["output"]["artifact_files"] == ["/work/output/a.json"]
    assert "context compaction" in old_tool_payload["output"]["stdout"]
    assert "context compaction" in old_tool_payload["output"]["dense_values"]


def test_compaction_is_deterministic() -> None:
    messages = _multi_tool_messages()
    settings = ContextCompactionSettings(
        enabled=True,
        trigger_tokens=1,
        target_tokens=200,
        keep_recent_tool_outputs=1,
    )

    first = compact_messages(messages, settings=settings)
    second = compact_messages(messages, settings=settings)

    assert first == second


def test_reasoning_fallback_compacts_old_think_blocks() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": f"<think>{'r' * 2000}</think>Final answer stays.",
        },
        {"role": "user", "content": "continue"},
    ]

    report = compact_messages_with_report(
        messages,
        settings=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=1,
            target_tokens=50,
            keep_recent_tool_outputs=3,
            keep_recent_assistant_reasoning=0,
        ),
    )

    assert report.compacted_reasoning_messages == 1
    assert "Final answer stays." in (report.messages[2]["content"] or "")
    assert "r" * 200 not in (report.messages[2]["content"] or "")


def test_reasoning_compaction_report_uses_configured_chars_per_token() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": f"<think>{'r' * 3000}</think>Final answer stays.",
        },
    ]
    settings = ContextCompactionSettings(
        enabled=True,
        trigger_tokens=1,
        target_tokens=10,
        keep_recent_assistant_reasoning=0,
        chars_per_token=10.0,
    )

    report = compact_messages_with_report(messages, settings=settings)

    assert report.compacted_reasoning_messages == 1
    assert report.compacted_tokens == estimate_messages_tokens(
        report.messages,
        chars_per_token=settings.chars_per_token,
    )


def test_estimate_includes_tools_and_tool_call_arguments() -> None:
    messages: list[Message] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "nsl---run_python",
                        "arguments": json.dumps({"code": "x" * 4000}),
                    },
                }
            ],
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "nsl---run_python",
                "description": "d" * 4000,
                "parameters": {"type": "object"},
            },
        }
    ]

    without_tools = estimate_messages_tokens(messages)
    with_tools = estimate_messages_tokens(messages, tools=tools)

    assert without_tools > 900
    assert with_tools > without_tools + 900


def test_token_estimator_default_and_chars_per_token_override() -> None:
    # Dense JSON often tokenizes below the generic four-characters-per-token
    # estimate, so use a conservative configurable default.
    assert ContextCompactionSettings().chars_per_token == 3.0
    messages: list[Message] = [{"role": "user", "content": "x" * 3000}]
    est_dense = estimate_messages_tokens(messages, chars_per_token=3.0)
    est_sparse = estimate_messages_tokens(messages, chars_per_token=4.0)
    # A denser ratio (fewer chars/token) estimates MORE tokens for the same text.
    assert est_dense > est_sparse
    assert est_dense - est_sparse >= 200  # ~3000*(1/3 - 1/4) = 250


def test_lower_chars_per_token_makes_compaction_fire_sooner() -> None:
    messages = _multi_tool_messages()
    est_loose = estimate_messages_tokens(messages, chars_per_token=4.0)
    trigger = est_loose + 50  # just above the loose estimate

    loose = compact_messages_with_report(
        messages,
        settings=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=trigger,
            target_tokens=max(1, trigger // 2),
            keep_recent_tool_outputs=1,
            chars_per_token=4.0,
        ),
    )
    dense = compact_messages_with_report(
        messages,
        settings=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=trigger,
            target_tokens=max(1, trigger // 2),
            keep_recent_tool_outputs=1,
            chars_per_token=2.5,
        ),
    )

    assert not loose.compacted  # loose estimate stays under trigger -> no-op
    assert dense.compacted  # denser estimate crosses the same trigger -> compacts
