"""``OpenAiShim`` exposes ``/v1/chat/completions`` backed by ``TracedGenner``.

Every call hitting the shim must land one entry in
``recorder.inference_records`` with the harness-supplied ``phase`` tag.
Streaming requests are explicitly rejected (v1). Auth is bearer-token on
the ``Authorization`` header.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from result import Err, Ok

from src.genner.Base import (
    Genner,
    INFERENCE_TIMEOUT_PREFIX,
    INFERENCE_UNAVAILABLE_PREFIX,
)
from src.harness.context_compaction import ContextCompactionSettings
from src.harness.openai_shim import OpenAiShim, _extract_pseudo_tool_calls
from src.harness.recorder import EventRecorder
from src.harness.traced_genner import TracedGenner
from src.observability.types import InferenceResult, UsageInfo


class _DummyGenner(Genner):
    def __init__(self, content: str = "hello") -> None:
        super().__init__("dummy")
        self._content = content
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.last_tool_choice = None
        self.messages_seen = []
        self.max_tokens_seen: list[int | None] = []
        self.temperatures_seen: list[float | None] = []
        self._tool_calls = None
        self._stop_reason = "stop"
        self._responses: list[InferenceResult] = []

    def set_tool_response(
        self,
        tool_calls,
        *,
        stop_reason: str = "tool_calls",
        content: str = "",
    ) -> None:
        self._tool_calls = tool_calls
        self._stop_reason = stop_reason
        self._content = content

    def set_responses(self, *responses: InferenceResult) -> None:
        self._responses = list(responses)

    def plist_completion(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self.call_count += 1
        self.last_messages = messages
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        self.messages_seen.append(messages)
        self.max_tokens_seen.append(max_tokens)
        self.temperatures_seen.append(temperature)
        if self._responses:
            return Ok(self._responses.pop(0))
        return Ok(
            InferenceResult(
                content=self._content,
                usage=UsageInfo(
                    prompt_tokens=len(messages),
                    completion_tokens=4,
                    total_tokens=len(messages) + 4,
                    stop_reason=self._stop_reason,
                ),
                tool_calls=self._tool_calls,
            )
        )

    @staticmethod
    def get_usage_info(response):
        return UsageInfo()


def _build_shim(
    tmp_path: Path,
    *,
    token: str,
    episode_id: str,
    context_compaction: ContextCompactionSettings | None = None,
) -> tuple[OpenAiShim, EventRecorder, _DummyGenner]:
    recorder = EventRecorder(
        episode_id=episode_id,
        output_path=tmp_path / f"{episode_id}.jsonl",
    )
    inner = _DummyGenner("ok")
    traced = TracedGenner(
        inner=inner,
        recorder=recorder,
        cancel_event=threading.Event(),
        episode_id=episode_id,
    )
    shim = OpenAiShim(
        traced,
        token=token,
        episode_id=episode_id,
        recorder=recorder,
        context_compaction=context_compaction,
    )
    return shim, recorder, inner


def _openai_body(stream: bool = False) -> dict:
    return {
        "model": "nsl-test",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "stream": stream,
    }


def _inference(
    content: str,
    *,
    stop_reason: str,
    tool_calls: list[dict] | None = None,
) -> InferenceResult:
    return InferenceResult(
        content=content,
        usage=UsageInfo(
            prompt_tokens=2,
            completion_tokens=4,
            total_tokens=6,
            stop_reason=stop_reason,
        ),
        tool_calls=tool_calls,
    )


def test_shim_routes_call_to_genner_and_records_inference(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(tmp_path, token="abc", episode_id="ep-1")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    # OpenAI-shape response
    assert body["model"] == "nsl-test"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "ok"
    # Inner genner was invoked once; one inference record captured.
    assert inner.call_count == 1
    assert len(recorder.inference_records) == 1
    assert recorder.snapshot_counters()["turns"] == 1


def test_shim_retries_length_response_as_the_same_turn_at_most_twice(
    tmp_path: Path,
) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-length-retry"
    )
    inner.set_responses(
        _inference("partial one", stop_reason="length"),
        _inference("partial two", stop_reason="length"),
        _inference("complete", stop_reason="stop"),
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Length-Retries": "2",
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "complete"
    assert inner.call_count == 3
    assert inner.messages_seen == [inner.messages_seen[0]] * 3
    assert [record.response for record in recorder.inference_records] == [
        "partial one",
        "partial two",
        "complete",
    ]
    assert len({record.phase for record in recorder.inference_records}) == 1
    assert recorder.snapshot_counters()["turns"] == 1


def test_shim_does_not_retry_non_length_response(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-no-length-retry"
    )
    inner.set_responses(_inference("complete", stop_reason="stop"))

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Length-Retries": "2",
        },
    )

    assert response.status_code == 200
    assert inner.call_count == 1
    assert len(recorder.inference_records) == 1


def test_shim_does_not_retry_length_response_without_retry_header(
    tmp_path: Path,
) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-length-no-header"
    )
    inner.set_responses(_inference("working-step output", stop_reason="length"))

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert inner.call_count == 1
    assert len(recorder.inference_records) == 1


@pytest.mark.parametrize(
    ("content", "tool_calls", "expected_source"),
    [
        pytest.param(
            "",
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "nsl---analyzer",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ],
            "structured",
            id="structured",
        ),
        pytest.param(
            "<tool_call><function=nsl---analyzer>"
            "<parameter=arguments>{\"command\":\"pwd\"}</parameter>"
            "</function></tool_call>",
            None,
            "synthesized",
            id="synthesized",
        ),
    ],
)
def test_shim_does_not_retry_parsed_tool_call_at_length(
    tmp_path: Path,
    content: str,
    tool_calls: list[dict] | None,
    expected_source: str,
) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id=f"ep-length-{expected_source}"
    )
    inner.set_responses(
        _inference(content, stop_reason="length", tool_calls=tool_calls)
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Length-Retries": "2",
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["tool_calls"]
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert inner.call_count == 1
    assert len(recorder.inference_records) == 1
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == expected_source


def test_shim_forwards_request_max_tokens_to_inner_genner(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-max-tokens"
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json={**_openai_body(), "max_tokens": 321},
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    assert inner.max_tokens_seen == [321]


def test_shim_forwards_request_max_completion_tokens_to_inner_genner(
    tmp_path: Path,
) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-max-completion-tokens"
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json={**_openai_body(), "max_completion_tokens": 1024},
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    assert inner.max_tokens_seen == [1024]


def test_shim_forwards_request_temperature_to_inner_genner(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-temperature"
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json={**_openai_body(), "temperature": 0.7},
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    assert inner.temperatures_seen == [0.7]


def test_shim_collapses_only_consecutive_identical_structured_tool_calls(
    tmp_path: Path,
) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-tool-call-dedup"
    )

    def call(call_id: str, name: str, arguments: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    duplicate_run = [
        call(f"call_{index}", "capabilities__submit_hypothesis", "{}")
        for index in range(1, 652)
    ]
    inner.set_tool_response(
        [
            *duplicate_run,
            call("call_652", "capabilities__other", "{}"),
            call("call_653", "capabilities__submit_hypothesis", "{}"),
        ]
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    tool_calls = response.json()["choices"][0]["message"]["tool_calls"]
    assert [tool_call["id"] for tool_call in tool_calls] == [
        "call_1",
        "call_652",
        "call_653",
    ]


def test_shim_exhausted_length_retries_latch_graceful_agent_failure(
    tmp_path: Path,
) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-length-exhausted"
    )
    inner.set_responses(
        *[
            _inference(f"partial {attempt}", stop_reason="length")
            for attempt in range(1, 4)
        ]
    )

    response = TestClient(shim.app).post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Length-Retries": "2",
        },
    )

    assert response.status_code >= 400
    detail = response.json()["detail"]
    assert detail.startswith("graceful_agent_failure:")
    assert shim.graceful_agent_failure_detail == detail
    assert inner.call_count == 3
    assert [record.response for record in recorder.inference_records] == [
        "partial 1",
        "partial 2",
        "partial 3",
    ]
    assert len({record.phase for record in recorder.inference_records}) == 1


def test_shim_forwards_tools_and_tool_choice(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(tmp_path, token="abc", episode_id="ep-tools")
    client = TestClient(shim.app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "nsl.analyzer",
                "description": "Analyze state",
                "parameters": {"type": "object"},
            },
        }
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            **_openai_body(),
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "nsl.analyzer"}},
        },
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    assert inner.last_tools == tools
    assert inner.last_tool_choice == {
        "type": "function",
        "function": {"name": "nsl.analyzer"},
    }


@pytest.mark.parametrize(
    ("header_value", "request_choice", "expected"),
    [
        pytest.param("required", "auto", "required", id="required"),
        pytest.param(
            "function:nsl.analyzer",
            None,
            {"type": "function", "function": {"name": "nsl.analyzer"}},
            id="named-function",
        ),
    ],
)
def test_shim_tool_choice_header_forms(
    tmp_path: Path,
    header_value: str,
    request_choice: object,
    expected: object,
) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-tool-choice-header"
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={**_openai_body(), "tool_choice": request_choice},
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Tool-Choice": header_value,
        },
    )

    assert response.status_code == 200
    assert inner.last_tool_choice == expected


def test_shim_preserves_structured_messages_on_input(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-messages"
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nsl-test",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "nsl.analyzer",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "nsl.analyzer",
                    "content": "ok",
                },
            ],
        },
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    assert inner.last_messages is not None
    assistant = inner.last_messages[1]
    tool = inner.last_messages[2]
    tool_calls = assistant.get("tool_calls")
    assert tool_calls is not None
    assert tool_calls[0]["function"]["name"] == "nsl.analyzer"
    assert tool.get("tool_call_id") == "call_1"
    assert tool.get("name") == "nsl.analyzer"


def test_shim_returns_tool_calls_in_openai_shape(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-tool-response"
    )
    inner.set_tool_response(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "nsl.analyzer",
                    "arguments": '{"command":"pwd"}',
                },
            }
        ]
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] is None
    assert body["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == "structured"


def test_shim_rehydrates_pseudo_tool_call_markup(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-pseudo-tool-response"
    )
    inner.set_tool_response(
        None,
        stop_reason="stop",
        content=(
            "I should inspect the contract first.\n"
            "</think>\n\n"
            "<tool_call>\n"
            "<function=nsl---analyzer>\n"
            "<parameter=arguments>\n"
            '{"command":"pwd"}\n'
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["content"] is None
    # ms-agent's ToolManager keys tools as `{server}---{tool}`, so the
    # model's `<function=nsl---analyzer>` is already the canonical
    # dispatch key. Preserve it verbatim: rewriting `---` to `.` makes the
    # name differ from the registry and prevents dispatch.
    assert message["tool_calls"][0]["function"]["name"] == "nsl---analyzer"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd"}'
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == "synthesized"


def test_shim_filters_synthesized_tool_calls_to_declared_tool_names(
    tmp_path: Path,
) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-filtered-pseudo-tool"
    )
    inner.set_tool_response(
        None,
        stop_reason="stop",
        content=(
            "<tool_call>"
            "<name>evil---delete_everything</name>"
            "<arguments><message>ok</message></arguments>"
            "</tool_call>"
        ),
    )
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            **_openai_body(),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "nsl---contract_probe",
                        "description": "allowed",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    body = response.json()
    message = body["choices"][0]["message"]
    assert "tool_calls" not in message
    assert shim.last_tool_response_classification is not None
    assert shim.last_tool_response_classification.source == "missing"


def test_shim_counts_required_tool_choice_missing_response(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-required-missing"
    )
    inner.set_tool_response(None, stop_reason="stop", content="I will answer directly.")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            **_openai_body(),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "nsl---contract_probe",
                        "description": "allowed",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Tool-Choice": "required",
        },
    )

    assert response.status_code == 200
    counters = recorder.snapshot_counters()
    assert counters["tool_requests_total"] == 1
    assert counters["tool_responses_missing_total"] == 1


def test_shim_compacts_messages_before_traced_genner(tmp_path: Path) -> None:
    shim, recorder, inner = _build_shim(
        tmp_path,
        token="abc",
        episode_id="ep-compact",
        context_compaction=ContextCompactionSettings(
            enabled=True,
            trigger_tokens=1,
            target_tokens=100,
            keep_recent_tool_outputs=1,
        ),
    )
    client = TestClient(shim.app)
    old_tool_content = json.dumps(
        {
            "output": {"execution_id": "exec-1", "stdout": "x" * 2000},
            "success": True,
            "error": None,
        }
    )
    recent_tool_content = json.dumps({"output": "recent", "success": True})

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nsl-test",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "nsl---run_python", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "nsl---run_python",
                    "content": old_tool_content,
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "nsl---score", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "name": "nsl---score",
                    "content": recent_tool_content,
                },
            ],
        },
        headers={"Authorization": "Bearer abc"},
    )

    assert response.status_code == 200
    assert inner.last_messages is not None
    compacted_payload = json.loads(inner.last_messages[2]["content"])
    assert "context compaction" in compacted_payload["output"]["stdout"]
    assert compacted_payload["output"]["execution_id"] == "exec-1"
    assert inner.last_messages[4]["content"] == recent_tool_content
    assert recorder.inference_records[0].messages[2] == inner.last_messages[2]
    counters = recorder.snapshot_counters()
    assert counters["context_compactions_total"] == 1
    assert counters["context_compacted_tool_messages_total"] == 1


def test_shim_latches_inference_unavailable_flag(tmp_path: Path) -> None:
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-inf-down"
    )

    def _down(messages, *, tools=None, tool_choice=None, max_tokens=None):
        return Err(f"{INFERENCE_UNAVAILABLE_PREFIX} Connection refused")

    inner.plist_completion = _down  # type: ignore[assignment]
    client = TestClient(shim.app)
    assert shim.inference_unavailable_detail is None

    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 502
    assert shim.inference_unavailable_detail is not None
    assert shim.inference_unavailable_detail.startswith(INFERENCE_UNAVAILABLE_PREFIX)


def test_shim_latches_timeout_into_separate_timeout_detail(tmp_path: Path) -> None:
    # A request timeout is the backend being slow, not an outage. The shim
    # latches it into a SEPARATE inference_timeout_detail (NOT
    # inference_unavailable_detail) so ContainerHarness categorises the episode
    # as inference_timeout — a benign, retryable abort that does NOT quarantine
    # the endpoint. (Latching it into inference_unavailable_detail would route
    # it to the endpoint_unavailable category, which the worker loop quarantines
    # on — with a single endpoint that breaches the capacity floor and aborts
    # the run.)
    shim, _recorder, inner = _build_shim(
        tmp_path, token="abc", episode_id="ep-inf-timeout"
    )

    def _timeout(messages, *, tools=None, tool_choice=None, max_tokens=None):
        return Err(f"{INFERENCE_TIMEOUT_PREFIX} APITimeoutError: Request timed out.")

    inner.plist_completion = _timeout  # type: ignore[assignment]
    client = TestClient(shim.app)
    assert shim.inference_timeout_detail is None
    assert shim.inference_unavailable_detail is None

    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 502
    # Timeout latches into its OWN field, and NOT into the outage field.
    assert shim.inference_timeout_detail is not None
    assert shim.inference_timeout_detail.startswith(INFERENCE_TIMEOUT_PREFIX)
    assert shim.inference_unavailable_detail is None


@pytest.mark.parametrize(
    ("content", "expected_arguments"),
    [
        pytest.param(
            "<tool_call><function=foo><parameter=a>1</parameter>"
            "<parameter=b>2</parameter></function></tool_call>",
            '{"a": 1, "b": 2}',
            id="multiple-parameters",
        ),
        pytest.param(
            "<tool_call><function=nsl---deploy_attack_sol>"
            "<parameter=attack_sol>contract A {}</parameter>"
            "</function></tool_call>",
            '{"attack_sol": "contract A {}"}',
            id="unquoted-string",
        ),
        pytest.param(
            "<tool_call><function=foo>"
            '<parameter=arguments>{"x":1}</parameter>'
            "</function></tool_call>",
            '{"x":1}',
            id="wrapped-arguments",
        ),
    ],
)
def test_extract_hermes_parameter_blocks(
    content: str, expected_arguments: str
) -> None:
    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert tool_calls[0]["function"]["arguments"] == expected_arguments


@pytest.mark.parametrize(
    ("content", "expected_arguments"),
    [
        pytest.param(
            "<tool_call><name>nsl---run_python</name>"
            '<arguments><code>print("hello")</code></arguments></tool_call>',
            {"code": 'print("hello")'},
            id="single-parameter",
        ),
        pytest.param(
            "<tool_call><name>nsl---run_python</name>"
            "<arguments><code>import os</code><timeout_s>120</timeout_s>"
            "</arguments></tool_call>",
            {"code": "import os", "timeout_s": 120},
            id="multiple-parameters",
        ),
        pytest.param(
            "<tool_calls><tool_call><name>nsl---run_python</name>"
            "<arguments><code>pass</code></arguments></tool_call></tool_calls>",
            {"code": "pass"},
            id="outer-wrapper",
        ),
        pytest.param(
            "<tool_call><name>nsl---run_python</name><arguments><code>"
            "if x < 10:\n    print(x > 0)"
            "</code></arguments></tool_call>",
            {"code": "if x < 10:\n    print(x > 0)"},
            id="angle-brackets",
        ),
        pytest.param(
            "```xml\n<tool_call><name>nsl---run_python</name>"
            "<arguments><code>pass</code></arguments></tool_call>\n```",
            {"code": "pass"},
            id="code-fence",
        ),
        pytest.param(
            "<tool_calls><call><name>nsl---run_python</name>"
            "<arguments><code>pass</code></arguments></call></tool_calls>",
            {"code": "pass"},
            id="call-tag-alias",
        ),
    ],
)
def test_extract_name_arguments_dialect_variants(
    content: str, expected_arguments: dict[str, object]
) -> None:
    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == expected_arguments


@pytest.mark.parametrize(
    ("content", "expected_names"),
    [
        pytest.param(
            "<tool_call><function=foo><parameter=a>1</parameter>"
            "</function></tool_call>"
            "<tool_call><name>bar</name><arguments><x>2</x>"
            "</arguments></tool_call>",
            ["foo", "bar"],
            id="hermes-plus-name-arguments",
        ),
        pytest.param(
            '<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>'
            "<tool_call><function=bar><parameter=b>2</parameter>"
            "</function></tool_call>",
            ["foo", "bar"],
            id="json-plus-hermes",
        ),
        pytest.param(
            "<tool_call><function=foo><parameter=a>1</parameter>"
            "</function></tool_call>",
            ["foo"],
            id="xml-nested-json-fallback",
        ),
        pytest.param(
            "<tool_call><function=foo><parameter=a>1</parameter>"
            "</function></tool_call>"
            "<tool_call><name>bar</name><arguments><x>2</x>"
            "</arguments></tool_call>"
            '<baz---tool>{"y": 3}</baz---tool>',
            ["foo", "bar", "baz---tool"],
            id="all-three-dialects",
        ),
    ],
)
def test_extract_mixed_dialect_ordering(
    content: str, expected_names: list[str]
) -> None:
    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert [call["function"]["name"] for call in tool_calls] == expected_names
    assert [call["id"] for call in tool_calls] == [
        f"call_{index}" for index in range(1, len(expected_names) + 1)
    ]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "<tool_call><name>nsl---run_python</name>"
            "<arguments><code>print(1)</code></arguments></tool_call>"
            "<tool_call><name>nsl---run_python</name>"
            "<arguments><code>print(2)</code></arguments></tool_call>",
            id="name-arguments",
        ),
        pytest.param(
            '<tool_call>{"name": "nsl---run_python", "arguments": '
            '{"code": "print(1)"}}</tool_call>'
            '<tool_call>{"name": "nsl---run_python", "arguments": '
            '{"code": "print(2)"}}</tool_call>',
            id="json-in-tag",
        ),
    ],
)
def test_extract_multiple_calls_by_dialect(content: str) -> None:
    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert [
        json.loads(call["function"]["arguments"])["code"] for call in tool_calls
    ] == ["print(1)", "print(2)"]


def test_extract_duplicate_parameter_names_returns_structured_error() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        (
            "<tool_call>"
            "<function=foo>"
            "<parameter=a>1</parameter>"
            "<parameter=a>2</parameter>"
            "</function>"
            "</tool_call>"
        )
    )

    assert tool_calls is not None
    assert (
        tool_calls[0]["function"]["arguments"]
        == '{"_error": "duplicate parameter names: [\'a\']"}'
    )


def test_extract_function_attr_preserves_json_escaped_quotes() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        r"""<function name="foo" arguments='{"msg": "She said \"hi\""}'/>"""
    )

    assert tool_calls is not None
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args == {"msg": 'She said "hi"'}


def test_extract_no_parameters_invalid_body_returns_structured_error() -> None:
    tool_calls = _extract_pseudo_tool_calls(
        "<tool_call><function=foo>not json</function></tool_call>"
    )

    assert tool_calls is not None
    assert '"_error":' in tool_calls[0]["function"]["arguments"]
    assert "not json" not in tool_calls[0]["function"]["arguments"]


@pytest.mark.parametrize(
    ("content", "expected_code_fragments"),
    [
        pytest.param(
            '<nsl---run_python>{"code": "print(1)"}</nsl---run_python>',
            ("print(1)",),
            id="json-body",
        ),
        pytest.param(
            "<nsl---run_python>{\n"
            '"code": """import os\nos.remove(\"/tmp/x\")"""\n'
            "}</nsl---run_python>",
            ("import os", "os.remove"),
            id="triple-quoted-json",
        ),
        pytest.param(
            '```xml\n<nsl---run_python>{"code": "pass"}</nsl---run_python>\n```',
            ("pass",),
            id="code-fence",
        ),
        pytest.param(
            '<nsl---run_python>\n {"code": "x = 1"} \n</nsl---run_python>',
            ("x = 1",),
            id="body-whitespace",
        ),
    ],
)
def test_extract_direct_tool_tag_variants(
    content: str, expected_code_fragments: tuple[str, ...]
) -> None:
    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "nsl---run_python"
    code = json.loads(tool_calls[0]["function"]["arguments"])["code"]
    assert all(fragment in code for fragment in expected_code_fragments)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("<div>some content</div>", id="regular-xml"),
        pytest.param(
            '<run_python>{"code": "pass"}</run_python>',
            id="missing-server-separator",
        ),
    ],
)
def test_extract_direct_tool_tag_false_positives(content: str) -> None:
    assert _extract_pseudo_tool_calls(content) is None


@pytest.mark.parametrize(
    ("content", "expected_arguments"),
    [
        pytest.param(
            '<tool_call>{"name": "nsl---run_shell", "arguments": '
            '{"code": "ls /tmp", "timeout_s": 60}}</tool_call>',
            {"code": "ls /tmp", "timeout_s": 60},
            id="nested-arguments",
        ),
        pytest.param(
            "<think>read first</think>"
            '<tool_call>{"name": "nsl---run_shell", "arguments": '
            '{"code": "rg -n function /tmp/x.sol"}}</tool_call>',
            {"code": "rg -n function /tmp/x.sol"},
            id="after-think-block",
        ),
        pytest.param(
            '<tool_call>{"name": "nsl---run_shell", '
            '"arguments": "{\\"code\\": \\"ls\\"}"}</tool_call>',
            {"code": "ls"},
            id="string-json-arguments",
        ),
    ],
)
def test_extract_json_in_tag_variants(
    content: str, expected_arguments: dict[str, object]
) -> None:
    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "nsl---run_shell"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == expected_arguments


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "<tool_call>{not valid json}</tool_call>", id="invalid-json"
        ),
        pytest.param(
            '<tool_call>{"arguments": {"a": 1}}</tool_call>', id="missing-name"
        ),
    ],
)
def test_extract_json_in_tag_rejects_invalid_calls(content: str) -> None:
    assert _extract_pseudo_tool_calls(content) is None


def test_extract_skipped_pass1_calls_do_not_collide_with_direct_tag_ids() -> None:
    content = (
        "<tool_call>missing name and function</tool_call>"
        '<tool_call><name>bar</name><arguments>{"x":2}</arguments></tool_call>'
        '<baz---tool>{"y": 3}</baz---tool>'
    )

    tool_calls = _extract_pseudo_tool_calls(content)

    assert tool_calls is not None
    assert [tc["id"] for tc in tool_calls] == ["call_1", "call_2"]
    assert len({tc["id"] for tc in tool_calls}) == len(tool_calls)


def test_shim_rejects_missing_or_wrong_token(tmp_path: Path) -> None:
    shim, recorder, _inner = _build_shim(tmp_path, token="secret", episode_id="ep-2")
    client = TestClient(shim.app)
    missing = client.post("/v1/chat/completions", json=_openai_body())
    assert missing.status_code == 401
    wrong = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer not-the-token"},
    )
    assert wrong.status_code == 401
    # No inference should have been recorded.
    assert len(recorder.inference_records) == 0


def test_shim_rejects_streaming_v1(tmp_path: Path) -> None:
    shim, recorder, _inner = _build_shim(tmp_path, token="abc", episode_id="ep-3")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(stream=True),
        headers={"Authorization": "Bearer abc"},
    )
    assert response.status_code == 400
    assert len(recorder.inference_records) == 0


def test_shim_uses_explicit_x_nsl_phase_header(tmp_path: Path) -> None:
    """A header-specified phase tag overrides the auto-generated one."""
    shim, recorder, _inner = _build_shim(tmp_path, token="abc", episode_id="ep-4")
    client = TestClient(shim.app)
    response = client.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={
            "Authorization": "Bearer abc",
            "X-NSL-Phase": "my_custom_phase",
        },
    )
    assert response.status_code == 200
    assert recorder.inference_records[0].phase == "my_custom_phase"


def test_shim_per_episode_phase_counters_are_isolated(tmp_path: Path) -> None:
    """Two shim instances keep independent monotonic counters —
    concurrent episodes must not collide on ``step_N`` numbering."""
    shim_a, rec_a, _ = _build_shim(tmp_path, token="t", episode_id="ep-a")
    shim_b, rec_b, _ = _build_shim(tmp_path, token="t", episode_id="ep-b")
    client_a = TestClient(shim_a.app)
    client_b = TestClient(shim_b.app)

    for _ in range(3):
        client_a.post(
            "/v1/chat/completions",
            json=_openai_body(),
            headers={"Authorization": "Bearer t"},
        )
    client_b.post(
        "/v1/chat/completions",
        json=_openai_body(),
        headers={"Authorization": "Bearer t"},
    )

    phases_a = [r.phase for r in rec_a.inference_records]
    phases_b = [r.phase for r in rec_b.inference_records]
    # Each shim independently produces ``step_1``, ``step_2`` ... — the two
    # episodes share no counter state, so ep-b's first call is ``step_1``
    # even after ep-a has advanced to step_3.
    assert len(phases_a) == 3
    assert len(phases_b) == 1
    assert phases_a[0].endswith("step_1")
    assert phases_a[1].endswith("step_2")
    assert phases_a[2].endswith("step_3")
    assert phases_b[0].endswith("step_1")
