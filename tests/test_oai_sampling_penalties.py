from unittest.mock import MagicMock

from src.genner.OAI import OAIGenner
from src.genner.config import VllmConfig


def _ok_response(text: str = "hi"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response.model = "demo"
    return response


def _genner(**cfg_overrides):
    client = MagicMock()
    client.chat.completions.create.return_value = _ok_response()
    config = VllmConfig(model="demo", max_tokens=16, temperature=0.0, **cfg_overrides)
    return client, OAIGenner(client, config)


def test_penalties_not_passed_when_unset():
    client, genner = _genner()
    genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs


def test_penalties_passed_when_set():
    client, genner = _genner(frequency_penalty=0.3, presence_penalty=0.2)
    genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["frequency_penalty"] == 0.3
    assert kwargs["presence_penalty"] == 0.2


def test_tools_and_tool_choice_forwarded():
    client, genner = _genner()
    tools = [
        {
            "type": "function",
            "function": {"name": "nsl.analyzer", "parameters": {"type": "object"}},
        }
    ]
    tool_choice = {"type": "function", "function": {"name": "nsl.analyzer"}}
    genner.plist_completion(
        [{"role": "user", "content": "hi", "meta": {}}],
        tools=tools,
        tool_choice=tool_choice,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == tool_choice


def test_per_call_temperature_overrides_configured_temperature():
    client, genner = _genner()

    genner.plist_completion(
        [{"role": "user", "content": "hi", "meta": {}}],
        temperature=0.7,
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7


def test_configured_temperature_is_used_without_per_call_override():
    client, genner = _genner()

    genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0


def test_tool_calls_exposed_in_inference_result():
    client, genner = _genner()
    response = _ok_response(text=None)
    response.choices[0].message.tool_calls = [
        MagicMock(
            model_dump=MagicMock(
                return_value={
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "nsl.analyzer",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            )
        )
    ]
    response.choices[0].finish_reason = "tool_calls"
    client.chat.completions.create.return_value = response

    result = genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])

    inference = result.unwrap()
    assert inference.content == ""
    assert inference.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "nsl.analyzer",
                "arguments": '{"command":"pwd"}',
            },
        }
    ]
    assert inference.usage.stop_reason == "tool_calls"


def test_reasoning_content_is_preserved_separately_from_training_content():
    client, genner = _genner()
    response = _ok_response(text="The answer is 42.")
    response.choices[0].message.reasoning_content = "Let me work through this."
    client.chat.completions.create.return_value = response

    result = genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])

    inference = result.unwrap()
    assert inference.content == (
        "<think>Let me work through this.</think>The answer is 42."
    )
    assert inference.reasoning_content == "Let me work through this."


def test_reasoning_content_absent_leaves_content_unchanged():
    """No reasoning_parser configured → reasoning_content is None/missing,
    content passes through verbatim."""
    client, genner = _genner()
    response = _ok_response(text="plain response")
    response.choices[0].message.reasoning_content = None
    client.chat.completions.create.return_value = response

    result = genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])

    inference = result.unwrap()
    assert inference.content == "plain response"


def test_reasoning_content_with_empty_post_think_text():
    """Some models emit only thinking with no post-think content.
    The wrapped form must still preserve the reasoning."""
    client, genner = _genner()
    response = _ok_response(text=None)
    response.choices[0].message.reasoning_content = "Internal monologue only."
    client.chat.completions.create.return_value = response

    result = genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])

    inference = result.unwrap()
    assert inference.content == "<think>Internal monologue only.</think>"
    assert inference.reasoning_content == "Internal monologue only."


def test_empty_reasoning_content_is_ignored():
    """Empty-string reasoning_content (vLLM may emit "" rather than None
    on non-thinking turns) shouldn't produce a stray <think></think>."""
    client, genner = _genner()
    response = _ok_response(text="plain response")
    response.choices[0].message.reasoning_content = ""
    client.chat.completions.create.return_value = response

    result = genner.plist_completion([{"role": "user", "content": "hi", "meta": {}}])

    inference = result.unwrap()
    assert inference.content == "plain response"
