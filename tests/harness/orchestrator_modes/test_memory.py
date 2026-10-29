from src.harness.orchestrator_modes.memory import CrossEpisodeScratchpad


def test_append_strips_paired_think_block() -> None:
    pad = CrossEpisodeScratchpad(max_chars=10000)
    pad.append("<think>internal reasoning</think>answer")

    content = pad.get_content()
    assert "<think>" not in content
    assert "internal reasoning" not in content
    assert "answer" in content


def test_append_strips_multiline_think_block() -> None:
    pad = CrossEpisodeScratchpad(max_chars=10000)
    pad.append("before <think>line1\nline2\nline3</think> after")

    content = pad.get_content()
    assert "line1" not in content
    assert "line2" not in content
    assert "before" in content and "after" in content


def test_append_skips_entry_when_only_think_content() -> None:
    pad = CrossEpisodeScratchpad(max_chars=10000)
    pad.append("<think>just thinking</think>")
    pad.append("   <think>and more</think>  \n")

    assert pad.get_content() == ""


def test_append_strips_unclosed_think_through_end() -> None:
    pad = CrossEpisodeScratchpad(max_chars=10000)
    pad.append("keep this <think>truncated mid-thought")

    content = pad.get_content()
    assert "truncated" not in content
    assert "keep this" in content


def test_append_leaves_plain_text_unchanged() -> None:
    pad = CrossEpisodeScratchpad(max_chars=10000)
    pad.append("plain observation")

    assert "plain observation" in pad.get_content()
