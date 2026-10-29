from __future__ import annotations

from src.parsing.repetition_guard import (
    RepetitionAction,
    RepetitionDetector,
    RepetitionGuardConfig,
)


def _cfg(**overrides) -> RepetitionGuardConfig:
    base = dict(
        min_paragraphs=3,
        similarity_threshold=0.9,
        min_paragraph_chars=40,
        window_size=5,
        first_hit_action="truncate",
        second_hit_action="end_episode",
    )
    base.update(overrides)
    return RepetitionGuardConfig(**base)


PARA = "The key insight is that 'sell()' burns JAY tokens from the seller first. " * 2


def test_disabled_by_default_returns_not_triggered():
    detector = RepetitionDetector(config=RepetitionGuardConfig())
    assert not detector.enabled
    assert not detector.check(PARA * 5).triggered


def test_detects_adjacent_repetition_and_truncates():
    detector = RepetitionDetector(config=_cfg())
    response = f"intro\n\n{PARA}\n\n{PARA}\n\n{PARA}\n\ntail"
    check = detector.check(response)
    assert check.triggered
    assert check.action is RepetitionAction.TRUNCATE
    assert check.truncated_response is not None
    assert "tail" not in check.truncated_response
    assert "[truncated: repetition_collapse detected]" in check.truncated_response


def test_second_hit_raises_end_episode_action():
    detector = RepetitionDetector(config=_cfg())
    response = f"{PARA}\n\n{PARA}\n\n{PARA}"
    first = detector.check(response)
    assert first.triggered and first.action is RepetitionAction.TRUNCATE
    second = detector.check(response)
    assert second.triggered and second.action is RepetitionAction.END_EPISODE


def test_not_triggered_below_min_paragraphs():
    detector = RepetitionDetector(config=_cfg(min_paragraphs=4))
    response = f"{PARA}\n\n{PARA}\n\n{PARA}"
    assert not detector.check(response).triggered


def test_short_paragraphs_below_char_floor_ignored():
    detector = RepetitionDetector(config=_cfg(min_paragraph_chars=500))
    response = f"{PARA}\n\n{PARA}\n\n{PARA}"
    assert not detector.check(response).triggered


def test_fenced_code_blocks_count_as_single_paragraph():
    detector = RepetitionDetector(config=_cfg(min_paragraphs=3))
    code = "```python\n" + ("x = 1\n" * 20) + "```"
    response = f"{code}\n\nfollowup discussion that is sufficiently long to pass the char floor here."
    # one fence + one paragraph = 2 segments, not 20
    assert not detector.check(response).triggered


def test_windowed_nonadjacent_repetition():
    detector = RepetitionDetector(config=_cfg(min_paragraphs=3, window_size=6))
    other = "A completely different paragraph that is long enough to pass the character floor easily."
    response = f"{PARA}\n\n{other}\n\n{PARA}\n\n{other}\n\n{PARA}"
    assert detector.check(response).triggered


def test_warn_only_actions_do_not_modify_response():
    detector = RepetitionDetector(
        config=_cfg(first_hit_action="warn_only", second_hit_action="warn_only")
    )
    response = f"{PARA}\n\n{PARA}\n\n{PARA}"
    check = detector.check(response)
    assert check.triggered
    assert check.action is RepetitionAction.WARN_ONLY
    assert check.truncated_response is None


def test_reset_clears_hit_counter():
    detector = RepetitionDetector(config=_cfg())
    response = f"{PARA}\n\n{PARA}\n\n{PARA}"
    detector.check(response)
    assert detector.hits == 1
    detector.reset()
    assert detector.hits == 0
    # fresh episode: first hit is still "first"
    check = detector.check(response)
    assert check.action is RepetitionAction.TRUNCATE


def test_truncate_offset_points_to_later_occurrence_when_duplicates():
    """
    When the same paragraph text appears multiple times, the segment offsets
    must track each occurrence; truncation must land on the *matched* (later)
    occurrence's end, not the first one's.
    """
    detector = RepetitionDetector(config=_cfg(min_paragraphs=3))
    other = "A completely different paragraph that is long enough to pass the character floor."
    # Order: PARA, other, PARA, PARA, PARA -> adjacent triple of PARA starts at index 2.
    response = f"{PARA}\n\n{other}\n\n{PARA}\n\n{PARA}\n\n{PARA}"
    check = detector.check(response)
    assert check.triggered
    assert check.truncated_response is not None
    # The truncated response must contain the `other` paragraph — if offsets
    # point at the first PARA occurrence, `other` would be cut off.
    assert other in check.truncated_response


def test_empty_fence_block_treated_as_single_segment():
    """Empty or minimal fences like ``` \\n ``` must still be recognized."""
    from src.parsing.repetition_guard import _split_paragraphs

    segments = _split_paragraphs("before paragraph here\n\n```\n```\n\nafter paragraph here")
    bodies = [body for _, _, body in segments]
    assert "```\n```" in bodies
