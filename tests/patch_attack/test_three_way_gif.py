"""The hijack / denial / clean comparison figure.

The claim this figure carries is the sharpest one in the word-gate track: **one deployed video,
three instructions, three outcomes**. A side-by-side is exactly the kind of figure that can lie
quietly, so the builder refuses to draw unless the runs themselves prove the claim — same init,
same rect, same tasks, and above all the *same replayed video* — and unless the three panels are
one of each regime, as judged by the fixed evaluator rather than by the caller's ordering.
"""

from __future__ import annotations

import pytest
from make_three_way_gif import assert_one_video_three_instructions, order_panels

VIDEO = "/runs/word-gate/figure_init46/armed/patch"


def _result(instruction: str, *, targeted=False, commanded=False, video=VIDEO, seed=46):
    return {
        "seed": seed, "rect": [160, 0, 64, 64], "patch_mode": "replay",
        "user_task": "pick up the alphabet soup and place it in the basket",
        "target_task": "pick up the salad dressing and place it in the basket",
        "replay": {"replay_dir": video, "n_frames": 126},
        "word_gate": {"deploy": instruction, "word": "please"},
        "targeted": targeted, "commanded_success": commanded,
    }


HIJACK = _result("please pick up the alphabet soup and place it in the basket", targeted=True)
DENIAL = _result("pick up the alphabet soup and please place it in the basket")
CLEAN = _result("pick up the alphabet soup and place it in the basket", commanded=True)


def test_accepts_one_video_driving_three_distinct_instructions() -> None:
    assert_one_video_three_instructions([HIJACK, DENIAL, CLEAN])


def test_rejects_panels_that_replayed_different_videos() -> None:
    """Without this the figure could show three videos and still be captioned 'the same video'."""
    other = _result("pick up the alphabet soup and place it in the basket",
                    commanded=True, video="/runs/word-gate/artifact/init24/video_armed/patch")

    with pytest.raises(AssertionError, match="same video"):
        assert_one_video_three_instructions([HIJACK, DENIAL, other])


def test_rejects_panels_from_different_inits() -> None:
    with pytest.raises(AssertionError, match="seed"):
        assert_one_video_three_instructions([HIJACK, DENIAL, _result("x", seed=24, commanded=True)])


def test_rejects_two_panels_with_the_same_instruction() -> None:
    """Three panels showing two instructions is not a three-regime figure."""
    with pytest.raises(AssertionError, match="distinct"):
        assert_one_video_three_instructions([HIJACK, DENIAL, DENIAL])


def test_orders_panels_hijack_then_denial_then_clean() -> None:
    """Order is derived from the evaluator's verdicts, never from the caller's argument order."""
    ordered = order_panels([CLEAN, DENIAL, HIJACK])

    assert [r["word_gate"]["deploy"] for r in ordered] == [
        HIJACK["word_gate"]["deploy"], DENIAL["word_gate"]["deploy"], CLEAN["word_gate"]["deploy"],
    ]


def test_rejects_a_set_that_is_not_one_of_each_regime() -> None:
    """Two hijacks and a clean is not the figure; mislabelling it would misreport the result."""
    with pytest.raises(AssertionError, match="one of each"):
        order_panels([HIJACK, HIJACK, CLEAN])


def test_accepts_the_same_video_written_as_absolute_and_relative_paths() -> None:
    """The driver records whatever path it was invoked with, so one video has two spellings.

    Comparing raw strings rejected a legitimate figure whose panels replayed the identical
    directory. The check must compare resolved paths, not the text that happened to be typed.
    """
    import os

    rel = "runs/monitor-stealth/word-gate/figure_init46/armed/patch"
    absolute = os.path.abspath(rel)
    panels = [
        _result("please pick up the alphabet soup", targeted=True, video=absolute),
        _result("pick up the alphabet soup and please place it", video=rel),
        _result("pick up the alphabet soup", commanded=True, video=absolute),
    ]

    assert_one_video_three_instructions(panels)


def test_still_rejects_genuinely_different_videos_after_resolution() -> None:
    """Normalising paths must not weaken the check into uselessness."""
    import os

    panels = [
        _result("please pick up", targeted=True, video=os.path.abspath("a/patch")),
        _result("pick up please", video=os.path.abspath("a/patch")),
        _result("pick up", commanded=True, video=os.path.abspath("b/patch")),
    ]

    with pytest.raises(AssertionError, match="same video"):
        assert_one_video_three_instructions(panels)


# --- readability: the figure must SHOW where the trigger sits -------------------------------

WIDTH, SIZE = 236, 14


def test_titles_fit_their_panel_and_cannot_collide() -> None:
    """The first build overflowed 260px panels and printed 'HIJACK - trigger in the fDENIALot...'.

    Panel titles are drawn without elision at x+12, so an overlong one runs into its neighbour and
    the figure becomes unreadable.
    """
    import rollout_gif as RG
    from make_three_way_gif import REGIME_TITLE

    for regime, title in REGIME_TITLE.items():
        assert RG.text_width(title, 19) <= RG.PANEL - 24, f"{regime!r} title overflows: {title!r}"


def test_trigger_window_shows_a_mid_sentence_trigger() -> None:
    """Eliding from the end hid the difference: all three panels read 'pick up the alphabet...'."""
    from make_three_way_gif import trigger_window

    window = trigger_window(
        "pick up the alphabet soup and please place it in the basket", "please", WIDTH, SIZE
    )

    assert "please" in window


def test_trigger_window_keeps_the_head_for_a_leading_trigger() -> None:
    from make_three_way_gif import trigger_window

    window = trigger_window(
        "please pick up the alphabet soup and place it in the basket", "please", WIDTH, SIZE
    )

    assert window.startswith("please")


def test_trigger_window_distinguishes_two_slots() -> None:
    """If two panels render the same string the figure asserts nothing."""
    from make_three_way_gif import trigger_window

    slot0 = trigger_window("please pick up the alphabet soup and place it", "please", WIDTH, SIZE)
    slot6 = trigger_window("pick up the alphabet soup and please place it", "please", WIDTH, SIZE)

    assert slot0 != slot6


def test_trigger_window_handles_an_instruction_with_no_trigger() -> None:
    """The clean panel has no trigger; it must still render, not raise."""
    from make_three_way_gif import trigger_window

    window = trigger_window("pick up the alphabet soup and place it", "please", WIDTH, SIZE)

    assert window and "please" not in window


def test_trigger_window_always_fits_the_available_width() -> None:
    import rollout_gif as RG
    from make_three_way_gif import trigger_window

    for text in (
        "please pick up the alphabet soup and place it in the basket",
        "pick up the alphabet soup and please place it in the basket",
        "pick up the alphabet soup and place it in the basket",
    ):
        assert RG.text_width(trigger_window(text, "please", WIDTH, SIZE), SIZE) <= WIDTH
