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
