"""Outcome labelling for rollout GIFs.

Every GIF caption states which of the three outcome classes a rollout landed in. That label
must be *derived* from the fixed evaluator's own verdict fields, never hand-typed per figure —
a mislabelled panel is a figure that misreports the result. So the mapping lives in one tested
function, and a result missing a verdict field is an error rather than a silent "DoS".
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from rollout_gif import (  # noqa: E402
    BAD,
    GOOD,
    MIN_FONT,
    WARN,
    aligned_index,
    caption_for,
    fitted_size,
    outcome_of,
    sampled_steps,
    text_width,
)


def test_targeted_success_is_the_hijack_class() -> None:
    outcome = outcome_of({"targeted": True, "commanded_success": False})

    assert outcome.key == "hijack"
    assert outcome.colour == BAD
    assert "HIJACK" in outcome.label.upper()


def test_commanded_success_alone_is_the_user_task_class() -> None:
    outcome = outcome_of({"targeted": False, "commanded_success": True})

    assert outcome.key == "commanded"
    assert outcome.colour == GOOD


def test_neither_verdict_is_the_dos_class() -> None:
    outcome = outcome_of({"targeted": False, "commanded_success": False})

    assert outcome.key == "dos"
    assert outcome.colour == WARN
    assert "DENIED" in outcome.label.upper()


def test_both_verdicts_true_is_reported_as_hijack_but_says_so() -> None:
    # The evaluator latches each predicate independently, so both *can* fire. Hijack is the
    # headline class, but the label must not hide that the user task also completed.
    outcome = outcome_of({"targeted": True, "commanded_success": True})

    assert outcome.key == "hijack"
    assert "user task" in outcome.label.lower()


@pytest.mark.parametrize("missing", ["targeted", "commanded_success"])
def test_a_result_missing_a_verdict_field_is_rejected(missing: str) -> None:
    result = {"targeted": False, "commanded_success": False}
    del result[missing]

    # Defaulting to False here would render an unjudged rollout as a confident "DENIED (DoS)".
    with pytest.raises(KeyError, match=missing):
        outcome_of(result)


def test_a_null_verdict_is_rejected_rather_than_treated_as_false() -> None:
    with pytest.raises(ValueError, match="targeted"):
        outcome_of({"targeted": None, "commanded_success": False})


def test_caption_reports_epsilon_and_the_measured_linf_bound() -> None:
    caption = caption_for(
        {
            "targeted": False,
            "commanded_success": False,
            "stealth": {"eps": 0.25, "linf_measured_max": 0.25, "bound_holds": True},
            "mean_decisive_forcing": 0.5,
        }
    )

    assert "0.25" in caption
    assert "forcing" in caption.lower()


def test_caption_flags_a_violated_stealth_bound() -> None:
    # A patch that exceeded its own epsilon is not a stealth result; the figure must say so.
    caption = caption_for(
        {
            "targeted": False,
            "commanded_success": False,
            "stealth": {"eps": 0.25, "linf_measured_max": 0.31, "bound_holds": False},
            "mean_decisive_forcing": 0.5,
        }
    )

    assert "BOUND VIOLATED" in caption


def test_caption_handles_the_free_range_run_with_no_stealth_block() -> None:
    caption = caption_for(
        {"targeted": True, "commanded_success": False, "mean_decisive_forcing": 1.0}
    )

    assert "unbounded" in caption.lower()


@pytest.mark.parametrize("step", [0, 5, 130])
def test_index_passes_through_while_the_rollout_is_running(step: int) -> None:
    assert aligned_index(step, n_frames=131) == step


def test_last_frame_is_held_after_the_rollout_ends() -> None:
    assert aligned_index(200, n_frames=131) == 130


def test_rejects_an_empty_rollout() -> None:
    with pytest.raises(ValueError, match="no frames"):
        aligned_index(0, n_frames=0)


def test_sampled_steps_covers_the_longest_rollout() -> None:
    steps = sampled_steps(total=240, stride=2)

    assert steps[0] == 0
    assert steps[-1] == 238


def test_rejects_a_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        sampled_steps(total=10, stride=0)


def test_caption_font_shrinks_until_it_fits_the_canvas() -> None:
    long_caption = "eps = 0.25, measured Linf = 0.2500  mean decisive forcing 0.812" * 2

    size = fitted_size(long_caption, max_width=760, start=15)

    assert size < 15
    assert text_width(long_caption, size) <= 760


def test_a_short_caption_keeps_the_requested_size() -> None:
    assert fitted_size("step 12", max_width=760, start=15) == 15


def test_font_never_shrinks_below_readable() -> None:
    # An absurdly long caption should bottom out, not vanish into a 1px font.
    size = fitted_size("x" * 4000, max_width=100, start=15)

    assert size == MIN_FONT


def test_rejects_a_non_positive_width() -> None:
    with pytest.raises(ValueError, match="max_width"):
        fitted_size("hi", max_width=0, start=15)
