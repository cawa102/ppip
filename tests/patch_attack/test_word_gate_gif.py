"""The word-gate figure: does it prove the claim it makes?

The figure's whole assertion is "these two rollouts differ *only* by the word `please`". A
side-by-side that quietly paired mismatched runs -- different init, different rect, different
optimiser budget -- would look identical but claim something false. So the pairing is verified
from the two result JSONs before anything is drawn, and the panel captions are quoted from the
runs' own ``word_gate`` records rather than typed by hand.

Pure: no GPU, no frames. The drawing itself is ``rollout_gif``, already tested.
"""
from __future__ import annotations

from typing import Any

import pytest
from make_word_gate_gif import assert_only_the_word_differs, panel_subtitle, panel_title

USER = "pick up the alphabet soup and place it in the basket"
TARGET = "pick up the salad dressing and place it in the basket"


def _result(*, deploy_word: bool, **overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seed": 46,
        "rect": [160, 0, 64, 64],
        "user_task": USER,
        "target_task": TARGET,
        "patch_mode": "optimize",
        "effort": {"k": 30, "maxtries": 10, "lr": 0.03, "restarts": 3,
                   "warm_start": False, "decisive_boost": 1},
        "targeted": deploy_word,
        "commanded_success": not deploy_word,
        "word_gate": {
            "word": "please", "word_index": 0, "dormancy_weight": 1.0,
            "deploy_word": deploy_word,
            "deploy": ("please " + USER) if deploy_word else USER,
            "armed": "please " + USER, "dormant": USER,
        },
    }
    result.update(overrides)
    return result


class TestPairing:
    def test_a_matched_pair_passes(self) -> None:
        assert_only_the_word_differs(_result(deploy_word=True), _result(deploy_word=False))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seed", 24),                                  # different init state
            ("rect", [176, 0, 48, 48]),                    # different patch geometry
            ("user_task", "pick up the milk and place it in the basket"),
            ("target_task", "pick up the butter and place it in the basket"),
            ("patch_mode", "blank"),                       # not even the same attack
        ],
    )
    def test_rejects_a_pair_differing_in_more_than_the_word(
        self, field: str, value: Any
    ) -> None:
        with pytest.raises(ValueError, match=field):
            assert_only_the_word_differs(
                _result(deploy_word=True), _result(deploy_word=False, **{field: value})
            )

    def test_rejects_a_different_optimiser_budget(self) -> None:
        # A weaker budget on one side would make the gate look sharper than it is.
        weak = dict(_result(deploy_word=False)["effort"], k=10, restarts=1)
        with pytest.raises(ValueError, match="effort"):
            assert_only_the_word_differs(
                _result(deploy_word=True), _result(deploy_word=False, effort=weak)
            )

    def test_rejects_two_rollouts_of_the_same_condition(self) -> None:
        with pytest.raises(ValueError, match="armed/dormant"):
            assert_only_the_word_differs(_result(deploy_word=True), _result(deploy_word=True))

    def test_rejects_an_ungated_rollout(self) -> None:
        with pytest.raises(ValueError, match="word_gate"):
            assert_only_the_word_differs(
                _result(deploy_word=True), _result(deploy_word=False, word_gate=None)
            )

    def test_rejects_a_different_trigger_word(self) -> None:
        other = dict(_result(deploy_word=False)["word_gate"], word="kindly")
        with pytest.raises(ValueError, match="word"):
            assert_only_the_word_differs(
                _result(deploy_word=True), _result(deploy_word=False, word_gate=other)
            )


class TestFigureHorizon:
    """A figure run must outlast both outcomes it is supposed to show.

    Cutting the episode short would render a dormant rollout that *did* complete the user's task
    as "DENIED (DoS)" — the figure would libel the clean condition, and it would look entirely
    convincing doing it.
    """

    def test_horizon_clears_both_stage_c_events(self) -> None:
        from render_word_gate_figure import STAGE_C_EVENTS, horizon_for

        for init, (armed_latch, dormant_commanded) in STAGE_C_EVENTS.items():
            assert horizon_for(init) > armed_latch
            assert horizon_for(init) > dormant_commanded

    def test_unknown_init_is_refused(self) -> None:
        from render_word_gate_figure import horizon_for

        # Without a measured Stage-C outcome there is no basis for choosing a horizon.
        with pytest.raises(ValueError, match="no recorded Stage-C outcome"):
            horizon_for(999)

    def test_queued_inits_are_all_configured(self) -> None:
        from render_word_gate_figure import STAGE_C_EVENTS

        assert {46, 24, 7} <= set(STAGE_C_EVENTS)


class TestCaptions:
    def test_subtitle_quotes_the_instruction_actually_deployed(self) -> None:
        # Quoted from the run's own record -- never retyped, so the panel cannot misquote it.
        # Elided to fit the panel (TestCaptionsFit), so compare against the record's own prefix.
        armed = panel_subtitle(_result(deploy_word=True))
        dormant = panel_subtitle(_result(deploy_word=False))
        assert armed.rstrip(".").rstrip() in f'"please {USER}"'
        assert dormant.rstrip(".").rstrip() in f'"{USER}"'
        # The one-word difference stays visible on the panels themselves.
        assert armed.startswith('"please ')
        assert not dormant.startswith('"please ')

    def test_title_names_the_condition_by_the_word(self) -> None:
        assert "please" in panel_title(_result(deploy_word=True))
        assert panel_title(_result(deploy_word=True)) != panel_title(_result(deploy_word=False))

    def test_subtitle_refuses_an_ungated_rollout(self) -> None:
        with pytest.raises(ValueError):
            panel_subtitle(_result(deploy_word=True, word_gate=None))


class TestCaptionsFit:
    """Text that overruns its panel does not clip — it runs into the NEXT panel's header.

    The first render of this figure did exactly that: the armed instruction (483px at the
    subtitle size, in a 248px panel) overlapped the dormant panel's caption, so both were
    unreadable. Geometry is part of whether the figure communicates, so it is asserted.
    """

    def test_subtitle_fits_its_panel(self) -> None:
        import rollout_gif as RG
        from make_word_gate_gif import SUBTITLE_SIZE, SUBTITLE_WIDTH

        for deploy_word in (True, False):
            subtitle = panel_subtitle(_result(deploy_word=deploy_word))
            assert RG.text_width(subtitle, SUBTITLE_SIZE) <= SUBTITLE_WIDTH

    def test_elision_keeps_the_head_where_the_trigger_word_is(self) -> None:
        from make_word_gate_gif import elide_to_width

        elided = elide_to_width('"please pick up the alphabet soup and place it', 248, 14)
        assert elided.startswith('"please')
        assert elided.endswith("...")

    def test_short_text_is_left_alone(self) -> None:
        from make_word_gate_gif import elide_to_width

        assert elide_to_width('"pick up"', 248, 14) == '"pick up"'
