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


class TestCaptions:
    def test_subtitle_quotes_the_instruction_actually_deployed(self) -> None:
        # Quoted from the run's own record -- never retyped, so the panel cannot misquote it.
        assert panel_subtitle(_result(deploy_word=True)) == f'"please {USER}"'
        assert panel_subtitle(_result(deploy_word=False)) == f'"{USER}"'

    def test_title_names_the_condition_by_the_word(self) -> None:
        assert "please" in panel_title(_result(deploy_word=True))
        assert panel_title(_result(deploy_word=True)) != panel_title(_result(deploy_word=False))

    def test_subtitle_refuses_an_ungated_rollout(self) -> None:
        with pytest.raises(ValueError):
            panel_subtitle(_result(deploy_word=True, word_gate=None))
