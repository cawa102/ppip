"""The forcing objectives — what the static patch is actually asked to achieve.

Three properties carry the whole design and each gets a test that would fail on the old
plain-cross-entropy-over-all-7-dims objective:

1. **Only decisive dims count.** The user- and target-instructed policies already agree on
   ~3 of 7 dims; gradient spent there is wasted *and* competes for the same epsilon budget.
2. **A won dim stops costing.** Cross-entropy keeps widening a logit gap that is already
   decisive; with one static patch serving many frames, that is capacity taken from a frame
   that has not been won yet. The hinge goes to zero and stays there.
3. **Distance between bins is meaningful.** The action head is categorical over 256 ordered
   bins, and the user/target disagreement is a median of ~23 bins — so "nearly right" and
   "wildly wrong" must not cost the same, and overshooting *toward* the target must not be
   punished like moving away from it.
"""

from __future__ import annotations

import pytest
import torch
from forcing_loss import (
    ACTION_ID_MAX,
    ACTION_ID_MIN,
    N_BINS,
    VOCAB_ANCHOR,
    bin_to_token,
    cvar,
    directional_hinge,
    margin_hinge,
    masked_cross_entropy,
    saturating_cross_entropy,
    soft_bins,
    token_to_bin,
    won_dims,
)

VOCAB = 32064
DIMS = (0, 2, 5)


def _logits(winner_ids: list[int], margin: float = 5.0) -> torch.Tensor:
    """`[7, VOCAB]` logits whose argmax on each dim is `winner_ids[dim]`, by `margin`."""
    logits = torch.zeros(7, VOCAB)
    for dim, token in enumerate(winner_ids):
        logits[dim, token] = margin
    return logits


def _tokens(bins: list[int]) -> torch.Tensor:
    return torch.tensor([bin_to_token(b) for b in bins], dtype=torch.long)


# --- the token/bin convention -----------------------------------------------------------


def test_the_token_to_bin_map_matches_openvla_s_id_equals_32000_minus_bin() -> None:
    assert token_to_bin(torch.tensor([ACTION_ID_MAX])).tolist() == [1]
    assert token_to_bin(torch.tensor([ACTION_ID_MIN])).tolist() == [N_BINS]
    assert bin_to_token(1) == VOCAB_ANCHOR - 1
    assert ACTION_ID_MAX - ACTION_ID_MIN + 1 == N_BINS


# --- 1. only decisive dims count --------------------------------------------------------


def test_cross_entropy_ignores_dims_the_two_instructions_already_agree_on() -> None:
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    logits = _logits([bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)])
    before = masked_cross_entropy(logits, teacher, DIMS)

    # Wreck a NON-decisive dim: a decisive-dim objective must not notice.
    logits[1] = 0.0
    logits[1, bin_to_token(200)] = 9.0

    assert masked_cross_entropy(logits, teacher, DIMS) == pytest.approx(before.item())


def test_all_seven_dims_reproduces_plain_cross_entropy() -> None:
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    logits = torch.randn(7, VOCAB)

    expected = torch.nn.functional.cross_entropy(logits, teacher)

    assert masked_cross_entropy(logits, teacher, tuple(range(7))) == pytest.approx(
        expected.item(), rel=1e-5
    )


@pytest.mark.parametrize(
    "loss_fn",
    [
        lambda lg, t, d: masked_cross_entropy(lg, t, d),
        lambda lg, t, d: margin_hinge(lg, t, d, kappa=1.0),
        lambda lg, t, d: saturating_cross_entropy(lg, t, d, kappa=1.0),
    ],
)
def test_a_frame_with_no_decisive_dims_costs_nothing_but_stays_differentiable(loss_fn) -> None:
    logits = torch.randn(7, VOCAB, requires_grad=True)

    loss = loss_fn(logits, _tokens([1] * 7), ())
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None  # a zero that detaches would break the batch backward


# --- 2. a won dim stops costing ---------------------------------------------------------


def test_the_hinge_is_zero_once_the_target_token_wins_by_the_margin() -> None:
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    logits = _logits([bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)], margin=5.0)

    assert margin_hinge(logits, teacher, DIMS, kappa=1.0).item() == 0.0


def test_the_hinge_is_positive_while_a_decisive_dim_is_still_lost() -> None:
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    logits = _logits([bin_to_token(b) for b in (99, 20, 30, 40, 50, 60, 70)], margin=5.0)

    assert margin_hinge(logits, teacher, DIMS, kappa=1.0).item() > 0.0


def test_over_winning_a_dim_costs_nothing_more_under_the_hinge_but_still_pays_under_ce() -> None:
    """The capacity property: this is the reason to prefer the hinge with one static patch."""
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    won = _logits([bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)], margin=5.0)
    won_harder = _logits([bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)], margin=50.0)

    # The hinge has nothing left to gain -> the gradient goes to the frames still unforced.
    assert margin_hinge(won, teacher, DIMS, kappa=1.0).item() == 0.0
    assert margin_hinge(won_harder, teacher, DIMS, kappa=1.0).item() == 0.0
    # Cross-entropy keeps paying to widen an already-decisive gap.
    assert masked_cross_entropy(won_harder, teacher, DIMS) < masked_cross_entropy(
        won, teacher, DIMS
    )


def test_the_hinge_gradient_vanishes_on_won_dims_and_survives_on_lost_ones() -> None:
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    logits = _logits([bin_to_token(b) for b in (10, 20, 99, 40, 50, 60, 70)], margin=5.0)
    logits.requires_grad_(True)

    margin_hinge(logits, teacher, (0, 2), kappa=1.0).backward()

    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() == 0.0  # dim 0 already won
    assert logits.grad[2].abs().sum().item() > 0.0  # dim 2 still lost


# --- 2b. saturation, isolated from the shape of the penalty ------------------------------
#
# `ce_saturating` exists because comparing `ce` against `hinge` changes two things at once —
# the penalty's *shape* and whether it *saturates* — so a difference between them cannot be
# attributed to either. It completes the grid:
#
#     shape \ saturation |  off             on
#     -------------------+---------------------------------
#     log-loss           |  ce_decisive     ce_saturating
#     linear margin      |  --              hinge
#
# `ce_decisive` -> `ce_saturating` isolates saturation; `ce_saturating` -> `hinge` isolates
# shape. The tests below pin both edges of that grid.


def _mixed_logits(won_margin: float, lost_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits where each dim leads by `won_margin`, except `lost_dim` whose lead is wrong."""
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    winners = [bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)]
    winners[lost_dim] = bin_to_token(199)  # a token the teacher never asked for
    return _logits(winners, margin=won_margin), teacher


def test_won_dims_is_exactly_the_set_on_which_the_hinge_has_gone_flat() -> None:
    """Both saturating objectives must agree on what counts as won.

    If they disagreed, `ce_saturating` vs `hinge` would compare release *thresholds* rather
    than penalty shapes, and the grid would answer a different question than the one asked.
    """
    logits, teacher = _mixed_logits(won_margin=8.0, lost_dim=2)

    won = won_dims(logits, teacher, DIMS, kappa=6.0)

    assert won == (0, 5)
    for dim in won:
        assert margin_hinge(logits, teacher, (dim,), kappa=6.0).item() == 0.0
    assert margin_hinge(logits, teacher, (2,), kappa=6.0).item() > 0.0


def test_saturating_cross_entropy_is_cross_entropy_over_only_the_unwon_dims() -> None:
    logits, teacher = _mixed_logits(won_margin=8.0, lost_dim=2)

    assert saturating_cross_entropy(logits, teacher, DIMS, kappa=6.0) == pytest.approx(
        masked_cross_entropy(logits, teacher, (2,)).item()
    )


def test_saturating_cross_entropy_is_plain_decisive_ce_while_nothing_is_won_yet() -> None:
    """With no dim past the margin the saturation never fires, so the two must coincide.

    This is what makes the grid controlled: the only difference `ce_saturating` introduces
    over `ce_decisive` is the release of won dims, never a change of scale or of support.
    """
    logits, teacher = _mixed_logits(won_margin=2.0, lost_dim=2)  # 2 < kappa: nothing won

    assert saturating_cross_entropy(logits, teacher, DIMS, kappa=6.0) == pytest.approx(
        masked_cross_entropy(logits, teacher, DIMS).item()
    )


def test_a_won_dim_stops_receiving_gradient_under_saturating_cross_entropy() -> None:
    """The property `ce_decisive` lacks: budget a won dim would have spent goes to a lost one."""
    logits, teacher = _mixed_logits(won_margin=8.0, lost_dim=2)
    logits.requires_grad_(True)

    saturating_cross_entropy(logits, teacher, DIMS, kappa=6.0).backward()

    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() == 0.0  # won -> released
    assert logits.grad[5].abs().sum().item() == 0.0  # won -> released
    assert logits.grad[2].abs().sum().item() > 0.0  # still lost -> still pushed


def test_ce_decisive_by_contrast_keeps_paying_on_a_dim_it_has_already_won() -> None:
    """The contrast that gives the previous test its meaning."""
    logits, teacher = _mixed_logits(won_margin=8.0, lost_dim=2)
    logits.requires_grad_(True)

    masked_cross_entropy(logits, teacher, DIMS).backward()

    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() > 0.0


def test_over_winning_is_free_under_saturating_ce_but_still_paid_under_ce_decisive() -> None:
    teacher = _tokens([10, 20, 30, 40, 50, 60, 70])
    winners = [bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)]
    won = _logits(winners, margin=8.0)
    won_harder = _logits(winners, margin=50.0)

    assert saturating_cross_entropy(won, teacher, DIMS, kappa=6.0).item() == 0.0
    assert saturating_cross_entropy(won_harder, teacher, DIMS, kappa=6.0).item() == 0.0
    assert masked_cross_entropy(won_harder, teacher, DIMS) < masked_cross_entropy(
        won, teacher, DIMS
    )


def test_the_two_saturating_objectives_release_the_same_dims_at_different_cost() -> None:
    """The shape edge of the grid: identical saturation set, different penalty on the rest."""
    logits, teacher = _mixed_logits(won_margin=8.0, lost_dim=2)

    ce_like = saturating_cross_entropy(logits, teacher, DIMS, kappa=6.0)
    hinge_like = margin_hinge(logits, teacher, DIMS, kappa=6.0)

    assert ce_like.item() > 0.0
    assert hinge_like.item() > 0.0
    assert ce_like.item() != pytest.approx(hinge_like.item())


def test_a_shallow_win_is_released_only_once_kappa_allows_it() -> None:
    """Same kappa semantics as the hinge — won means leading *by kappa*, not merely leading.

    This is the knob that saturation costs. `ce_decisive` has no such parameter and so cannot
    be silently mis-set the way kappa=3 capped every hinge run before 2026-08-04.
    """
    logits, teacher = _mixed_logits(won_margin=4.0, lost_dim=2)

    assert won_dims(logits, teacher, DIMS, kappa=3.0) == (0, 5)
    assert won_dims(logits, teacher, DIMS, kappa=6.0) == ()


def test_the_won_set_is_decided_without_leaking_gradient_through_the_comparison() -> None:
    """A differentiable mask would push logits toward *looking* won instead of *being* won."""
    logits, teacher = _mixed_logits(won_margin=8.0, lost_dim=2)
    logits.requires_grad_(True)

    won_dims(logits, teacher, DIMS, kappa=6.0)

    assert logits.grad is None


# --- 3. bins are ordered, and direction matters -----------------------------------------


def test_soft_bins_decodes_a_confident_prediction_to_its_own_bin() -> None:
    logits = _logits([bin_to_token(b) for b in (10, 20, 30, 40, 50, 60, 70)], margin=40.0)

    assert soft_bins(logits, temperature=1.0)[0].item() == pytest.approx(10.0, abs=0.1)
    assert soft_bins(logits, temperature=1.0)[6].item() == pytest.approx(70.0, abs=0.1)


def test_the_directional_objective_is_satisfied_once_the_action_reaches_the_teacher() -> None:
    user = _tokens([100] * 7)
    teacher = _tokens([130] * 7)
    at_teacher = _logits([bin_to_token(130)] * 7, margin=40.0)

    assert directional_hinge(at_teacher, teacher, user, DIMS).item() == pytest.approx(0.0, abs=1e-3)


def test_overshooting_toward_the_target_is_not_punished() -> None:
    """A redirection attack that pushes *past* the teacher is still a redirection."""
    user = _tokens([100] * 7)
    teacher = _tokens([130] * 7)
    past_teacher = _logits([bin_to_token(160)] * 7, margin=40.0)

    assert directional_hinge(past_teacher, teacher, user, DIMS).item() == pytest.approx(
        0.0, abs=1e-3
    )


def test_moving_away_from_the_target_costs_more_than_not_moving() -> None:
    user = _tokens([100] * 7)
    teacher = _tokens([130] * 7)
    unmoved = _logits([bin_to_token(100)] * 7, margin=40.0)
    backwards = _logits([bin_to_token(70)] * 7, margin=40.0)

    assert directional_hinge(backwards, teacher, user, DIMS) > directional_hinge(
        unmoved, teacher, user, DIMS
    )


def test_a_near_miss_costs_less_than_a_wild_miss_unlike_cross_entropy() -> None:
    """Cross-entropy treats the 256 bins as unordered labels; the action space is not."""
    user = _tokens([100] * 7)
    teacher = _tokens([130] * 7)
    near = _logits([bin_to_token(128)] * 7, margin=40.0)
    wild = _logits([bin_to_token(105)] * 7, margin=40.0)

    assert directional_hinge(near, teacher, user, DIMS) < directional_hinge(
        wild, teacher, user, DIMS
    )
    # ...whereas cross-entropy charges both the same: neither is the teacher's token.
    assert masked_cross_entropy(near, teacher, DIMS) == pytest.approx(
        masked_cross_entropy(wild, teacher, DIMS).item(), rel=1e-4
    )


def test_a_dim_where_the_two_instructions_agree_contributes_nothing_to_the_direction() -> None:
    # No disagreement -> no direction to travel -> the term must not blow up on a zero divisor.
    same = _tokens([100] * 7)
    logits = _logits([bin_to_token(100)] * 7, margin=40.0)

    loss = directional_hinge(logits, same, same, DIMS)

    assert torch.isfinite(loss) and loss.item() == pytest.approx(0.0, abs=1e-6)


# --- worst-case aggregation -------------------------------------------------------------


def test_cvar_at_q_one_is_the_plain_mean() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert cvar(values, 1.0).item() == pytest.approx(2.5)


def test_cvar_averages_only_the_worst_frames() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert cvar(values, 0.5).item() == pytest.approx(3.5)  # mean of {4, 3}


def test_cvar_always_keeps_at_least_one_frame() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 9.0])

    assert cvar(values, 0.01).item() == pytest.approx(9.0)


def test_cvar_is_never_below_the_mean() -> None:
    torch.manual_seed(0)
    values = torch.rand(32)

    for q in (0.1, 0.25, 0.5, 1.0):
        assert cvar(values, q).item() >= values.mean().item() - 1e-6


def test_cvar_rejects_a_quantile_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="q must be"):
        cvar(torch.tensor([1.0]), 0.0)
