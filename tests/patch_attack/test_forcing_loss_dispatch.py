"""`action_loss` — the single objective dispatch shared by both optimiser paths.

Until now the static optimiser (`stealth_optimize.frame_loss`) and the per-frame closed-loop
attack (`run_confined_episode`) picked their objective independently: the first through
`forcing_loss`, the second hard-coded to `F.cross_entropy` over all 7 dims. That is how a
session ended up unable to say which objective produced which published result. One dispatch,
on raw tensors, so both callers select an objective the same way and record the same name.

Pure torch on logits — no model, no simulator.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

import forcing_loss as FL  # noqa: E402

VOCAB = 32064
ACTION_DIM = 7
#: Real action-token ids so `directional` can decode bins (`bin = 32000 - id`).
TEACHER = torch.tensor([31900, 31800, 31700, 31600, 31500, 31400, 31300])
USER = torch.tensor([31905, 31810, 31700, 31600, 31500, 31400, 31300])
DECISIVE = (0, 1)  # the dims where TEACHER and USER actually differ


def _logits(favouring: torch.Tensor | None = None, strength: float = 8.0) -> torch.Tensor:
    logits = torch.zeros(ACTION_DIM, VOCAB, requires_grad=False)
    if favouring is not None:
        logits[torch.arange(ACTION_DIM), favouring] = strength
    return logits.clone().requires_grad_(True)


def test_ce_reproduces_plain_cross_entropy_over_all_seven_dims() -> None:
    """`ce` must be bit-identical to what the closed-loop path does today.

    This is the behaviour-preservation contract: every existing corner result was produced by
    `F.cross_entropy(logits, teacher)`, so the ported dispatch has to reproduce it exactly or
    the epsilon ladder is not comparable to them.
    """
    logits = _logits()

    got = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="ce")

    assert torch.allclose(got, F.cross_entropy(logits, TEACHER))


def test_ce_ignores_the_decisive_mask() -> None:
    # `ce` is defined over all 7 dims by construction, so the mask must not change it.
    logits = _logits()

    all_dims = FL.action_loss(logits, TEACHER, USER, tuple(range(7)), objective="ce")
    two_dims = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="ce")

    assert torch.allclose(all_dims, two_dims)


def test_ce_decisive_restricts_to_the_contested_dims() -> None:
    logits = _logits()

    got = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="ce_decisive")

    assert torch.allclose(got, FL.masked_cross_entropy(logits, TEACHER, DECISIVE))


def test_hinge_is_zero_once_the_teacher_wins_by_kappa() -> None:
    # Saturation is the whole point of the margin objective: a won dim stops consuming budget.
    logits = _logits(favouring=TEACHER, strength=50.0)

    got = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="hinge", kappa=6.0)

    assert got.item() == 0.0


def test_hinge_is_positive_while_a_decisive_dim_is_unforced() -> None:
    logits = _logits(favouring=USER, strength=50.0)

    got = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="hinge", kappa=6.0)

    assert got.item() > 0.0


def test_larger_kappa_demands_a_deeper_margin() -> None:
    """The bug that capped every hinge run: too small a kappa releases a dim too early.

    A dim winning by 4 logits satisfies kappa=3 but not kappa=6, so kappa=6 keeps pushing where
    kappa=3 has already gone flat.
    """
    logits = torch.zeros(ACTION_DIM, VOCAB)
    logits[torch.arange(ACTION_DIM), TEACHER] = 4.0
    logits = logits.requires_grad_(True)

    assert FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="hinge", kappa=3.0).item() == 0
    assert FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="hinge", kappa=6.0).item() > 0


def test_default_kappa_is_six_not_three() -> None:
    """kappa=3 measured 0.667 forcing at free budget; kappa=6 measured 1.000.

    See `docs/plans/2026-08-04-epsilon-threshold-design.md` section 3 for the sweep.
    """
    assert FL.DEFAULT_KAPPA == 6.0


def test_directional_scores_progress_toward_the_teacher() -> None:
    logits = _logits(favouring=USER, strength=50.0)

    got = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="directional")

    assert got.item() > 0.0


def test_anchor_adds_a_decisive_ce_term_to_a_saturating_objective() -> None:
    # Margin 8 > kappa 6, so the hinge is saturated at zero — but 8 logits is a shallow enough
    # lead that cross-entropy is still measurably positive. At a huge lead CE underflows to 0.0
    # and the anchor would be indistinguishable from no anchor.
    logits = _logits(favouring=TEACHER, strength=8.0)

    bare = FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="hinge", kappa=6.0)
    anchored = FL.action_loss(
        logits, TEACHER, USER, DECISIVE, objective="hinge", kappa=6.0, anchor=0.1
    )

    assert bare.item() == 0.0
    assert anchored.item() > 0.0


def test_no_decisive_dims_gives_a_saturating_objective_zero_loss() -> None:
    """Agreement on every dim means there is nothing to force — so no perturbation is spent.

    This is a *feature* for stealth: on frames where the two instructions already agree the
    patch stays at its initialisation, which for the stealth parameterization is the untouched
    carrier. `ce` cannot express that; it keeps pushing regardless.
    """
    logits = _logits()

    got = FL.action_loss(logits, TEACHER, USER, (), objective="hinge", kappa=6.0)

    assert got.item() == 0.0
    assert got.requires_grad  # still differentiable, so a batch backward does not break


def test_unknown_objective_is_rejected() -> None:
    logits = _logits()

    with pytest.raises(ValueError, match="unknown objective"):
        FL.action_loss(logits, TEACHER, USER, DECISIVE, objective="nonsense")


def test_objectives_tuple_is_the_dispatch_domain() -> None:
    logits = _logits()

    for name in FL.OBJECTIVES:
        FL.action_loss(logits, TEACHER, USER, DECISIVE, objective=name)


@pytest.mark.parametrize("objective", ["ce", "ce_decisive", "hinge", "directional"])
def test_every_objective_backpropagates_to_the_logits(objective: str) -> None:
    logits = _logits(favouring=USER, strength=8.0)

    FL.action_loss(logits, TEACHER, USER, DECISIVE, objective=objective, kappa=6.0).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
