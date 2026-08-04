"""Unit tests for ``two_branch_loss`` — WP3 of the word-gated patch experiment
(``docs/plans/2026-07-30-word-gated-patch.md``).

Mock-tensor tests (small CPU logits, no model, no GPU). The loss is the search-side
objective the word-gate optimizer minimizes:

    CE(f(P_M(o,ε), c⊕w), aᵀ)  +  λ · CE(f(P_M(o,ε), c), aᵁ)

— force the target ``aᵀ`` when the magic word is present, reproduce the clean action
``aᵁ`` when it is absent. It mirrors the existing per-frame CE
(``ce_monitor_patch_attack.py:244``): ``F.cross_entropy(logits.reshape(7,-1), teacher.reshape(7))``.
This is the *method* objective, never a reported score (the evaluator judges rollouts).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from two_branch_loss import ACTION_DIM, two_branch_loss


def _peaked_logits(tokens: torch.Tensor, *, vocab: int = 16, peak: float = 30.0) -> torch.Tensor:
    """Logits that place (nearly) all probability on ``tokens`` — one row per action dim."""
    logits = torch.zeros(ACTION_DIM, vocab)
    for i, t in enumerate(tokens.reshape(ACTION_DIM).tolist()):
        logits[i, int(t)] = peak
    return logits


TARGET = torch.tensor([1, 2, 3, 4, 5, 6, 0])
CLEAN = torch.tensor([7, 6, 5, 4, 3, 2, 1])


class TestArmedBranch:
    def test_loss_is_the_armed_ce_when_the_dormant_branch_is_perfect(self) -> None:
        # dormant logits perfectly predict the clean teacher -> dormant CE ≈ 0, so the
        # whole loss is the armed (targeting) branch's cross-entropy.
        armed_logits = torch.zeros(ACTION_DIM, 16)  # uniform -> CE = log(16)
        dormant_logits = _peaked_logits(CLEAN)

        loss = two_branch_loss(armed_logits, TARGET, dormant_logits, CLEAN, lam=1.0)

        expected_armed = F.cross_entropy(armed_logits, TARGET)
        assert float(loss) == pytest.approx(float(expected_armed), abs=1e-4)


class TestDormantBranch:
    def test_dormant_ce_is_added_at_lambda_one(self) -> None:
        # armed logits perfectly force the target (armed CE ≈ 0), dormant logits are
        # uniform (dormant CE = log(16)) -> the loss must include that dormant term.
        armed_logits = _peaked_logits(TARGET)
        dormant_logits = torch.zeros(ACTION_DIM, 16)

        loss = two_branch_loss(armed_logits, TARGET, dormant_logits, CLEAN, lam=1.0)

        expected = F.cross_entropy(armed_logits, TARGET) + F.cross_entropy(dormant_logits, CLEAN)
        assert float(loss) == pytest.approx(float(expected), abs=1e-4)


class TestLambdaWeighting:
    def test_lambda_scales_only_the_dormant_branch(self) -> None:
        # armed CE ≈ 0, so the whole loss is λ·(dormant CE): doubling λ doubles the loss.
        armed_logits = _peaked_logits(TARGET)
        dormant_logits = torch.zeros(ACTION_DIM, 16)

        loss1 = two_branch_loss(armed_logits, TARGET, dormant_logits, CLEAN, lam=1.0)
        loss2 = two_branch_loss(armed_logits, TARGET, dormant_logits, CLEAN, lam=2.0)

        assert float(loss2) == pytest.approx(2.0 * float(loss1), abs=1e-4)

    def test_lambda_zero_drops_the_dormant_branch(self) -> None:
        armed_logits = torch.zeros(ACTION_DIM, 16)
        dormant_logits = torch.zeros(ACTION_DIM, 16)

        loss = two_branch_loss(armed_logits, TARGET, dormant_logits, CLEAN, lam=0.0)

        assert float(loss) == pytest.approx(float(F.cross_entropy(armed_logits, TARGET)), abs=1e-4)


class TestShapeRobustness:
    def test_accepts_batched_logits_and_teacher_shapes(self) -> None:
        # The real optimize loop hands [1,7,V] logits and a [1,7] teacher
        # (ce_monitor_patch_attack.py:201,243); both must reduce to the flat per-token loss.
        armed_flat = _peaked_logits(TARGET)
        dormant_flat = torch.zeros(ACTION_DIM, 16)
        flat = two_branch_loss(armed_flat, TARGET, dormant_flat, CLEAN, lam=1.0)

        batched = two_branch_loss(
            armed_flat.reshape(1, ACTION_DIM, 16),
            TARGET.reshape(1, ACTION_DIM),
            dormant_flat.reshape(1, ACTION_DIM, 16),
            CLEAN.reshape(1, ACTION_DIM),
            lam=1.0,
        )

        assert float(batched) == pytest.approx(float(flat), abs=1e-4)


class TestDifferentiable:
    def test_gradient_flows_to_an_upstream_patch_parameter(self) -> None:
        # ε is the optimized variable; the loss must backprop to whatever produced the
        # logits. Simulate a single leaf feeding both branches and check its grad lands.
        raw = torch.zeros(ACTION_DIM, 16, requires_grad=True)
        loss = two_branch_loss(raw * 2.0, TARGET, raw * 0.5, CLEAN, lam=1.0)

        loss.backward()

        assert raw.grad is not None
        assert bool(torch.any(raw.grad != 0))


class TestValidation:
    def test_rejects_teacher_without_seven_action_tokens(self) -> None:
        logits = torch.zeros(ACTION_DIM, 16)
        bad_target = torch.tensor([1, 2, 3])  # not 7 tokens
        with pytest.raises(ValueError):
            two_branch_loss(logits, bad_target, logits, CLEAN, lam=1.0)

    def test_rejects_logits_not_divisible_into_seven_rows(self) -> None:
        bad_logits = torch.zeros(5, 16)  # 80 values -> not 7 action-token rows
        good_logits = torch.zeros(ACTION_DIM, 16)
        with pytest.raises(ValueError):
            two_branch_loss(bad_logits, TARGET, good_logits, CLEAN, lam=1.0)
