"""Wiring of the selectable objective and the worst-case batch selection into `optimize`.

`forcing_loss` unit-tests the objectives on synthetic logits; this pins the parts only the
optimizer owns — that gradients still reach `raw` through the refactored forward, that the
CVaR pool really spends its gradient on the *hardest* frames rather than a random draw, and
that `--objective ce` still reproduces the pre-2026-07-30 loss so old runs stay comparable.

The model is a stub: a differentiable function of the composited pixels, so the loop is
exercised end to end on CPU without loading 7B of weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import stealth_loop
import stealth_optimize as SO
import torch
import vla_diff
from forcing_loss import ACTION_DIM, VOCAB_ANCHOR, bin_to_token
from stealth_optimize import Frame, frame_loss

VOCAB = VOCAB_ANCHOR + 64
RECT = (100, 100, 8, 8)


def _frame(fill: int, teacher_bins: tuple[int, ...], user_bins: tuple[int, ...]) -> Frame:
    return Frame(
        init=0,
        step=fill,
        image=np.full((224, 224, 3), fill, dtype=np.uint8),
        teacher=tuple(bin_to_token(b) for b in teacher_bins),
        clean_user=tuple(bin_to_token(b) for b in user_bins),
        decisive_dims=tuple(i for i in range(ACTION_DIM) if teacher_bins[i] != user_bins[i]),
    )


def _frames() -> list[Frame]:
    teacher = (10, 20, 30, 40, 50, 60, 70)
    user = (99, 20, 130, 40, 50, 160, 70)  # dims 0, 2, 5 contested
    return [_frame(fill, teacher, user) for fill in (20, 70, 120, 170, 220)]


class _StubPolicy:
    """Records every call; brighter frames are easier, so the darkest are the hardest."""

    def __init__(self) -> None:
        self.grad_markers: list[float] = []
        self.nograd_markers: list[float] = []

    def __call__(
        self,
        model: object,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor,
        target_action_ids: torch.Tensor,
    ) -> torch.Tensor:
        signal = pixel_values.mean() * 5.0  # differentiable in the patch
        (self.grad_markers if torch.is_grad_enabled() else self.nograd_markers).append(
            float(signal.detach())
        )
        onehot = torch.zeros(ACTION_DIM, VOCAB)
        onehot[range(ACTION_DIM), target_action_ids.reshape(ACTION_DIM)] = 1.0
        return (onehot * signal).unsqueeze(0)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubPolicy:
    policy = _StubPolicy()
    monkeypatch.setattr(SO, "DEVICE", "cpu")
    monkeypatch.setattr(vla_diff, "action_token_logits", policy)
    return policy


def _optimize(stub_frames: list[Frame], **kwargs: object) -> tuple[np.ndarray, dict]:
    base = torch.full((1, 3, RECT[2], RECT[3]), 0.5)
    defaults: dict = dict(
        base=base, eps=0.5, rect=RECT, user_ids=torch.zeros(1, 4, dtype=torch.long),
        steps=1, batch_size=2, lr=0.1, tv_weight=0.0, seed=0, log_every=1000,
    )
    return SO.optimize(object(), object(), stub_frames, **{**defaults, **kwargs})


# --- the loss still reaches the patch ----------------------------------------------------


@pytest.mark.parametrize("objective", SO.OBJECTIVES)
def test_every_objective_moves_the_patch_and_respects_the_epsilon_ball(
    stub: _StubPolicy, objective: str
) -> None:
    patch, diagnostics = _optimize(_frames(), steps=3, objective=objective)

    assert diagnostics["objective"] == objective
    assert np.abs(patch - 0.5).max() > 0.0, "no gradient reached raw"
    assert diagnostics["linf_measured"] <= 0.5 + 1e-6


def test_the_legacy_objective_is_still_exactly_cross_entropy_over_all_seven_dims() -> None:
    frame = _frames()[0]
    logits = torch.randn(ACTION_DIM, VOCAB)
    teacher = torch.tensor(frame.teacher, dtype=torch.long)

    legacy = torch.nn.functional.cross_entropy(logits, teacher)
    got = frame_loss(logits, frame, objective="ce", kappa=3.0, temperature=1.0, anchor=0.0)

    assert got.item() == pytest.approx(legacy.item(), rel=1e-6)


def test_the_decisive_objectives_ignore_the_dims_the_instructions_agree_on() -> None:
    frame = _frames()[0]
    logits = torch.randn(ACTION_DIM, VOCAB)
    settled = [i for i in range(ACTION_DIM) if i not in frame.decisive_dims]
    terms = dict(kappa=3.0, temperature=1.0, anchor=0.0)

    before = frame_loss(logits, frame, objective="ce_decisive", **terms)  # type: ignore[arg-type]
    logits[settled] = torch.randn(len(settled), VOCAB) * 50.0  # wreck the settled dims
    after = frame_loss(logits, frame, objective="ce_decisive", **terms)  # type: ignore[arg-type]

    assert after.item() == pytest.approx(before.item())


def test_an_unknown_objective_is_rejected_before_any_gpu_time_is_spent(
    stub: _StubPolicy,
) -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        _optimize(_frames(), objective="mse")
    assert stub.grad_markers == []  # rejected up front, not part-way through a run


# --- worst-case (CVaR) batch selection ---------------------------------------------------


def test_without_a_pool_the_batch_is_drawn_at_random_and_costs_no_extra_forwards(
    stub: _StubPolicy,
) -> None:
    _optimize(_frames(), steps=1, batch_size=2, pool_size=0)

    assert len(stub.grad_markers) == 2
    assert stub.nograd_markers == []  # the scoring pass must not run when it is disabled


def test_a_pool_scores_every_candidate_frame_then_spends_gradient_on_the_hardest(
    stub: _StubPolicy,
) -> None:
    frames = _frames()

    _optimize(frames, steps=1, batch_size=2, pool_size=len(frames))

    assert len(stub.nograd_markers) == len(frames), "the whole pool must be scored"
    assert len(stub.grad_markers) == 2, "only the batch gets a backward"
    # In the stub a brighter frame gives a larger teacher logit, so the darkest frames carry
    # the largest loss. Those are exactly the two that must receive the gradient.
    hardest = sorted(stub.nograd_markers)[:2]
    assert sorted(stub.grad_markers) == pytest.approx(sorted(hardest), rel=1e-3)


def test_a_pool_smaller_than_the_batch_degrades_to_no_selection(stub: _StubPolicy) -> None:
    _optimize(_frames(), steps=1, batch_size=4, pool_size=2)

    assert stub.nograd_markers == []
    assert len(stub.grad_markers) == 4


# --- what the run records ----------------------------------------------------------------


def test_diagnostics_separate_nominal_area_from_what_the_model_actually_consumes(
    stub: _StubPolicy,
) -> None:
    _patch, diagnostics = _optimize(
        _frames(), rect=(160, 0, 64, 64), base=torch.full((1, 3, 64, 64), 0.5)
    )

    assert diagnostics["nominal_area_fraction"] == pytest.approx(64 * 64 / (224 * 224), abs=1e-4)
    # A bottom-left flush rect straddles the crop border, so the model sees less than nominal.
    assert diagnostics["input_area_fraction"] < diagnostics["nominal_area_fraction"]
    assert diagnostics["retained_fraction"] == pytest.approx(0.829, abs=0.005)


def test_diagnostics_record_the_clamp_headroom_that_confounds_a_base_sweep(
    stub: _StubPolicy,
) -> None:
    _patch, diagnostics = _optimize(_frames(), eps=0.5, base=torch.full((1, 3, 8, 8), 0.9))

    # base=0.9, eps=0.5 -> reachable [0.4, 1.0] = 0.6 of the nominal 1.0 ball.
    assert diagnostics["headroom_fraction"] == pytest.approx(0.6, abs=1e-5)


def test_the_tail_of_the_loss_is_recorded_not_only_its_mean(stub: _StubPolicy) -> None:
    _patch, diagnostics = _optimize(_frames(), steps=2, pool_size=5)

    assert diagnostics["final_loss_worst_quartile"] >= diagnostics["final_loss"] - 1e-6


# --- the ledger must not confuse two objectives ------------------------------------------


def test_candidate_ids_separate_objectives_so_ledger_rows_cannot_collide() -> None:
    ids = {
        stealth_loop.candidate_id("aurora", 1.0, 0.0, "BL:64", 0, objective)
        for objective in SO.OBJECTIVES
    }

    assert len(ids) == len(SO.OBJECTIVES)


def test_the_legacy_objective_keeps_the_original_id_so_old_rows_still_resume() -> None:
    # A renamed id would make the loop re-run every candidate already in the ledger.
    assert stealth_loop.candidate_id("aurora", 1.0, 0.0, "BL:64", 0, "ce") == (
        "aurora_eps1_tv0_BL64_s0"
    )
