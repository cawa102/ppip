"""The epsilon bound is a published claim enforced by one line of parameterization, so it is
tested as a property (over many random `raw`), not just on a happy-path example."""

from __future__ import annotations

import pytest
import shared_inits
import torch
from stealth_patch import (
    composite,
    from_hwc,
    headroom_fraction,
    l_inf,
    stealth_patch,
    to_hwc,
    total_variation,
)


def _base(height: int = 16, width: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, height, width).clamp(0.05, 0.95)


# --- the bound ------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [0.0, 0.02, 0.08, 0.32, 1.0])
@pytest.mark.parametrize("scale", [1.0, 10.0, 1000.0])
def test_the_perturbation_never_escapes_the_ball_however_large_raw_grows(
    eps: float, scale: float
) -> None:
    base = _base()
    torch.manual_seed(1)
    raw = torch.randn_like(base) * scale

    patch = stealth_patch(raw, base, eps)

    # tanh bounds it by construction; the clamp can only shrink it further.
    assert l_inf(patch, base) <= eps + 1e-6
    assert patch.min() >= 0.0 and patch.max() <= 1.0


def test_epsilon_zero_is_exactly_the_pure_logo_control() -> None:
    base = _base()
    torch.manual_seed(2)

    patch = stealth_patch(torch.randn_like(base) * 5.0, base, 0.0)

    assert torch.equal(patch, base.clamp(0.0, 1.0))


def test_zero_init_reproduces_the_base_at_any_epsilon() -> None:
    base = _base()
    # raw=0 -> tanh(0)=0 -> the optimizer starts from the pure logo, not from noise.
    assert torch.equal(stealth_patch(torch.zeros_like(base), base, 0.32), base)


def test_a_negative_epsilon_is_rejected_rather_than_silently_flipped() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stealth_patch(torch.zeros(1, 3, 4, 4), _base(4, 4), -0.1)


def test_larger_epsilon_admits_a_larger_perturbation() -> None:
    base = _base()
    torch.manual_seed(3)
    raw = torch.randn_like(base)

    assert l_inf(stealth_patch(raw, base, 0.02), base) < l_inf(stealth_patch(raw, base, 0.32), base)


# --- clamp headroom (the confound in any comparison across carriers) --------------------


def test_a_mid_grey_base_has_the_full_epsilon_ball_available() -> None:
    assert headroom_fraction(torch.full((1, 3, 8, 8), 0.5), 0.1) == pytest.approx(1.0)


def test_a_saturated_base_loses_headroom_the_clamp_takes_away() -> None:
    # base=0.98, eps=0.1 -> reachable [0.88, 1.0] = 0.12 of the nominal 0.2 ball.
    assert headroom_fraction(torch.full((1, 3, 8, 8), 0.98), 0.1) == pytest.approx(0.6)


def test_a_near_white_carrier_has_strictly_less_budget_than_grey_at_the_same_epsilon() -> None:
    # This is why a base sweep cannot attribute a difference to structure alone.
    white_ish = torch.full((1, 3, 8, 8), 0.96)  # solstice's #FDF6E3 foreground
    grey = torch.full((1, 3, 8, 8), 0.5)

    assert headroom_fraction(white_ish, 0.32) < headroom_fraction(grey, 0.32)


def test_headroom_is_undefined_rather_than_one_at_epsilon_zero() -> None:
    # eps=0 is the pure-logo control: there is no ball, so no share of it is reachable.
    assert headroom_fraction(torch.full((1, 3, 8, 8), 0.5), 0.0) == 0.0


# --- total variation ------------------------------------------------------------------


def test_total_variation_is_zero_on_a_flat_field_and_positive_on_noise() -> None:
    torch.manual_seed(4)

    assert total_variation(torch.full((1, 3, 8, 8), 0.5)).item() == pytest.approx(0.0)
    assert total_variation(torch.rand(1, 3, 8, 8)).item() > 0.1


def test_total_variation_penalises_high_frequency_more_than_a_smooth_ramp() -> None:
    ramp = torch.linspace(0, 1, 8).view(1, 1, 1, 8).expand(1, 3, 8, 8).contiguous()
    checkerboard = torch.zeros(1, 3, 8, 8)
    checkerboard[..., ::2, ::2] = 1.0
    checkerboard[..., 1::2, 1::2] = 1.0

    assert total_variation(ramp) < total_variation(checkerboard)


# --- compositing ----------------------------------------------------------------------


def test_composite_replaces_only_the_rect_and_leaves_the_frame_untouched() -> None:
    frame = torch.zeros(1, 3, 224, 224)
    patch = torch.ones(1, 3, 64, 64)
    rect = (160, 0, 64, 64)  # BL 64x64

    out = composite(frame, patch, rect)

    assert out[:, :, 160:224, 0:64].min() == 1.0
    assert out[:, :, 0:160, :].max() == 0.0  # everything above the rect untouched
    assert out[:, :, :, 64:].max() == 0.0  # everything right of the rect untouched
    assert frame.max() == 0.0  # the input was not mutated


def test_gradients_reach_raw_through_the_composite() -> None:
    base = _base(64, 64)
    raw = torch.zeros_like(base, requires_grad=True)
    frame = torch.zeros(1, 3, 224, 224)

    out = composite(frame, stealth_patch(raw, base, 0.1), (160, 0, 64, 64))
    out.sum().backward()

    assert raw.grad is not None and raw.grad.abs().sum() > 0


def test_a_patch_that_does_not_fill_its_rect_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not fill rect"):
        composite(torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 32, 32), (160, 0, 64, 64))


def test_a_rect_outside_the_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        composite(torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 64, 64), (200, 0, 64, 64))


# --- layout round-trip ----------------------------------------------------------------


def test_hwc_round_trip_preserves_the_patch() -> None:
    patch = _base(8, 8)
    assert torch.allclose(from_hwc(to_hwc(patch)), patch)


def test_to_hwc_produces_the_layout_the_evaluator_expects() -> None:
    # HijackBackend.set_patch indexes the patch as [h, w, 3].
    assert to_hwc(_base(64, 64)).shape == (64, 64, 3)


# --- the split the gate depends on ----------------------------------------------------


def test_optimize_and_gate_inits_partition_the_training_half() -> None:
    shared_inits.verify_precommit()

    assert not set(shared_inits.OPTIMIZE_INITS) & set(shared_inits.GATE_INITS)
    assert set(shared_inits.OPTIMIZE_INITS) | set(shared_inits.GATE_INITS) == set(
        shared_inits.TRAIN_INITS
    )
    assert not set(shared_inits.GATE_INITS) & set(shared_inits.HELDOUT_INITS)


# --- checkpointing (multi-hour runs must survive being killed) --------------------------


def test_checkpoint_round_trip_restores_parameter_step_and_adam_moments(tmp_path) -> None:
    from stealth_optimize import load_checkpoint, save_checkpoint

    raw = torch.zeros(1, 3, 4, 4, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.1)
    for _ in range(3):  # build up Adam state so the restore has something to preserve
        optimizer.zero_grad()
        (raw * torch.arange(48.0).view(1, 3, 4, 4)).sum().backward()
        optimizer.step()
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, raw, optimizer, 3)
    expected = raw.detach().clone()
    expected_moment = optimizer.state[raw]["exp_avg"].clone()

    fresh = torch.zeros(1, 3, 4, 4, requires_grad=True)
    fresh_optimizer = torch.optim.Adam([fresh], lr=0.1)
    step = load_checkpoint(path, fresh, fresh_optimizer)

    assert step == 3
    assert torch.allclose(fresh.detach(), expected)
    # Restoring the parameter alone would restart Adam cold and discard its adaptive scaling.
    assert torch.allclose(fresh_optimizer.state[fresh]["exp_avg"], expected_moment)


def test_a_resumed_run_continues_rather_than_restarting(tmp_path) -> None:
    from stealth_optimize import load_checkpoint, save_checkpoint

    raw = torch.full((1, 3, 2, 2), 0.5, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.1)
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, raw, optimizer, 700)

    resumed = torch.zeros(1, 3, 2, 2, requires_grad=True)
    start = load_checkpoint(path, resumed, torch.optim.Adam([resumed], lr=0.1))

    assert start == 700  # the loop must resume at 700, not redo the first 700 steps
    assert torch.allclose(resumed.detach(), torch.full((1, 3, 2, 2), 0.5))


# --- masked regions (the area axis past what one rect allows) ---------------------------


def test_a_two_band_mask_covers_both_bands_and_nothing_between() -> None:
    from stealth_patch import rects_to_mask

    mask = rects_to_mask([(0, 0, 22, 224), (161, 0, 63, 224)])

    assert mask.shape == (1, 1, 224, 224)
    assert mask[0, 0, :22, :].min() == 1.0
    assert mask[0, 0, 161:, :].min() == 1.0
    assert mask[0, 0, 22:161, :].max() == 0.0  # the objects' rows stay untouched


def test_mask_area_matches_the_reported_fraction() -> None:
    from stealth_patch import rects_to_mask

    # 22 + 63 = 85 rows of 224 -> 37.9% of the frame.
    assert rects_to_mask([(0, 0, 22, 224), (161, 0, 63, 224)]).mean().item() == pytest.approx(
        85 / 224, abs=1e-6
    )


def test_a_region_outside_the_frame_is_rejected() -> None:
    from stealth_patch import rects_to_mask

    with pytest.raises(ValueError, match="outside"):
        rects_to_mask([(200, 0, 60, 224)])


def test_masked_composite_replaces_only_the_masked_pixels() -> None:
    from stealth_patch import composite_masked, rects_to_mask

    frame = torch.zeros(1, 3, 224, 224)
    patch = torch.ones(1, 3, 224, 224)
    mask = rects_to_mask([(0, 0, 22, 224), (161, 0, 63, 224)])

    out = composite_masked(frame, patch, mask)

    assert out[:, :, :22, :].min() == 1.0
    assert out[:, :, 161:, :].min() == 1.0
    assert out[:, :, 22:161, :].max() == 0.0
    assert frame.max() == 0.0  # input untouched


def test_gradients_reach_raw_through_the_masked_composite() -> None:
    from stealth_patch import composite_masked, rects_to_mask

    base = torch.full((1, 3, 224, 224), 0.5)
    raw = torch.zeros_like(base, requires_grad=True)
    mask = rects_to_mask([(0, 0, 22, 224), (161, 0, 63, 224)])

    patched = stealth_patch(raw, base, 1.0)
    composite_masked(torch.zeros(1, 3, 224, 224), patched, mask).sum().backward()

    assert raw.grad is not None
    # Gradient flows only where the mask is on -- the untouched band gets exactly zero.
    assert raw.grad[0, 0, 100, 100].item() == 0.0
    assert raw.grad[0, 0, 5, 5].abs().item() > 0
