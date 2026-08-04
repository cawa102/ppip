"""The additive stealth kwargs for the per-frame confined attack (`run_confined_episode`).

The per-frame attack is the only regime that has ever produced a targeted hijack, and it
builds its patch as `sigmoid(raw)` — free-range, any pixel value. Constraining that to an
L-infinity ball around a logo is the stealth question asked of the *working* mechanism
rather than the static one, so the resolver below has to be exact about one thing: when
both kwargs are absent the caller must get the original free-range path back, untouched.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

import stealth_patch as SP  # noqa: E402

FRAME = (1, 3, 224, 224)


def _base(value: float = 0.5) -> torch.Tensor:
    return torch.full(FRAME, value)


def test_returns_none_when_neither_kwarg_is_given() -> None:
    # Arrange / Act
    resolved = SP.resolve_confined_stealth(None, None)

    # Assert — the free-range caller keeps its original path
    assert resolved is None


def test_raises_when_base_given_without_eps() -> None:
    with pytest.raises(ValueError, match="both"):
        SP.resolve_confined_stealth(_base(), None)


def test_raises_when_eps_given_without_base() -> None:
    with pytest.raises(ValueError, match="both"):
        SP.resolve_confined_stealth(None, 0.06)


@pytest.mark.parametrize("eps", [-0.01, 1.5])
def test_raises_when_eps_outside_unit_interval(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        SP.resolve_confined_stealth(_base(), eps)


def test_accepts_eps_zero_as_the_pure_logo_control() -> None:
    # eps=0 is a legitimate run, not a degenerate one: it is the control that shows the
    # hijack comes from the perturbation and not from the logo merely being present.
    resolved = SP.resolve_confined_stealth(_base(), 0.0)

    assert resolved is not None
    assert resolved.eps == 0.0


def test_raises_when_base_is_not_a_full_frame() -> None:
    with pytest.raises(ValueError, match="shape"):
        SP.resolve_confined_stealth(torch.full((1, 3, 64, 64), 0.5), 0.06)


def test_valid_kwargs_resolve_to_base_and_eps() -> None:
    base = _base(0.3)

    resolved = SP.resolve_confined_stealth(base, 0.06)

    assert resolved is not None
    assert resolved.eps == 0.06
    assert torch.equal(resolved.base, base)


def test_zero_raw_reproduces_the_base_exactly() -> None:
    # The initialisation the confined loop uses (`raw = zeros`) must start at the pure logo,
    # so step 0 of a stealth run is the control rather than an arbitrary point in the ball.
    base = _base(0.42)
    resolved = SP.resolve_confined_stealth(base, 0.06)
    assert resolved is not None

    patch = SP.stealth_patch(torch.zeros(FRAME), resolved.base, resolved.eps)

    assert torch.equal(patch, base)


def test_bound_holds_for_the_restart_initialisation() -> None:
    # Later restarts init `raw ~ randn*1.5`, which saturates tanh; the ball must still hold.
    base = _base(0.42)
    resolved = SP.resolve_confined_stealth(base, 0.06)
    assert resolved is not None

    patch = SP.stealth_patch(torch.randn(FRAME) * 1.5, resolved.base, resolved.eps)

    assert SP.l_inf(patch, base) <= 0.06 + 1e-6
