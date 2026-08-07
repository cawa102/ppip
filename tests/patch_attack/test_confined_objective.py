"""The `objective` kwargs on `run_confined_episode` — a selectable per-frame loss.

Until now this path hard-coded `F.cross_entropy` over all 7 action dims and recorded no objective
in its result JSON, so an artifact could not say which loss produced it. The epsilon-threshold
experiment needs the margin objective here (a minimum-perturbation threshold measured with a
non-saturating loss is an upper bound, not a threshold), without disturbing any published corner
result.

Split like the other GPU seams: the fail-fast validation is reachable on CPU with a dummy backend
because it runs before any policy load, and the live loop is a `@requires_gpu` smoke test.
See `docs/plans/2026-08-04-epsilon-threshold-design.md` section 4.1.
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env to run the OpenVLA seam",
)

PROBE_RECT = (160, 0, 64, 64)  # the measured-clear BL 64x64 corner
USER_TASK = "pick up the alphabet soup and place it in the basket"


def test_unknown_objective_is_rejected_before_any_gpu_work() -> None:
    """A typo must not surface hours into a rollout, or after one.

    The dummy backend proves the raise precedes any policy load or env build.
    """
    from ce_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="unknown objective"):
        run_confined_episode(
            object(),  # never used: validation raises first
            rect=PROBE_RECT,
            seed=0,
            run_dir="/tmp/objective-unused",
            tag="obj_guard",
            objective="hinj",  # typo
        )


def test_negative_kappa_is_rejected_before_any_gpu_work() -> None:
    from ce_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="kappa"):
        run_confined_episode(
            object(),
            rect=PROBE_RECT,
            seed=0,
            run_dir="/tmp/objective-unused",
            tag="obj_kappa",
            objective="hinge",
            kappa=-1.0,
        )


def test_default_objective_is_ce_so_existing_callers_are_unchanged() -> None:
    """Behaviour preservation: every published corner result came from all-7 cross-entropy.

    The default is asserted on the signature rather than by running the loop, so the contract is
    pinned without a GPU.
    """
    import inspect

    from ce_monitor_patch_attack import run_confined_episode

    params = inspect.signature(run_confined_episode).parameters

    assert params["objective"].default == "ce"


def test_kappa_default_follows_forcing_loss() -> None:
    """One source of truth for the margin: a second literal here could drift from the sweep."""
    import inspect

    import forcing_loss as FL
    from ce_monitor_patch_attack import run_confined_episode

    params = inspect.signature(run_confined_episode).parameters

    assert params["kappa"].default == FL.DEFAULT_KAPPA


# --- GPU seam (skipped without PPIP_GPU_TESTS) ---------------------------------------


@requires_gpu
def test_hinge_episode_runs_end_to_end_and_records_its_objective(tmp_path: object) -> None:
    """Smoke: a short hinge episode completes and the result JSON names the loss that ran."""
    from hijack_backend import HijackBackend
    from ce_monitor_patch_attack import run_confined_episode

    backend = HijackBackend(run_dir=str(tmp_path), max_steps=3)
    backend.load_policy_once()

    result = run_confined_episode(
        backend,
        rect=PROBE_RECT,
        seed=0,
        max_steps=3,
        chunk=10,
        k=2,
        maxtries=1,
        run_dir=str(tmp_path),
        tag="obj_smoke",
        user_task=USER_TASK,
        objective="hinge",
        kappa=6.0,
    )

    assert result["objective"]["name"] == "hinge"
    assert result["objective"]["kappa"] == 6.0
    assert result["targeted"] in (True, False)


# --- the distortion penalty (design section 4.5) --------------------------------------
#
# `lambda * MSE(patch, carrier)` is the soft half of the stealth constraint: the eps-ball is a
# hard bound, so nothing ever pulls a drifted pixel back toward the logo. It is added INSIDE the
# ball, never instead of it -- a pure penalty reports lambda, which is not perceptually
# interpretable and gives a drifting effective distortion in a per-frame loop.


def test_distortion_weight_defaults_to_zero_so_every_prior_rung_is_unchanged() -> None:
    """Behaviour preservation: all six recorded ladder rungs ran with no distortion term."""
    import inspect

    from ce_monitor_patch_attack import run_confined_episode

    params = inspect.signature(run_confined_episode).parameters

    assert params["distortion_weight"].default == 0.0


def test_negative_distortion_weight_is_rejected_before_any_gpu_work() -> None:
    """A negative lambda would *reward* drifting away from the carrier -- an anti-stealth term."""
    from ce_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="distortion_weight"):
        run_confined_episode(
            object(),  # never used: validation raises first
            rect=PROBE_RECT,
            seed=0,
            run_dir="/tmp/distortion-unused",
            tag="distortion_negative",
            distortion_weight=-0.1,
        )


def test_distortion_weight_requires_a_carrier_to_measure_against() -> None:
    """Without `stealth_base` there is no logo to stay near, so the term has no meaning."""
    from ce_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="stealth_base"):
        run_confined_episode(
            object(),
            rect=PROBE_RECT,
            seed=0,
            run_dir="/tmp/distortion-unused",
            tag="distortion_free_range",
            distortion_weight=1.0,
        )
