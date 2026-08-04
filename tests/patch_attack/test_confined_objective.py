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
