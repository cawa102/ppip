"""The `hinge_monitor_patch_attack` entry point — a saturating-objective-only front door.

`ce_monitor_patch_attack` holds the shared per-frame core and defaults to all-7 cross-entropy,
the loss every published corner result (`runs/monitor-corner/`) was produced with. This sibling
module exists so a run's loss family is attributable from the module it was launched from rather
than from an `objective=` kwarg that a sweep script can shadow — so the two things worth testing
are that it *defaults* to the hinge and that it *refuses* to run a cross-entropy episode.

Everything here is CPU-reachable: the guard runs before any policy load, and delegation is
checked against a fake core, so no GPU seam is needed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

PROBE_RECT = (160, 0, 64, 64)  # the measured-clear BL 64x64 corner
BASE_KWARGS: dict[str, Any] = {
    "rect": PROBE_RECT,
    "seed": 0,
    "run_dir": "/tmp/hinge-entry-unused",
    "tag": "hinge_entry",
}


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the delegated core with a recorder; return the dict it writes into."""
    import hinge_monitor_patch_attack as H

    seen: dict[str, Any] = {}

    def fake(backend: Any, **kwargs: Any) -> dict[str, Any]:
        seen["backend"] = backend
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(H, "_run_confined_episode", fake)
    return seen


def test_default_objective_is_the_margin_hinge(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    from hinge_monitor_patch_attack import run_confined_episode

    seen = _capture(monkeypatch)

    # Act
    run_confined_episode(object(), **BASE_KWARGS)

    # Assert
    assert seen["objective"] == "hinge"


def test_default_kappa_tracks_forcing_loss_and_is_not_the_capped_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kappa=3.0 silently capped every hinge run at 2-of-3 decisive dims before 2026-08-04.

    Pinning the delegated value to `forcing_loss.DEFAULT_KAPPA` (not a literal) is what stops
    that regression from being reintroduced here, so assert both the link and the value.
    """
    # Arrange
    import forcing_loss as FL
    from hinge_monitor_patch_attack import run_confined_episode

    seen = _capture(monkeypatch)

    # Act
    run_confined_episode(object(), **BASE_KWARGS)

    # Assert
    assert seen["kappa"] == FL.DEFAULT_KAPPA
    assert FL.DEFAULT_KAPPA == 6.0


def test_directional_is_accepted_as_the_other_saturating_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from hinge_monitor_patch_attack import run_confined_episode

    seen = _capture(monkeypatch)

    # Act
    run_confined_episode(object(), objective="directional", **BASE_KWARGS)

    # Assert
    assert seen["objective"] == "directional"


def test_remaining_kwargs_are_forwarded_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper must not shadow or reorder the core's contract — only the objective default."""
    # Arrange
    from hinge_monitor_patch_attack import run_confined_episode

    seen = _capture(monkeypatch)
    backend = object()

    # Act
    run_confined_episode(backend, lr=0.05, decisive_boost=3, anchor=0.1, **BASE_KWARGS)

    # Assert
    assert seen["backend"] is backend
    assert seen["lr"] == 0.05
    assert seen["decisive_boost"] == 3
    assert seen["anchor"] == 0.1
    assert seen["rect"] == PROBE_RECT


@pytest.mark.parametrize("objective", ["ce", "ce_decisive"])
def test_cross_entropy_is_refused_and_names_the_ce_module(objective: str) -> None:
    """The filename is the guarantee: a CE run must not be launchable from the hinge module."""
    from hinge_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="ce_monitor_patch_attack"):
        run_confined_episode(object(), objective=objective, **BASE_KWARGS)


def test_typo_is_rejected_before_any_gpu_work() -> None:
    """The dummy backend proves the raise precedes any policy load or env build."""
    from hinge_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="not saturating"):
        run_confined_episode(object(), objective="hinj", **BASE_KWARGS)
