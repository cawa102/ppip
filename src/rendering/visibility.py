"""Measure and visualise how much of the camera frame the injected prompt occupies.

The rollout backend renders a MuJoCo **segmentation** frame (per-pixel geom id) and
passes it here with the injected prompt geom's id:

  * `prompt_pixel_fraction` -> the objective "was the prompt in view?" signal, stored
    per rollout as `RolloutOutcome.prompt_visibility` and aggregated by the metrics.
  * `visibility_overlay` -> a highlighted frame for presentation figures.

Both are pure numpy; the segmentation render itself is handled by the GPU rollout
environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.integer[Any]]


def prompt_pixel_fraction(segmentation: IntArray, geom_id: int) -> float:
    """Fraction of `segmentation` pixels belonging to the prompt geom (0.0-1.0)."""
    if segmentation.size == 0:
        return 0.0
    return float(np.count_nonzero(segmentation == geom_id) / segmentation.size)


def visibility_overlay(
    frame: NDArray[np.uint8],
    segmentation: IntArray,
    geom_id: int,
    *,
    color: tuple[int, int, int] = (255, 0, 255),
    alpha: float = 0.5,
) -> NDArray[np.uint8]:
    """Return a copy of `frame` with the prompt's pixels blended toward `color`.

    Used to produce an at-a-glance "this is the injected instruction the policy sees"
    figure. Does not mutate `frame`.
    """
    out = frame.copy()
    mask = segmentation == geom_id
    tint = np.array(color, dtype=np.float32)
    blended = (1.0 - alpha) * out[mask].astype(np.float32) + alpha * tint
    out[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    return out
