"""The rollout backend boundary — the single seam between the fixed evaluator
and the (GPU-only) OpenVLA+LIBERO machinery.

The evaluator orchestration depends only on this Protocol, so it can be fully
tested on CPU with a fake backend. The real `OpenVLARolloutBackend` (which loads
the model, renders the visual prompt into the LIBERO scene, and runs the closed
loop) lives behind the same interface and is wired in only on the GPU host.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from evaluator.metrics import RolloutOutcome


@runtime_checkable
class RolloutBackend(Protocol):
    """Runs the configured rollouts for one candidate and returns per-episode outcomes."""

    def run_rollouts(
        self,
        *,
        candidate: dict[str, Any],
        seeds: list[int],
        rollouts_per_candidate: int,
    ) -> list[RolloutOutcome]:
        """Return one RolloutOutcome per episode (len == len(seeds) * rollouts_per_candidate)."""
        ...
