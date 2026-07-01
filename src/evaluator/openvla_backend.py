"""The real OpenVLA+LIBERO rollout backend (GPU-only body).

This is the concrete `RolloutBackend` the harness uses on a GPU host. Only the
CPU-observable contract is exercised in tests here; the closed-loop rollout body
is implemented and verified on the GPU machine, following the reference harness.

Defaults are grounded in the reference OpenVLA+LIBERO project:
  * model_id        openvla/openvla-7b-finetuned-libero-spatial
  * unnorm_key      libero_spatial   (action de-normalisation stats key)
  * max_steps       220              (libero_spatial episode cap)
  * num_steps_wait  10               (no-op settle steps at episode start)

Intended rollout body (per candidate, for each seed x rollout):
  1. Load model/processor once (AutoModelForVision2Seq + AutoProcessor, bf16).
  2. Build the LIBERO env for the task suite; render the candidate's visual prompt
     into the scene (src/rendering) at its placement/style.
  3. Run the closed loop like the reference `run_episode`: get_libero_image ->
     get_action -> normalize/invert gripper -> env.step, until `done` or max_steps.
  4. commanded_success = env `done` on the user task; targeted_success = the
     attacker target task's success check over the same rollout.
  5. Return one RolloutOutcome per episode.
"""

from __future__ import annotations

from typing import Any

from evaluator.metrics import RolloutOutcome

_DEFAULT_MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
_DEFAULT_UNNORM_KEY = "libero_spatial"
_DEFAULT_MAX_STEPS = 220
_DEFAULT_NUM_STEPS_WAIT = 10


class OpenVLABackendUnavailable(RuntimeError):
    """Raised when the OpenVLA/LIBERO/torch stack is not importable on this host."""


def _require_openvla_stack() -> Any:
    """Import the GPU stack lazily; raise an actionable error if it is absent."""
    try:
        import torch  # noqa: F401  (presence check only)
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only where deps exist
        raise OpenVLABackendUnavailable(
            "The OpenVLA rollout backend needs the GPU stack (torch, transformers, "
            "openvla, LIBERO). Install the 'gpu' extra on a CUDA host and run there; "
            f"import failed with: {exc}"
        ) from exc
    return torch


class OpenVLARolloutBackend:
    """Runs real OpenVLA rollouts for a candidate on a GPU host."""

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        unnorm_key: str = _DEFAULT_UNNORM_KEY,
        task_suite: str = "libero_spatial",
        max_steps: int = _DEFAULT_MAX_STEPS,
        num_steps_wait: int = _DEFAULT_NUM_STEPS_WAIT,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.unnorm_key = unnorm_key
        self.task_suite = task_suite
        self.max_steps = max_steps
        self.num_steps_wait = num_steps_wait
        self.device = device

    def run_rollouts(
        self,
        *,
        candidate: dict[str, Any],
        seeds: list[int],
        rollouts_per_candidate: int,
    ) -> list[RolloutOutcome]:
        """Run rollouts for a candidate (GPU-only; see module docstring)."""
        _require_openvla_stack()
        # The GPU host implements the closed loop described in the module docstring
        # and returns one RolloutOutcome per episode.
        raise NotImplementedError(  # pragma: no cover - implemented on the GPU host
            "OpenVLARolloutBackend.run_rollouts must be implemented on the GPU host; "
            "see the module docstring for the reference rollout procedure."
        )
