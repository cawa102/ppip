"""Stealth measurement for an executed per-frame patch sequence.

The epsilon ladder needs two numbers per rung that the rollout itself does not produce:

**LPIPS vs the carrier** -- how far the executed patch drifts from the plain logo, perceptually
rather than in raw pixels. This is the *spatial* stealth claim.

**Churn** -- mean absolute step-to-step change in the patch. The per-frame attack re-optimises
every step, so a patch that is near-invisible in any single frame can still shimmer at 20 Hz.
Churn is the honest bound on what the study can claim: spatial stealth only. It is reported as
a measured limitation, not hidden.

Everything is computed from the recorded ``patch/f*.png`` crops, which are the uint8 images
OpenVLA actually consumed. There is no re-rendering step, so these describe the executed
attack. The uint8 quantisation is likewise the model's own, not an artefact of measurement --
except when checking the epsilon bound, where rounding can push the measured Linf a fraction of
a level past a float bound that in fact held (see :data:`QUANT_TOLERANCE`).

Usage:
  ~/vla-injection/.venv/bin/python experiments/patch_attack/stealth_metrics.py \
      <rec_dir>/patch <carrier.png> [out.json]
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# uint8 rounding can move any pixel by up to half a level in each of two quantised images.
QUANT_TOLERANCE: Final = 1.0 / 255.0

# Below this a step-to-step change is invisible in an 8-bit display.
VISIBLE_DELTA: Final = 5.0 / 255.0


@dataclass(frozen=True)
class ChurnStats:
    """How much the patch is re-randomised from one control step to the next."""

    mean_abs_delta: float
    max_abs_delta: float
    frac_steps_above_threshold: float
    n_transitions: int


def _as_unit_float(frames: NDArray[Any]) -> NDArray[Any]:
    if frames.size == 0 or frames.shape[0] == 0:
        raise ValueError("empty frame sequence -- nothing to measure")
    if frames.dtype == np.uint8:
        scaled: NDArray[Any] = frames.astype(np.float32) / 255.0
        return scaled
    return frames.astype(np.float32)


def churn_stats(frames: NDArray[Any], threshold: float = VISIBLE_DELTA) -> ChurnStats:
    """Temporal stability of the patch sequence, shape ``(T, H, W, C)``."""
    seq = _as_unit_float(frames)
    if seq.shape[0] < 2:
        return ChurnStats(0.0, 0.0, 0.0, 0)

    deltas = np.abs(np.diff(seq, axis=0))
    per_step = deltas.reshape(deltas.shape[0], -1).mean(axis=1)
    return ChurnStats(
        mean_abs_delta=float(per_step.mean()),
        max_abs_delta=float(deltas.max()),
        frac_steps_above_threshold=float((per_step > threshold).mean()),
        n_transitions=int(deltas.shape[0]),
    )


def linf_vs_carrier(frames: NDArray[Any], carrier: NDArray[Any]) -> float:
    """Largest per-pixel deviation from the carrier anywhere in the sequence."""
    return float(_abs_delta(frames, carrier).max())


def _abs_delta(frames: NDArray[Any], carrier: NDArray[Any]) -> NDArray[Any]:
    """`|frame - carrier|` over the sequence, with the shape check both callers need."""
    seq = _as_unit_float(frames)
    base = _as_unit_float(carrier[None])[0]
    if base.shape != seq.shape[1:]:
        raise ValueError(f"carrier shape {base.shape} != frame shape {seq.shape[1:]}")
    delta: NDArray[Any] = np.abs(seq - base[None])
    return delta


@dataclass(frozen=True)
class BallOccupancy:
    """How much of the granted epsilon budget the executed patch actually spent.

    The ladder's axis is the budget *granted*; this is the budget *spent*, and the two diverge
    Measured over the finished ladder (2026-08-07): the *typical* pixel spends only 39-58% of the
    budget even though L-inf reaches the cap at every rung, and occupancy is NON-MONOTONE --
    pinned pixels peak at 35.3% (eps=0.06) and decay to 10.2% (eps=0.25). The hijack threshold
    sits on that peak, i.e. the outcome flips where the ball stops binding.

    The tail fractions are reported beside the mean because they describe different patches: a
    patch with half its pixels pinned at the boundary and half untouched has the same mean as
    one sitting uniformly at half budget, and only the first has no room left to give back.
    """

    eps: float
    mean_ratio: float
    median_ratio: float
    frac_above_90pct: float
    frac_above_50pct: float
    frac_below_10pct: float
    n_frames: int


def ball_occupancy(frames: NDArray[Any], carrier: NDArray[Any], eps: float) -> BallOccupancy:
    """`|delta| / eps` statistics over the sequence, shape ``(T, H, W, C)``.

    `eps <= 0` is rejected rather than divided by: the eps=0 pure-logo control has no ball to
    occupy, and its own check is `linf_vs_carrier == 0` -- which is simultaneously the proof
    that the carrier PNG matches the executed base bit-for-bit.
    """
    if eps <= 0.0:
        raise ValueError(
            f"eps must be positive to measure occupancy, got {eps}; for the eps=0 control "
            "use linf_vs_carrier, which must read exactly 0"
        )
    ratio = _abs_delta(frames, carrier) / eps
    return BallOccupancy(
        eps=float(eps),
        mean_ratio=float(ratio.mean()),
        median_ratio=float(np.median(ratio)),
        frac_above_90pct=float((ratio > 0.9).mean()),
        frac_above_50pct=float((ratio > 0.5).mean()),
        frac_below_10pct=float((ratio < 0.1).mean()),
        n_frames=int(np.asarray(frames).shape[0]),
    )


def bound_holds(measured: float, eps: float) -> bool:
    """Whether an executed Linf respects its budget, allowing for uint8 rounding."""
    return measured <= eps + QUANT_TOLERANCE


def load_frames(directory: str) -> NDArray[Any]:
    from PIL import Image

    files = sorted(f for f in os.listdir(directory) if f.endswith(".png"))
    if not files:
        raise SystemExit(f"no patch frames in {directory}")
    return np.stack(
        [np.array(Image.open(os.path.join(directory, f)).convert("RGB")) for f in files]
    )


def lpips_vs_carrier(frames: NDArray[Any], carrier: NDArray[Any]) -> dict[str, float]:
    """Mean/max perceptual distance from the carrier, over the sequence.

    Runs on CPU: the GPU is occupied by the rollout this is measuring, and a few hundred
    64x64 crops are cheap enough that contending for it would be the slower choice.
    """
    import lpips as lpips_lib  # type: ignore[import-not-found]
    import torch

    net = lpips_lib.LPIPS(net="alex", verbose=False)
    seq = torch.from_numpy(_as_unit_float(frames)).permute(0, 3, 1, 2) * 2 - 1
    base = torch.from_numpy(_as_unit_float(carrier[None])).permute(0, 3, 1, 2) * 2 - 1

    with torch.no_grad():
        scores = [float(net(seq[i : i + 1], base)) for i in range(seq.shape[0])]
    return {"lpips_mean": float(np.mean(scores)), "lpips_max": float(np.max(scores))}


def measure(rec_dir: str, carrier_path: str, eps: float | None = None) -> dict[str, Any]:
    """All per-rung stealth numbers, from one recording directory.

    Two perceptual distances are reported because they answer different questions and are not
    interchangeable:

    ``patch_vs_carrier`` -- does the patch still look like the logo? This is the stealth claim.

    ``frame_vs_clean`` -- does the scene the model sees differ from the unattacked scene? This
    is larger (LPIPS is not area-normalised, so confining the patch to 8% of the frame does not
    scale the distance down by 8%), and it is the number to quote for whole-image detectability.

    ``eps`` adds the ``ball_occupancy`` block -- the budget *spent* against the budget
    *granted*. Optional rather than required so the eps=0 control and any pre-ladder recording
    still measure, but every ladder rung should pass it: design section 5 requires granted and
    spent to be reported together, and ``finalize_rung`` already knows the rung's budget.
    """
    from PIL import Image

    frames = load_frames(os.path.join(rec_dir, "patch"))
    size = (int(frames.shape[2]), int(frames.shape[1]))  # PIL wants (width, height)
    carrier = np.array(Image.open(carrier_path).convert("RGB").resize(size))

    stats: dict[str, Any] = {
        "rec_dir": rec_dir,
        "carrier": carrier_path,
        "n_frames": int(frames.shape[0]),
        "churn": asdict(churn_stats(frames)),
        "linf_vs_carrier": linf_vs_carrier(frames, carrier),
        "patch_vs_carrier": lpips_vs_carrier(frames, carrier),
    }
    if eps is not None and eps > 0.0:
        stats["ball_occupancy"] = asdict(ball_occupancy(frames, carrier, eps))

    clean_dir = os.path.join(rec_dir, "clean_input")
    policy_dir = os.path.join(rec_dir, "policy_input")
    if os.path.isdir(clean_dir) and os.path.isdir(policy_dir):
        stats["frame_vs_clean"] = lpips_pairwise(load_frames(policy_dir), load_frames(clean_dir))
    return stats


def lpips_pairwise(a: NDArray[Any], b: NDArray[Any]) -> dict[str, float]:
    """Mean/max perceptual distance between two aligned sequences, step by step."""
    import lpips as lpips_lib
    import torch

    n = min(a.shape[0], b.shape[0])
    net = lpips_lib.LPIPS(net="alex", verbose=False)
    xa = torch.from_numpy(_as_unit_float(a[:n])).permute(0, 3, 1, 2) * 2 - 1
    xb = torch.from_numpy(_as_unit_float(b[:n])).permute(0, 3, 1, 2) * 2 - 1

    with torch.no_grad():
        scores = [float(net(xa[i : i + 1], xb[i : i + 1])) for i in range(n)]
    return {"lpips_mean": float(np.mean(scores)), "lpips_max": float(np.max(scores))}


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--eps")]
    eps = next(
        (float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--eps=")), None
    )
    if len(argv) < 2:
        raise SystemExit(
            "usage: stealth_metrics.py <rec_dir> <carrier.png> [out.json] [--eps=E]\n"
            "  --eps adds the ball-occupancy block (budget spent vs budget granted)"
        )
    stats = measure(argv[0], argv[1], eps=eps)
    print(json.dumps(stats, indent=2))
    if len(argv) > 2:
        with open(argv[2], "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"wrote {argv[2]}")


if __name__ == "__main__":
    main()
