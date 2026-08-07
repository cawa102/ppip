"""The stealth patch parameterization — one static patch, provably inside an L-infinity ball.

This is the single line the whole stealth axis turns on:

    patch = clamp(base + eps * tanh(raw), 0, 1)

`tanh` bounds the perturbation to `(-eps, eps)` **by construction**, so no projection step,
no clipping schedule, and no way for the optimizer to drift outside the budget it claims.
`raw = 0` gives exactly the base image, which makes the epsilon-zero control (the pure logo)
the literal initialisation rather than a separate code path.

One ladder spans the whole experiment: `eps = 0` is the pure-logo control, `eps ~ 0.02-0.32`
is the stealth curve, and `eps = 1` is effectively free-range — the capacity ceiling that
tells us whether a *static* patch can hijack at all before any stealth budget is spent.

`total_variation` is applied to the **perturbation**, not the patch. Penalising the patch
would fight the logo's own edges (a logo is mostly edges); penalising delta pushes it toward
smooth shading rather than visible noise at the same epsilon.

Everything here is pure torch and CPU-testable — no model, no simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FRAME_SIDE = 224


@dataclass(frozen=True)
class ConfinedStealth:
    """A validated `(base, eps)` pair for the per-frame confined attack.

    `base` is a FULL-FRAME `[1,3,224,224]` image because that is the shape the confined loop
    composites — only the pixels under the rect mask survive, so what sits outside is
    irrelevant and is never read.
    """

    base: torch.Tensor
    eps: float


def resolve_confined_stealth(
    base: torch.Tensor | None, eps: float | None
) -> ConfinedStealth | None:
    """Validate the additive stealth kwargs; `None` means keep the free-range path.

    Returning `None` for the both-absent case is what makes the kwargs additive: every
    existing caller of `run_confined_episode` passes neither and must get `sigmoid(raw)`
    back, bit-identically. Half-specified pairs are rejected rather than defaulted, because
    silently choosing an epsilon would let a run claim a stealth budget nobody set.
    """
    if base is None and eps is None:
        return None
    if base is None or eps is None:
        raise ValueError(
            "stealth_base and stealth_eps must both be set, or both be absent (got "
            f"base={'set' if base is not None else 'None'}, eps={eps!r})"
        )
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"stealth eps must lie in [0, 1], got {eps}")
    if tuple(base.shape) != (1, 3, FRAME_SIDE, FRAME_SIDE):
        raise ValueError(
            f"stealth_base shape {tuple(base.shape)} is not the full frame "
            f"(1, 3, {FRAME_SIDE}, {FRAME_SIDE})"
        )
    return ConfinedStealth(base=base, eps=eps)


def stealth_patch(raw: torch.Tensor, base: torch.Tensor, eps: float) -> torch.Tensor:
    """`clamp(base + eps*tanh(raw), 0, 1)`, shape `[1,3,h,w]`, values in [0,1].

    The clamp can only ever *shrink* the perturbation (where base is near 0 or 1), so the
    L-infinity bound holds regardless of `raw`.
    """
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    return (base + eps * torch.tanh(raw)).clamp(0.0, 1.0)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute finite difference — a scalar measure of high-frequency content."""
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return x.sum() * 0.0
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def distortion(
    patch: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float | None = None,
) -> torch.Tensor:
    """Mean squared deviation from the carrier — the *soft* half of the stealth constraint.

    The epsilon ball is a hard bound: inside it every point is equally free, so nothing ever
    pulls a pixel back toward the carrier once it has drifted. This term does that pulling. It
    is the distortion term of C&W-L2, which this project's parameterization otherwise drops in
    favour of the box constraint (`clamp(base + eps*tanh(raw))`).

    It is **added inside the ball, never instead of it** (design 2026-08-04, section 4.5). A
    pure penalty would report `lambda`, which is not perceptually interpretable, not comparable
    across carriers, and — in a per-frame loop re-solving on a changing frame — yields a
    different effective distortion every step, which is no threshold at all.

    `mask` restricts the average to the pixels the attacker actually placed; outside it the
    patch is never shown, so including those would make the penalty depend on frame size.

    `eps` normalises by the budget, returning the mean **squared ball occupancy** — the same
    quantity `stealth_metrics.ball_occupancy` reports, squared. Raw MSE scales with `eps**2`,
    so one `lambda` would mean different things at different rungs; normalised, it does not.
    """
    squared = (patch - base) ** 2
    if eps is not None:
        if eps <= 0.0:
            raise ValueError(f"eps must be positive to normalise a distortion, got {eps}")
        squared = squared / (eps * eps)
    if mask is None:
        return squared.mean()
    # Divide by the mask's own weight, not by the frame's pixel count: the latter would shrink
    # the penalty as the rect shrinks, silently retuning lambda with the patch size.
    weight = mask.expand_as(squared).sum()
    return (squared * mask).sum() / weight.clamp(min=1.0)


def composite(
    frame: torch.Tensor, patch: torch.Tensor, rect: tuple[int, int, int, int]
) -> torch.Tensor:
    """Replace `rect` of `frame` `[B,3,224,224]` with `patch` `[1,3,h,w]`; autograd-safe.

    Writes into a clone rather than the input, so `frame` is never mutated and gradients
    still flow back to `patch` through the slice.
    """
    r0, c0, height, width = rect
    if patch.shape[-2:] != (height, width):
        raise ValueError(f"patch {tuple(patch.shape[-2:])} does not fill rect {(height, width)}")
    if r0 < 0 or c0 < 0 or r0 + height > FRAME_SIDE or c0 + width > FRAME_SIDE:
        raise ValueError(f"rect {rect} falls outside the {FRAME_SIDE}x{FRAME_SIDE} frame")
    out = frame.clone()
    out[:, :, r0 : r0 + height, c0 : c0 + width] = patch
    return out


def rects_to_mask(
    rects: list[tuple[int, int, int, int]], side: int = FRAME_SIDE, device: str = "cpu"
) -> torch.Tensor:
    """`[1,1,side,side]` binary mask covering every rect — the general patch region.

    A single rect caps out around 28% of this scene's frame before it starts covering task
    objects. Measuring the *area* axis past that point needs a disjoint region (a band above
    and below the objects), which a rect cannot express but a mask can.
    """
    mask = torch.zeros(1, 1, side, side, device=device)
    for r0, c0, height, width in rects:
        if r0 < 0 or c0 < 0 or r0 + height > side or c0 + width > side:
            raise ValueError(f"region {(r0, c0, height, width)} falls outside the frame")
        mask[:, :, r0 : r0 + height, c0 : c0 + width] = 1.0
    return mask


def composite_masked(
    frame: torch.Tensor, patch: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Replace `frame` with `patch` wherever `mask` is 1; autograd-safe, no mutation."""
    return frame * (1.0 - mask) + patch * mask


def l_inf(patch: torch.Tensor, base: torch.Tensor) -> float:
    """Measured `|patch - base|_inf`. The published stealth claim, recomputed not trusted."""
    return float((patch - base).abs().max().item())


def headroom_fraction(base: torch.Tensor, eps: float) -> float:
    """Share of the epsilon ball still reachable after the `[0,1]` clamp, in `[0,1]`.

    The clamp bites wherever `base` is near 0 or 1, and it zeroes the gradient there. So two
    carriers at the *same* eps do not get the same budget: mid-grey has the full 2*eps range
    everywhere, while a logo with a near-white foreground (`solstice`'s `#FDF6E3`) has almost
    no upward room on those pixels.

    That makes carrier comparisons — logo vs `scrambled:` vs `flat:` vs `gray` — confounded by
    available headroom, not only by structure. Reporting this alongside each run keeps the
    confound visible instead of silently biasing the base sweep.
    """
    if eps <= 0.0:
        return 0.0
    reachable = (base + eps).clamp(max=1.0) - (base - eps).clamp(min=0.0)
    return float((reachable / (2.0 * eps)).mean().item())


def to_hwc(patch: torch.Tensor) -> torch.Tensor:
    """`[1,3,h,w]` -> `[h,w,3]`, the layout the evaluator's `set_patch` and .npy artifacts use."""
    return patch.detach()[0].permute(1, 2, 0).cpu()


def from_hwc(patch_hwc: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """`[h,w,3]` -> `[1,3,h,w]`, the layout the optimizer works in."""
    return patch_hwc.permute(2, 0, 1)[None].to(device)
