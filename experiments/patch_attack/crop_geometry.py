"""What the policy *actually* sees of a patch, after the 0.9-area centre crop.

`get_vla_action` crops the central **90% of the frame's area** before the ViT, and
`vla_diff.preprocess` reproduces that exactly. With `side = sqrt(0.9)`, the resampler only
ever reads pixels in **[5.72, 217.28]** of [0, 223] — a ~5.7 px border on every edge is
discarded before the model can be influenced by it.

`occlusion_probe.corner_rect` puts every corner patch **flush to the frame edge**, so two of
its four sides sit in that dead border. The cost is not marginal, and it grows as the patch
shrinks (the border is a fixed toll paid on two sides):

    BL 64x64   82.9% of the patch reaches the model
    BL 48x48   77.5%
    BL 40x40   73.4%
    BL 32x32   67.3%          <- a third of the patch never reaches the ViT

Translating the same rect inside the window recovers all of it at **zero cost in nominal
area** (+20.6% effective area at 64x64, +48.5% at 32x32) — the patch is not made bigger, it
is moved off the dead border.

Hence two different area numbers, and this module keeps them apart:

  * `nominal_area_fraction` — the region's share of the 224x224 **frame**. This is what a
    human or a defender sees, and what "a 3.2% patch" in a published sweep means. Unchanged.
  * `input_area_fraction` — its share of the 224x224 **tensor the ViT consumes**, measured by
    pushing the region's indicator through the real `center_crop_resize`. This is the number
    that bounds capacity, because it is proportional to the visual tokens the patch controls.

Deriving the second from `vla_diff` rather than re-deriving the geometry is deliberate: if the
preprocessing ever changes, these numbers change with it instead of quietly going stale.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import vla_diff

#: Side of the policy's square observation, in pixels.
POLICY_SIDE: Final[int] = 224

#: `get_vla_action` crops this fraction of the frame's **area** (not its side).
CROP_AREA: Final[float] = 0.9

#: A rect and a mask are the two ways this repo names a patch region.
Rect = tuple[int, int, int, int]


def crop_window(side: float = vla_diff._CROP_SIDE, n: int = POLICY_SIDE) -> tuple[float, float]:
    """First and last pixel coordinate the crop's resampler reads, in the `n`-wide frame.

    Mirrors `vla_diff.center_crop_resize`: `grid_sample(align_corners=True)` maps [-1, 1] onto
    [0, n-1], and the grid spans [-side, side].
    """
    return (1.0 - side) / 2.0 * (n - 1), (1.0 + side) / 2.0 * (n - 1)


def crop_safe_bounds(side: float = vla_diff._CROP_SIDE, n: int = POLICY_SIDE) -> tuple[int, int]:
    """Smallest and largest integer pixel index (inclusive) lying inside the sampled window."""
    lo, hi = crop_window(side, n)
    return math.ceil(lo), math.floor(hi)


def rect_mask(rect: Rect, n: int = POLICY_SIDE) -> torch.Tensor:
    """`[1,1,n,n]` indicator of `rect`, the common currency for the area measures below."""
    r0, c0, height, width = rect
    if r0 < 0 or c0 < 0 or r0 + height > n or c0 + width > n:
        raise ValueError(f"rect {rect} falls outside the {n}x{n} frame")
    mask = torch.zeros(1, 1, n, n)
    mask[:, :, r0 : r0 + height, c0 : c0 + width] = 1.0
    return mask


def _as_mask(region: Rect | torch.Tensor, n: int = POLICY_SIDE) -> torch.Tensor:
    return region if isinstance(region, torch.Tensor) else rect_mask(region, n)


def nominal_area_fraction(region: Rect | torch.Tensor, n: int = POLICY_SIDE) -> float:
    """Share of the **frame** the region covers — what a human sees, and what sweeps report."""
    return float(_as_mask(region, n).mean().item())


def input_area_fraction(region: Rect | torch.Tensor, n: int = POLICY_SIDE) -> float:
    """Share of the **model's input tensor** the region covers, after the real centre crop.

    Measured by resampling the region's indicator through `vla_diff.center_crop_resize`, so
    partially-cropped edge pixels are counted by the bilinear weight they actually receive
    rather than being rounded in or out.
    """
    return float(vla_diff.center_crop_resize(_as_mask(region, n)).mean().item())


def retained_fraction(region: Rect | torch.Tensor, n: int = POLICY_SIDE) -> float:
    """Fraction of the region that survives the crop; 1.0 iff it lies wholly inside the window.

    A region fully inside is *magnified* to `nominal / CROP_AREA` of the input, so that ratio
    — not the nominal area — is the ceiling this is measured against.
    """
    nominal = nominal_area_fraction(region, n)
    if nominal == 0.0:
        return 0.0
    return input_area_fraction(region, n) / (nominal / CROP_AREA)


def shift_rect_into_crop(rect: Rect, n: int = POLICY_SIDE) -> Rect:
    """Translate `rect` — same size — so it lies entirely inside the sampled window.

    Moves by the minimum needed and leaves an already-inside rect untouched. Raises rather
    than shrinking, because silently returning a smaller patch would corrupt the area axis.

    Note this only shifts; the caller still owns non-occlusion. Moving a bottom-flush corner
    up by ~6 px walks it toward the objects' rows, so re-check `occlusion_probe` before
    adopting a shifted rect as an attack region.
    """
    lo, hi = crop_safe_bounds(n=n)
    r0, c0, height, width = rect
    span = hi - lo + 1
    if height > span or width > span:
        raise ValueError(
            f"rect {rect} cannot fit inside the crop window [{lo}, {hi}] ({span}px)"
        )
    return min(max(r0, lo), hi - height + 1), min(max(c0, lo), hi - width + 1), height, width
