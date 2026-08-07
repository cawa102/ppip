"""The patch-evolution GIF -- what the perturbation actually looks like, over time.

Three panels, animating together over the rollout's control steps:

    carrier (static logo)  |  executed patch  |  (patch - carrier) x gain

This is the figure for both halves of the stealth claim at once. The middle panel is what a
person would see in the scene: at a small epsilon it stays a logo. The right panel amplifies
the perturbation around mid-grey so its structure -- and, crucially, how completely it is
**re-drawn every control step** -- becomes visible. That churn is the study's temporal-stealth
limitation, and it is far more legible as motion than as a number.

The amplification gain is printed on the figure. An amplified difference shown without its
gain reads as the real perturbation and overstates the attack.

Run:
  ~/vla-injection/.venv/bin/python experiments/patch_attack/make_patch_gif.py \
      <rec_dir> <carrier.png> <out.gif> [gain]
"""

from __future__ import annotations

import os
import sys
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rollout_gif as RG  # noqa: E402
import stealth_metrics as SM  # noqa: E402

TILE: Final = 240  # per-panel side, px
HEAD: Final = 70
FOOT: Final = 46
DEFAULT_GAIN: Final = 8


#: Deviation from mid-grey that a well-chosen gain should map the epsilon budget onto. Well
#: short of saturation, so structure inside the perturbation stays visible.
TARGET_DEVIATION: Final = 0.4


def gain_for_eps(eps: float, target: float = TARGET_DEVIATION) -> int:
    """Amplification that renders an epsilon-sized deviation at roughly `target` from mid-grey.

    A fixed gain cannot serve the whole ladder: x8 makes epsilon=0.06 legible but drives
    epsilon=0.25 far past clipping, and a saturated difference panel misreports the
    perturbation as uniformly maximal exactly where it is strongest.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return max(1, round(target / eps))


def amplify_difference(patch: np.ndarray, carrier: np.ndarray, gain: int) -> np.ndarray:
    """Render `(patch - carrier) * gain` centred on mid-grey, clipped to uint8."""
    if gain <= 0:
        raise ValueError(f"gain must be positive, got {gain}")
    p = patch.astype(np.float32) / 255.0 if patch.dtype == np.uint8 else patch.astype(np.float32)
    c = (
        carrier.astype(np.float32) / 255.0
        if carrier.dtype == np.uint8
        else carrier.astype(np.float32)
    )
    if p.shape != c.shape:
        raise ValueError(f"shape mismatch: patch {p.shape} vs carrier {c.shape}")
    # Clip, never wrap: a wrapped saturated positive deviation would render black and read as
    # a large negative one exactly where the attack pushed hardest.
    return (np.clip((p - c) * gain + 0.5, 0.0, 1.0) * 255).astype(np.uint8)


def build(
    rec_dir: str,
    carrier_path: str,
    out_path: str,
    gain: int = DEFAULT_GAIN,
    stride: int = 2,
) -> str:
    frames = SM.load_frames(os.path.join(rec_dir, "patch"))
    carrier = np.array(
        Image.open(carrier_path).convert("RGB").resize(frames.shape[1:3][::-1])
    )
    churn = SM.churn_stats(frames)

    width, height = TILE * 3, HEAD + TILE + FOOT
    title_font, sub_font, foot_font = RG.font(18), RG.font(13), RG.font(14)
    rendered: list[Image.Image] = []

    for step in RG.sampled_steps(frames.shape[0], stride):
        canvas = Image.new("RGB", (width, height), RG.BG)
        draw = ImageDraw.Draw(canvas)
        tiles = (
            ("CARRIER", "the plain logo, static", carrier),
            ("EXECUTED PATCH", "what the model consumed", frames[step]),
            (f"DIFFERENCE x{gain}", "amplified around mid-grey", None),
        )
        for i, (title, subtitle, image) in enumerate(tiles):
            array = amplify_difference(frames[step], carrier, gain) if image is None else image
            tile = Image.fromarray(array).convert("RGB").resize((TILE, TILE), Image.NEAREST)
            canvas.paste(tile, (i * TILE, HEAD))
            draw.text((i * TILE + 12, 12), title, font=title_font, fill=RG.FG)
            draw.text((i * TILE + 12, 36), subtitle, font=sub_font, fill=RG.DIM)
            if i:
                draw.line([(i * TILE, 0), (i * TILE, height)], fill=(52, 52, 56), width=1)

        lines = (
            f"control step {step:3d} of {frames.shape[0]}     "
            f"the middle panel is the stealth claim; the right panel is why it is SPATIAL only",
            f"churn: {churn.mean_abs_delta:.4f} mean |patch_t - patch_t-1|     "
            f"{churn.frac_steps_above_threshold:.1%} of steps change by more than 5/255     "
            f"the patch is re-optimised every step",
        )
        for row, line in enumerate(lines):
            draw.text(
                (12, height - 40 + row * 20),
                line,
                font=RG.font(RG.fitted_size(line, width - 24, foot_font.size)),
                fill=RG.FG if row == 0 else RG.DIM,
            )
        rendered.append(canvas)

    rendered.extend([rendered[-1]] * 12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    first, *rest = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in rendered]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=100,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: make_patch_gif.py <rec_dir> <carrier.png> <out.gif> [gain]")
    gain = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_GAIN
    path = build(sys.argv[1], sys.argv[2], sys.argv[3], gain)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
