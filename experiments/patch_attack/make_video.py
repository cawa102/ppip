"""Compile recorded frames into an mp4 (+gif) for the presentation.

Usage:
  python make_video.py <frames_dir> <out.mp4> "Caption text" [--pair] [--delta]
    <frames_dir> must contain scene/f####.png (and policy_input/f####.png if --pair,
    plus clean_input/f####.png if --delta).
    --pair  renders the true scene (left) beside the attacker-perturbed policy input (right).
    --delta adds a third panel: the per-frame perturbation delta = policy_input - clean_input,
            gray-centered and amplified (env DELTA_AMP, default 3x) so the noise is legible.

Env knobs:
  LABEL_L/LABEL_M/LABEL_R  per-panel captions.
  LEFT_SCENE_DIR           frames dir (containing f####.png) to use for the LEFT panel instead
                           of <frames_dir>/scene — e.g. a CLEAN baseline rollout, so the left
                           panel shows the *user's expected action* rather than the attacked
                           room camera. The two sequences are aligned by step index and the
                           shorter one holds its last frame, so both narratives finish.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = int(os.environ.get("VIDEO_FPS", "20"))
OUT_H = 384  # panel height
DELTA_AMP = float(os.environ.get("DELTA_AMP", "3.0"))  # noise-panel amplification (honest, labelled)
# Panel captions are overridable so the same layout can frame different comparisons
# (e.g. left = "room camera (reality)" or "user's expected action").
LABEL_L = os.environ.get("LABEL_L", "room camera (reality)")
LABEL_M = os.environ.get("LABEL_M", "robot's AI input (attacked)")
LABEL_R = os.environ.get("LABEL_R", f"attacker's added noise (delta x{DELTA_AMP:g})")
LEFT_SCENE_DIR = os.environ.get("LEFT_SCENE_DIR", "")  # optional independent left-panel rollout


def _load(p: str, h: int) -> Image.Image:
    im = Image.open(p).convert("RGB")
    return im.resize((int(im.width * h / im.height), h), Image.NEAREST)


def _banner(canvas: Image.Image, caption: str) -> Image.Image:
    w, h = canvas.width, 64
    bar = Image.new("RGB", (canvas.width, canvas.height + h), (17, 17, 17))
    bar.paste(canvas, (0, 0))
    d = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    d.text((14, canvas.height + 18), caption, fill=(240, 240, 240), font=font)
    return bar


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _delta_vis(clean_path: str, pert_path: str, h: int) -> Image.Image:
    """Gray-centred, amplified view of the actual per-frame perturbation (policy_input - clean)."""
    clean = np.asarray(Image.open(clean_path).convert("RGB")).astype(np.int16)
    pert = np.asarray(Image.open(pert_path).convert("RGB")).astype(np.int16)
    vis = np.clip(128.0 + DELTA_AMP * (pert - clean), 0, 255).astype(np.uint8)
    im = Image.fromarray(vis)
    return im.resize((int(im.width * h / im.height), h), Image.NEAREST)


def _compose(panels: list[tuple[Image.Image, str, tuple[int, int, int]]]) -> Image.Image:
    """Lay panels left->right on a dark strip, each with its coloured label."""
    gap = 8
    total_w = sum(p[0].width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, OUT_H), (17, 17, 17))
    d = ImageDraw.Draw(canvas)
    f = _font(15)
    x = 0
    for im, label, color in panels:
        canvas.paste(im, (x, 0))
        if label:
            # dark backing strip so the label stays legible over busy noise panels
            tb = d.textbbox((x + 8, 6), label, font=f)
            d.rectangle((tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2), fill=(0, 0, 0))
            d.text((x + 8, 6), label, fill=color, font=f)
        x += im.width + gap
    return canvas


def main() -> None:
    import imageio.v2 as imageio

    frames_dir, out_mp4, caption = sys.argv[1], sys.argv[2], sys.argv[3]
    opts = sys.argv[4:]
    delta = "--delta" in opts
    pair = "--pair" in opts or delta  # the delta panel needs the perturbed policy input too
    scene = sorted(glob.glob(os.path.join(frames_dir, "scene", "f*.png")))
    if not scene:
        raise SystemExit(f"no frames in {frames_dir}/scene")
    # The left panel may come from a different rollout (e.g. the clean/expected one). Both
    # sequences start from the same init state, so align by index and hold the shorter one's
    # last frame — neither story gets cut off.
    left = sorted(glob.glob(os.path.join(LEFT_SCENE_DIR, "f*.png"))) if LEFT_SCENE_DIR else scene
    if not left:
        raise SystemExit(f"no frames in LEFT_SCENE_DIR={LEFT_SCENE_DIR}")
    n_out = max(len(scene), len(left))
    frames = []
    for i in range(n_out):
        sp = scene[min(i, len(scene) - 1)]
        lp = left[min(i, len(left) - 1)]
        bn = os.path.basename(sp)
        panels: list[tuple[Image.Image, str, tuple[int, int, int]]] = [
            (_load(lp, OUT_H), LABEL_L, (120, 255, 120)),
        ]
        if pair:
            pp = os.path.join(frames_dir, "policy_input", bn)
            blank = Image.new("RGB", (OUT_H, OUT_H), (0, 0, 0))
            right = _load(pp, OUT_H) if os.path.exists(pp) else blank
            panels.append((right, LABEL_M, (255, 140, 140)))
        if delta:
            cp = os.path.join(frames_dir, "clean_input", bn)
            pp = os.path.join(frames_dir, "policy_input", bn)
            dv = (
                _delta_vis(cp, pp, OUT_H)
                if os.path.exists(cp) and os.path.exists(pp)
                else Image.new("RGB", (OUT_H, OUT_H), (128, 128, 128))
            )
            panels.append((dv, LABEL_R, (255, 210, 120)))
        canvas = panels[0][0] if len(panels) == 1 else _compose(panels)
        frames.append(np.asarray(_banner(canvas, caption)))
    # hold the last frame ~1.5s so the outcome is readable
    frames += [frames[-1]] * int(FPS * 1.5)
    imageio.mimsave(out_mp4, frames, fps=FPS, quality=8, macro_block_size=1)
    gif = out_mp4.rsplit(".", 1)[0] + ".gif"
    imageio.mimsave(gif, frames[:: max(1, FPS // 10)], duration=1.0 / 10, loop=0)
    print(f"[video] wrote {out_mp4} ({n_out} frames) and {gif}", flush=True)


if __name__ == "__main__":
    main()
