"""Compile recorded frames into an mp4 (+gif) for the presentation.

Usage:
  python make_video.py <frames_dir> <out.mp4> "Caption text" [--pair]
    <frames_dir> must contain scene/f####.png (and policy_input/f####.png if --pair).
    --pair renders the true scene (left) beside the attacker-perturbed policy input (right).
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = int(os.environ.get("VIDEO_FPS", "20"))
OUT_H = 384  # panel height


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


def main() -> None:
    import imageio.v2 as imageio

    frames_dir, out_mp4, caption = sys.argv[1], sys.argv[2], sys.argv[3]
    pair = "--pair" in sys.argv[4:]
    scene = sorted(glob.glob(os.path.join(frames_dir, "scene", "f*.png")))
    if not scene:
        raise SystemExit(f"no frames in {frames_dir}/scene")
    frames = []
    for sp in scene:
        left = _load(sp, OUT_H)
        if pair:
            pp = os.path.join(frames_dir, "policy_input", os.path.basename(sp))
            right = _load(pp, OUT_H) if os.path.exists(pp) else Image.new("RGB", (OUT_H, OUT_H), (0, 0, 0))
            gap = 8
            canvas = Image.new("RGB", (left.width + gap + right.width, OUT_H), (17, 17, 17))
            canvas.paste(left, (0, 0)); canvas.paste(right, (left.width + gap, 0))
            d = ImageDraw.Draw(canvas)
            try:
                f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            except Exception:  # noqa: BLE001
                f = ImageFont.load_default()
            d.text((8, 6), "room camera (reality)", fill=(120, 255, 120), font=f)
            d.text((left.width + gap + 8, 6), "robot's AI input (attacked)", fill=(255, 140, 140), font=f)
        else:
            canvas = left
        frames.append(np.asarray(_banner(canvas, caption)))
    # hold the last frame ~1.5s so the outcome is readable
    frames += [frames[-1]] * int(FPS * 1.5)
    imageio.mimsave(out_mp4, frames, fps=FPS, quality=8, macro_block_size=1)
    gif = out_mp4.rsplit(".", 1)[0] + ".gif"
    imageio.mimsave(gif, frames[:: max(1, FPS // 10)], duration=1.0 / 10, loop=0)
    print(f"[video] wrote {out_mp4} ({len(scene)} frames) and {gif}", flush=True)


if __name__ == "__main__":
    main()
