"""Draw candidate corner-patch rectangles on a real 224x224 policy frame so we can VISUALLY
confirm a corner rectangle does NOT cover the graspable objects (alphabet_soup / salad_dressing).

Pure PIL, no GPU. Reads a saved clean_input frame and writes overlays with the rect outlined.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HOME = os.path.expanduser("~")
FRAME = os.environ.get(
    "CO_FRAME",
    os.path.join(HOME, "autoresearch/runs/monitor-patch/run2_rec/clean_input/f0000.png"),
)
OUT = os.environ.get("CO_OUT", os.path.join(HOME, "autoresearch/runs/monitor-corner/overlays"))


def corner_rect(corner: str, s: int) -> tuple[int, int, int, int]:
    """(r0,c0,h,w) square of side s anchored at a frame corner of the 224x224 image."""
    n = 224
    if corner == "TL":
        return (0, 0, s, s)
    if corner == "TR":
        return (0, n - s, s, s)
    if corner == "BL":
        return (n - s, 0, s, s)
    if corner == "BR":
        return (n - s, n - s, s, s)
    raise ValueError(corner)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    base = np.array(Image.open(FRAME).convert("RGB").resize((224, 224)))
    specs = sys.argv[1:] or ["TL:100", "TR:100", "BL:90", "BR:100"]
    for spec in specs:
        corner, s = spec.split(":")
        r0, c0, h, w = corner_rect(corner, int(s))
        im = Image.fromarray(base.copy()).resize((448, 448), Image.NEAREST)
        d = ImageDraw.Draw(im)
        # scale rect to the 2x preview
        d.rectangle([c0 * 2, r0 * 2, (c0 + w) * 2 - 1, (r0 + h) * 2 - 1],
                    outline=(255, 0, 0), width=3)
        path = os.path.join(OUT, f"overlay_{corner}_{s}.png")
        im.save(path)
        print(f"{spec:>8}  rect(r0,c0,h,w)=({r0},{c0},{h},{w})  -> {path}", flush=True)


if __name__ == "__main__":
    main()
