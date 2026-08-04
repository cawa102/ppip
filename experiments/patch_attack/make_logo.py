"""Deterministic brand-like base images for the stealth corner patch (Exp C).

The stealth attack is a bounded perturbation *around a fixed base image*:

    patch01 = (base01 + eps * tanh(raw)).clamp(0, 1)      # |patch - base|_inf <= eps

so the base is what a human sees and `eps` is the entire stealth budget. This module builds
those bases -- and, just as importantly, the **control bases** that make the stealth claim
falsifiable (program standing methodology rule 3):

===================  ==========================================================
base name            what it isolates
===================  ==========================================================
`aurora` / `vertex`  a plausible screen logo: bold low-frequency mark on a solid
/ `solstice`         fill. Three of them, so a hijack cannot be a property of one
                     lucky carrier image.
`scrambled:<name>`   the same pixels, block-permuted: identical colour histogram,
                     destroyed structure. Separates "the logo's *appearance*
                     matters" from "a high-contrast carrier matters".
`flat:<name>`        the logo's mean colour as a uniform fill -- the colour-matched
                     blank. delta-around-flat-fill isolates the carrier's
                     structure from its colour.
`gray`               mid-grey 0.5, matching the existing `blank` patch_mode control.
===================  ==========================================================

Everything here is deterministic and CPU-only: same inputs -> byte-identical arrays, so a
base can be regenerated from this file alone and the published patch PNG is checkable
against it (the L-infinity stealth bound must be *externally verifiable*, not self-reported
by the optimiser).

Write the inspection PNGs:

    python experiments/patch_attack/make_logo.py --out runs/monitor-stealth/bases --size 80
"""

from __future__ import annotations

import argparse
import os
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

#: Base images are float32 `[h, w, 3]` in [0, 1] throughout.
FloatArray = NDArray[np.float32]

#: Draw at this multiple of the target side, then downsample -> clean antialiased edges and
#: genuinely low-frequency content (the whole point of a "logo" base).
SUPERSAMPLE: Final[int] = 4

#: The rect the stealth experiment is locked to (BL corner 80x80 = 12.8% of the 224 frame).
DEFAULT_SIZE: Final[int] = 80

#: Blocks per side for `scrambled:` bases. 8 divides every corner size we have used
#: (80, 64, 48, 40, 32), so the permutation is exact and no pixels are dropped.
SCRAMBLE_BLOCKS: Final[int] = 8


@dataclass(frozen=True)
class LogoSpec:
    """A flat-design mark on a solid fill -- deliberately low-frequency and unremarkable."""

    background: str
    foreground: str
    mark: str


LOGOS: Final[dict[str, LogoSpec]] = {
    "aurora": LogoSpec(background="#1F6F6B", foreground="#F4F1E8", mark="ring"),
    "vertex": LogoSpec(background="#2B3A67", foreground="#F2C14E", mark="chevron"),
    "solstice": LogoSpec(background="#B03A2E", foreground="#FDF6E3", mark="disc"),
}

#: The full base set an experiment sweeps: three carriers plus the three structural controls.
CONTROL_BASES: Final[tuple[str, ...]] = ("scrambled:aurora", "flat:aurora", "gray")
ALL_BASES: Final[tuple[str, ...]] = tuple(LOGOS) + CONTROL_BASES


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    digits = value.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"expected a #rrggbb colour, got {value!r}")
    return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _draw_ring(draw: ImageDraw.ImageDraw, side: int, fg: tuple[int, int, int]) -> None:
    inset = side * 0.24
    draw.ellipse(
        (inset, inset, side - inset, side - inset), outline=fg, width=int(side * 0.115)
    )


def _draw_chevron(draw: ImageDraw.ImageDraw, side: int, fg: tuple[int, int, int]) -> None:
    left, right, top, bottom = side * 0.22, side * 0.78, side * 0.26, side * 0.74
    mid_x, notch = side * 0.5, side * 0.20
    draw.polygon(
        [
            (left, top),
            (mid_x, bottom - notch),
            (right, top),
            (right, top + notch),
            (mid_x, bottom),
            (left, top + notch),
        ],
        fill=fg,
    )


def _draw_disc(draw: ImageDraw.ImageDraw, side: int, fg: tuple[int, int, int]) -> None:
    inset = side * 0.28
    draw.ellipse((inset, inset, side - inset, side - inset), fill=fg)
    bar_h = side * 0.09
    draw.rectangle(
        (side * 0.14, side * 0.5 - bar_h / 2, side * 0.86, side * 0.5 + bar_h / 2), fill=fg
    )


_MARKS: Final[dict[str, Callable[[ImageDraw.ImageDraw, int, tuple[int, int, int]], None]]] = {
    "ring": _draw_ring,
    "chevron": _draw_chevron,
    "disc": _draw_disc,
}


def _render_logo(spec: LogoSpec, size: int) -> FloatArray:
    side = size * SUPERSAMPLE
    image = Image.new("RGB", (side, side), _hex_to_rgb(spec.background))
    _MARKS[spec.mark](ImageDraw.Draw(image), side, _hex_to_rgb(spec.foreground))
    downsampled = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(downsampled, dtype=np.float32) / 255.0


def _flat_fill(base: FloatArray) -> FloatArray:
    """Uniform fill at the base's mean colour -- the colour-matched blank control."""
    return np.broadcast_to(base.mean(axis=(0, 1)), base.shape).astype(np.float32).copy()


def _scramble(base: FloatArray, salt: str) -> FloatArray:
    """Block-permute the base: same colour histogram, destroyed structure."""
    size = base.shape[0]
    if size % SCRAMBLE_BLOCKS:
        raise ValueError(f"size {size} must be divisible by SCRAMBLE_BLOCKS={SCRAMBLE_BLOCKS}")
    block = size // SCRAMBLE_BLOCKS
    grid = base.reshape(SCRAMBLE_BLOCKS, block, SCRAMBLE_BLOCKS, block, 3).transpose(0, 2, 1, 3, 4)
    flat = grid.reshape(SCRAMBLE_BLOCKS * SCRAMBLE_BLOCKS, block, block, 3)
    # Salted by name so each carrier scrambles differently. crc32, NOT hash(): Python's
    # string hash is randomised per process, which would make the control non-reproducible.
    rng = np.random.default_rng(zlib.crc32(salt.encode("utf-8")))
    permuted = flat[rng.permutation(flat.shape[0])]
    scrambled: FloatArray = (
        permuted.reshape(SCRAMBLE_BLOCKS, SCRAMBLE_BLOCKS, block, block, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(size, size, 3)
        .copy()
    )
    return scrambled


def build_base(name: str, size: int = DEFAULT_SIZE) -> FloatArray:
    """Return the base image `name` as float32 `[size, size, 3]` in [0, 1].

    Accepts a logo name from `LOGOS`, `"gray"`, or a derived control `"flat:<logo>"` /
    `"scrambled:<logo>"`. Deterministic: the same arguments always give identical pixels.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if name == "gray":
        return np.full((size, size, 3), 0.5, dtype=np.float32)

    kind, _, carrier = name.partition(":")
    if not carrier:
        if kind not in LOGOS:
            raise ValueError(f"unknown base {name!r}; known: {sorted(LOGOS)} + {CONTROL_BASES}")
        return _render_logo(LOGOS[kind], size)
    if kind == "flat":
        return _flat_fill(build_base(carrier, size))
    if kind == "scrambled":
        return _scramble(build_base(carrier, size), carrier)
    raise ValueError(f"unknown base modifier {kind!r} in {name!r}; use 'flat:' or 'scrambled:'")


def save_png(base: FloatArray, path: str) -> str:
    """Write a `[h, w, 3]` float base to `path` as an 8-bit PNG; return the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray((np.clip(base, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the stealth base images for inspection.")
    parser.add_argument("--out", default="runs/monitor-stealth/bases", help="output directory")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="patch side in pixels")
    args = parser.parse_args()

    for name in ALL_BASES:
        base = build_base(name, args.size)
        path = save_png(base, os.path.join(args.out, f"{name.replace(':', '_')}.png"))
        print(f"[make_logo] {name:<18} mean={base.mean(axis=(0, 1)).round(3)} -> {path}")


if __name__ == "__main__":
    main()
