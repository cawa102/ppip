"""Render a candidate's visual prompt into a texture image (CPU-pure).

This is the approach-agnostic core of the rendering layer: it turns the readable
text + style of an attack candidate into an RGB image of a text-bearing label. The
mechanism that injects that label into the LIBERO scene (a MuJoCo geom/texture) is a
separate, GPU-dependent step; it consumes the image produced here. Keeping this piece
free of MuJoCo/torch makes it fully unit-testable without a GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

# CSS-generic family names carried in candidate style are not real font files; they
# (and a missing family) fall back to PIL's bundled scalable default font.
_GENERIC_FONT_FAMILIES = frozenset({"sans-serif", "serif", "monospace", "default"})
_HEX_LENGTH = 6


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse a ``#RRGGBB`` string into an ``(r, g, b)`` triple of 0-255 ints."""
    digits = value.lstrip("#")
    if len(digits) != _HEX_LENGTH:
        raise ValueError(f"expected a #RRGGBB hex color, got {value!r}")
    return (
        int(digits[0:2], 16),
        int(digits[2:4], 16),
        int(digits[4:6], 16),
    )


def _load_font(
    family: str | None, size_px: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolve a candidate ``font_family`` to a PIL font at ``size_px`` pixels.

    A real, loadable TTF path/name is honoured; CSS-generic names and unresolvable
    values fall back to the bundled default so rendering never fails on font choice.
    """
    if family and family.lower() not in _GENERIC_FONT_FAMILIES:
        try:
            return ImageFont.truetype(family, size_px)
        except OSError:
            pass
    return ImageFont.load_default(size=size_px)


def render_prompt_texture(text: str, style: dict[str, Any]) -> NDArray[np.uint8]:
    """Render ``text`` styled per ``style`` into an ``(H, W, 3)`` uint8 RGB image.

    ``style`` follows the candidate schema: ``foreground_color`` / ``background_color``
    as ``#RRGGBB``, ``font_size`` in pixels, optional ``font_family``. The canvas is
    sized to the text plus symmetric padding proportional to the font size.
    """
    foreground = parse_hex_color(style["foreground_color"])
    background = parse_hex_color(style["background_color"])
    font_px = int(round(float(style["font_size"])))
    font = _load_font(style.get("font_family"), font_px)

    # Measure the text so the label is sized to fit it exactly, then padded.
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = measure.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    padding = max(2, font_px // 4)

    width = int(text_w + 2 * padding)
    height = int(text_h + 2 * padding)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    # Offset by the bbox origin so the glyphs sit inside the padding.
    draw.text((padding - left, padding - top), text, font=font, fill=foreground)

    return np.asarray(image, dtype=np.uint8)


def render_prompt_from_candidate(candidate: dict[str, Any]) -> NDArray[np.uint8]:
    """Render the texture for a validated candidate's visual prompt + style."""
    return render_prompt_texture(candidate["visual_prompt"]["text"], candidate["style"])
