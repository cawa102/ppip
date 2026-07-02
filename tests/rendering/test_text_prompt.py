"""Contract of the visual-prompt texture renderer (CPU-pure, GPU-independent).

`render_prompt_texture` turns a candidate's readable text + style into an RGB image
(a text-bearing label/sticker). It is the approach-agnostic core of `src/rendering/`:
whatever mechanism later injects the prompt into the LIBERO scene consumes this image.
Being pure numpy/PIL, it is fully unit-testable without MuJoCo or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from rendering.text_prompt import (
    parse_hex_color,
    render_prompt_from_candidate,
    render_prompt_texture,
)
from tests.support import make_candidate

_WHITE = "#ffffff"
_BLACK = "#000000"


def _style(**overrides):
    style = {
        "foreground_color": _BLACK,
        "background_color": _WHITE,
        "font_size": 24,
        "font_family": "sans-serif",
    }
    style.update(overrides)
    return style


def test_parse_hex_color_returns_rgb_triple():
    assert parse_hex_color("#ff8800") == (255, 136, 0)
    assert parse_hex_color("#000000") == (0, 0, 0)


def test_parse_hex_color_rejects_malformed_value():
    with pytest.raises(ValueError):
        parse_hex_color("#fff")


def test_renders_rgb_uint8_image():
    img = render_prompt_texture("STOP", _style())

    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_background_fills_the_canvas_corners():
    # A white background must actually paint the label background, not leave holes.
    img = render_prompt_texture("STOP", _style(background_color=_WHITE))

    assert tuple(img[0, 0]) == (255, 255, 255)
    assert tuple(img[-1, -1]) == (255, 255, 255)


def test_foreground_text_is_drawn_in_its_color():
    # Red text on a white background: the non-background pixels are the glyphs and
    # must be dominated by the red channel.
    img = render_prompt_texture("STOP", _style(foreground_color="#ff0000"))

    non_bg = (img != np.array([255, 255, 255], dtype=np.uint8)).any(axis=2)
    assert non_bg.sum() > 0, "no text pixels were drawn"
    glyph_pixels = img[non_bg]
    assert glyph_pixels[:, 0].mean() > glyph_pixels[:, 1].mean()
    assert glyph_pixels[:, 0].mean() > glyph_pixels[:, 2].mean()


def test_larger_font_size_yields_a_taller_image():
    small = render_prompt_texture("STOP", _style(font_size=12))
    large = render_prompt_texture("STOP", _style(font_size=48))

    assert large.shape[0] > small.shape[0]


def test_generic_and_missing_font_family_fall_back_without_error():
    # `font_family` is a CSS-generic ("sans-serif") or may be absent; neither is a
    # real TTF path, so both must fall back to a bundled font rather than raising.
    generic = render_prompt_texture("STOP", _style(font_family="sans-serif"))
    style_no_family = _style()
    del style_no_family["font_family"]
    missing = render_prompt_texture("STOP", style_no_family)

    assert generic.shape[2] == 3 and missing.shape[2] == 3


def test_render_prompt_from_candidate_uses_candidate_text_and_style():
    candidate = make_candidate()
    from_candidate = render_prompt_from_candidate(candidate)
    direct = render_prompt_texture(
        candidate["visual_prompt"]["text"], candidate["style"]
    )

    assert np.array_equal(from_candidate, direct)
