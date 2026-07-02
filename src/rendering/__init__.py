"""Visual-prompt rendering for AutoPPIA-VLA.

`text_prompt` is the CPU-pure core: it renders a candidate's text + style into a
texture image. Injecting that texture into a live LIBERO/MuJoCo scene is a separate,
GPU-dependent step consumed by `evaluator.openvla_backend`.
"""

from rendering.text_prompt import (
    parse_hex_color,
    render_prompt_from_candidate,
    render_prompt_texture,
)

__all__ = [
    "parse_hex_color",
    "render_prompt_from_candidate",
    "render_prompt_texture",
]
