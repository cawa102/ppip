"""Visual-prompt rendering for AutoPPIA-VLA.

`text_prompt` is the CPU-pure core: it renders a candidate's text + style into a
texture image. Injecting that texture into a live LIBERO/MuJoCo scene is a separate,
GPU-dependent step consumed by `evaluator.openvla_backend`.
"""

from rendering.geometry import PromptGeom, build_prompt_geom
from rendering.inject import build_injection_xml, inject_prompt, write_texture_png
from rendering.text_prompt import (
    parse_hex_color,
    render_prompt_from_candidate,
    render_prompt_texture,
)

__all__ = [
    "PromptGeom",
    "build_injection_xml",
    "build_prompt_geom",
    "inject_prompt",
    "parse_hex_color",
    "render_prompt_from_candidate",
    "render_prompt_texture",
    "write_texture_png",
]
