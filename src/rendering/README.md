# Rendering

This directory will contain utilities for inserting visual prompts into LIBERO scenes.

Start with simple readable text planes or labels before adding object textures or hybrid prompt objects.

## Status

**Done — texture core (`text_prompt.py`, CPU-pure).** `render_prompt_texture(text,
style)` / `render_prompt_from_candidate(candidate)` turn a candidate's readable text
+ style (foreground/background hex, `font_size`, `font_family`) into an `(H, W, 3)`
uint8 RGB label image. No MuJoCo/GPU dependency, so it is unit-tested in the standard
suite (`tests/rendering/`). This is the approach-agnostic core consumed by whatever
injects the label into the scene.

**Pending — scene injection (GPU host).** Placing that label into the live
LIBERO/MuJoCo scene at `candidate["placement"]` (`scene_anchor`, `position`,
`rotation`, `scale`, all inside `evaluator.validation.PLACEMENT_BOUNDS`) and handing
the modified env to the closed loop, called from `OpenVLARolloutBackend.run_rollouts`
(`src/evaluator/openvla_backend.py`). The injection mechanism (image-space overlay vs.
true 3D textured geom) is an open design decision. Level progression: readable text
plane (`semantic_visual_prompt` / `optimized_typographic_prompt`) first, hybrid
object/texture cues (`hybrid_prompt_object`) only as a stretch.
