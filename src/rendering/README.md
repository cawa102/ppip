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

**Done — Option A scene injection (verified on GPU).** The label is injected as a
thin, **visual-only** (`contype=0`, `conaffinity=0`) textured box geom placed in the
LIBERO/MuJoCo scene at `candidate["placement"]` (`position`, `rotation`, `scale`,
within `evaluator.validation.PLACEMENT_BOUNDS`). Two modules:

- `geometry.py` (`build_prompt_geom`) — CPU-pure: candidate placement/style →
  `PromptGeom` (world pos, orientation quaternion from XYZ-degree Euler, half-extents
  sized to the texture aspect × scale, texture). Unit-tested.
- `inject.py` — `build_injection_xml` / `write_texture_png` are CPU-pure (add the
  texture/material asset + geom to the model XML) and unit-tested; `inject_prompt` is
  the thin MuJoCo seam (`get_xml` → inject → `reset_from_xml_string`). Verified on the
  GPU host: injected into a live `libero_spatial` scene, the label renders in agentview
  with correct perspective and occlusion.

The image-space 2D-overlay alternative was rejected (out-of-scene / hidden-channel).
Level progression: readable text plane (`semantic_visual_prompt` /
`optimized_typographic_prompt`) first, hybrid object/texture cues
(`hybrid_prompt_object`) only as a stretch.

**Open tuning (not mechanism):** default panel sizing for long text, and texture
orientation/readability vs. camera (depends on `rotation`). These are candidate-design
knobs the search loop will exercise.
