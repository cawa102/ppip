# Rendering

This directory will contain utilities for inserting visual prompts into LIBERO scenes.

Start with simple readable text planes or labels before adding object textures or hybrid prompt objects.

## Status: deferred to the GPU host

Rendering requires MuJoCo/LIBERO, so it is implemented and verified on the GPU
machine, not in the CPU-only harness phase. It is called from
`OpenVLARolloutBackend.run_rollouts` (`src/evaluator/openvla_backend.py`): given a
validated candidate, it should place a text plane in the LIBERO scene at
`candidate["placement"]` (`scene_anchor`, `position`, `rotation`, `scale`, all
inside `evaluator.validation.PLACEMENT_BOUNDS`) styled per `candidate["style"]`
(foreground/background hex, `font_size`, `font_family`), then hand the modified
env to the closed loop. Level progression: readable text plane
(`semantic_visual_prompt` / `optimized_typographic_prompt`) first, hybrid
object/texture cues (`hybrid_prompt_object`) only as a stretch.
