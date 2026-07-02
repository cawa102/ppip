# Presentation Figures

Assets for explaining AutoPPIA-VLA to an audience (MSc defence / talk).

## `pipeline.svg` — the system in one picture

A conceptual diagram of the whole loop, emphasising the three things a reviewer cares about:

1. **The contribution is the autonomous *discovery* loop**, not any single attack — the
   yellow **search side** proposes candidates; equal budget across all conditions.
2. **The fixed evaluator is an integrity boundary** (green dashed box): the search side may
   only *read* its outputs, so the agent can never game its own score.
3. **The attack is in-scene and measured rigorously**: render label → inject a visual-only
   3D geom into the LIBERO/MuJoCo scene → OpenVLA rollout → `commanded` vs `targeted`
   success (benchmark predicates) + a `prompt_visibility` gate.

SVG is vector, so it stays crisp on any projector and opens directly in a browser,
Keynote, PowerPoint, or Google Slides (drag-and-drop or Insert → Picture). Edit the text
directly in the file if wording changes.

## Example scene figures (generated from real rollouts)

The rollout logger writes these under `runs/<run_id>/candidates/<candidate_id>/` during a
run — they are the "what the policy actually sees" slides:

- `prompt_texture.png` — the rendered typographic label (the injected instruction).
- `seed<k>_ep<j>_first.png` — the first agentview frame with the label in the scene.
- A `visibility_overlay` variant highlights exactly which pixels are the prompt (from the
  MuJoCo segmentation mask), which visually justifies the visibility gate.

The strongest single slide is a **before/after pair**: the clean scene vs. the same scene
with the injected label, alongside the outcome — the robot placing the *attacker's* object
in the basket instead of the user's (pure targeted substitution). These are produced once
`OpenVLARolloutBackend.run_rollouts` runs on the GPU host (pin to a free card:
`CUDA_VISIBLE_DEVICES=<idx> MUJOCO_GL=egl`).
