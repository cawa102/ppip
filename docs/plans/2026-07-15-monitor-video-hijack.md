# Physically-Realizable Monitor-Video Hijack — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prove one optimized in-scene *monitor video* can targeted-hijack stock OpenVLA-7B on LIBERO through the real renderer (camera never touched), then measure how much survives open-loop replay.

**Architecture:** A visual-only textured "monitor" geom is placed in the LIBERO scene; its texture is re-uploaded **in place** (no sim reset) every control step. The attacker designs each texture white-box offline; at deployment the recorded sequence is *merely played*. Every scored rollout's policy input is a fresh MuJoCo render of the scene-with-monitor — enforced by a new backend that hashes three canonical image stages. The design/search side never writes the camera buffer and never touches the fixed `eval_goal_state` scorer.

**Tech Stack:** Python, OpenVLA-7B, LIBERO/robosuite/MuJoCo (EGL), torch (`vla_diff` differentiable path), pytest. GPU 1 only (`CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl`).

**Provenance:** Design rationale in `PLAN.md`; full grill + 4-round Codex review in `PLAN-REVIEW-LOG.md`. Read those for *why* each decision was made. Reuses the proven `alphabet_soup → salad_dressing` pair and the fixed target predicate.

---

## Reading order for a fresh session
1. `docs/research/research-log.md` → "➡️ Next session: START HERE".
2. This plan (`PLAN.md` + `PLAN-REVIEW-LOG.md` for rationale).
3. `experiments/patch_attack/adaptive_attack.py` (the current camera-buffer hijack this work replaces), `vla_diff.py` (differentiable path), `src/rendering/inject.py` + `geometry.py` (texture injection), `src/evaluator/openvla_backend.py` (`OpenVLARolloutBackend`, esp. `run_rollouts` @410, image read @~224).

## The two gates (do not skip)
- **GATE A — after Task 1:** if in-place texture upload without a sim reset is infeasible in the robosuite/LIBERO wrapper, **STOP**. Reset-based variants are *out-of-claim* (they reintroduce the OSC-controller reset the result depends on avoiding) unless a controller-state-preserving reset is first *validated*. Report and escalate to the user.
- **GATE B — after Task 8:** benchmark Phase 1 only if Phase 0 produced a real hijack (targeted success with the optimized video beating both controls by margin). Otherwise the deliverable is the boundary result; do not build Phase 1.

---

## Phase 0 — prove ONE video hijacks through the real renderer

- [x] **Task 1: In-place monitor texture-upload spike (GATE A — riskiest plumbing, do first)** — DONE 2026-07-15. **GATE A: PASS.** `src/rendering/monitor.py` (`build_monitor_asset`, `MonitorTextureHandle` via `mjr_uploadTexture`, `mask_local_hash`, `outside_mask_delta`, `dilate_mask`) + `experiments/patch_attack/monitor_upload_probe.py` + `tests/rendering/test_monitor.py` (4 CPU + 1 GPU spike). Probe on GPU 1: 20 in-place uploads, **0 resets, distinct_hashes=20, max_outside_delta=0.0 (bit-identical outside a 2px-dilated monitor mask), max_eef_jump=0.0** (visual-only, no physics disturbance). mujoco 3.9 uses `tex_data` (not the plan's `tex_rgb`/mujoco-py name); context = `sim._render_context_offscreen.con`.

**Files:**
- Create: `src/rendering/monitor.py`
- Create: `experiments/patch_attack/monitor_upload_probe.py`
- Test: `tests/rendering/test_monitor.py`

**What:** Prove one **fixed `W×H`** monitor texture asset can be mutated and re-uploaded to the *active* render context every control step, in one continuous process, with **no** `reset_from_xml_string`. Detect change with a mask-local hash; confirm the rest of the frame is stable within tolerance.

**Interface:**
- `build_monitor_asset(candidate: dict, tex_hw: tuple[int,int]) -> PromptGeom` — fixed-size visual-only geom (`contype=0/conaffinity=0`), reuses `geometry.build_prompt_geom` conventions.
- `class MonitorTextureHandle`: `resolve(env) -> None` (find `texid` in `model.tex_*`); `upload(rgb: NDArray[uint8]) -> None` (mutate `model.tex_rgb` slice + re-upload to render context); `dims -> tuple[int,int]`.
- `mask_local_hash(image, mask) -> str` and `outside_mask_delta(a, b, mask) -> float`.

**Test scenarios (CPU-pure where possible; GPU-guarded via `PPIP_GPU_TESTS`):**
- `build_monitor_asset` yields a fixed-dim texture regardless of content.
- Two different uploads change the mask-local hash; outside-mask delta stays under tolerance.
- (GPU) uploading across 20 steps in one process never calls `reset_from_xml_string`; OSC controller state is continuous (end-effector pose does not jump on a no-op action after upload).

**Dependencies:** `rendering.geometry`, `rendering.inject` (asset XML build only), MuJoCo/robosuite env.

**Notes:** In-place upload cannot change texture dimensions — allocate max size once. If no upload seam exists in the wrapper, this is GATE A: stop and report. Full-frame exact-equality checks are too brittle (AA/resolution) — use the mask-local hash + tolerance.

**Commit:** `feat(monitor): in-place per-step texture upload spike + change-detection`

- [x] **Task 2: Monitor placement + UV/mirror calibration + projected-quad mask** — DONE 2026-07-15. Added to `src/rendering/monitor.py`: `center_crop_mask` (pre-crop→post-crop 224, cross-checked vs `vla_diff.center_crop_resize`, IoU>0.98), `UVMap`/`Homography`/`homography_from_correspondences` (DLT)/`homography_quad_to_texture`, `calibrate_uv` (self-calibrating white-on-black corner detection → mirror handled empirically), `monitor_mask_224` (self-calibrating black↔white contrast diff → dilate → post-crop). Tests: 3 CPU-pure + 1 GPU (`test_calibrate_uv_round_trips_and_mask_is_in_frame`: corners in-frame + convex, interior marker round-trip <12px, mask non-empty). **Critical discovery (feeds Task 4):** `obs['agentview_image']` does NOT reflect an mjr upload (separate/cached render path), but `sim.render` does and is byte-identical otherwise — so `_policy_input_frame` builds the policy input from a FRESH `sim.render` fed through `get_libero_image` (the Task-4 "fresh render after upload" invariant, now the single source of truth). Deviation: `monitor_mask_224(env, handle)` (needs the handle to contrast-probe) not `(env, geom)`. ruff + mypy --strict clean.

**Files:**
- Modify: `src/rendering/monitor.py`
- Test: `tests/rendering/test_monitor.py`

**What:** Fix a world pose that is visible to the agentview camera and does **not** occlude the target/user objects; calibrate the front-face UV + MuJoCo mirror flip with a test grid; compute the monitor's projected region as a mask in **post-crop 224 policy-input coordinates** (accounts for `vla_diff` center-crop).

**Interface:**
- `calibrate_uv(env, handle) -> UVMap` — maps texture (u,v) → image location via a rendered grid.
- `monitor_mask_224(env, geom) -> NDArray[bool]` — projected quad in post-crop policy-input coords.
- `homography_quad_to_texture(uv_map) -> Homography` — invert for texture synthesis.

**Test scenarios:**
- UV grid round-trips within pixel tolerance; mirror flip corrected (text reads un-mirrored).
- Mask lies within frame, excludes target/user object footprints at their canonical poses.
- Mask recomputed under the center-crop matches the region `vla_diff.preprocess` actually samples.

**Dependencies:** `rendering.geometry`, `vla_diff` (crop constants).

**Notes:** Region is *constant up to* crop + dynamic robot occlusion — log per-step visibility, do not assume constant.

**Commit:** `feat(monitor): UV/mirror calibration + post-crop projected mask`

- [x] **Task 3: Phase-aware target-progress metric (CPU-pure)** — DONE 2026-07-15. `experiments/patch_attack/progress_metrics.py` (`Phase` IntEnum APPROACH<GRASP<CARRY<CONTAINMENT, `ProgressState`, `phase_progress(object_states, eef_pose, target_obj, basket_region, initial_target_pos)`) + `tests/patch_attack/test_progress_metrics.py` (6 CPU-pure). Phase gate: displacement≥3cm ⇒ CARRY/CONTAINMENT (scalar=target→basket), else eef<5cm ⇒ GRASP, else APPROACH (scalar=eef→target). Containment requires displacement AND region membership (guards false hijack). Signature adds `initial_target_pos` (plan omitted it; displacement needs a rest reference). ruff + mypy --strict clean.

**Files:**
- Create: `experiments/patch_attack/progress_metrics.py`
- Test: `tests/patch_attack/test_progress_metrics.py`

**What:** A phased progress signal replacing bare `min_target_dist`: eef→target distance (approach) → contact/grasp state → target-object displacement (lift/carry) → target→basket containment distance (place).

**Interface:**
- `@dataclass(frozen=True) ProgressState(phase: Phase, scalar: float)`
- `phase_progress(object_states, eef_pose, target_obj, basket_region) -> ProgressState`

**Test scenarios:**
- Synthetic states drive each phase transition in order.
- Pre-grasp approach decreases scalar while target object is stationary (the blunt-metric failure case).
- Containment fires only after displacement + region membership.

**Dependencies:** `evaluator.adjudicate` types only (read); pure math.

**Notes:** CPU-testable with hand-built states — no OpenVLA. This is both the Task-6 inner *outcome* gate and a reported metric.

**Commit:** `feat(patch_attack): phase-aware target-progress metric`

- [x] **Task 4: `MonitorHijackBackend` with canonical-stage invariant** — DONE 2026-07-15 (CPU + **GPU verified**; the GPU `step_with_texture` test passed once GPU 1 freed of the external `adaptive_attack.py` job). `experiments/patch_attack/monitor_hijack_backend.py`: `canonical_stage_hashes` (S1 raw render / S2 224 policy input / S3 pixel_values), `assert_policy_input_fresh` (S2 must equal transform(S1) — rejects stale obs), `StalePolicyInputError`, `StepResult`, `MonitorHijackBackend(OpenVLARolloutBackend).step_with_texture` (upload→fresh render→feed exact post-upload frame→step; camera buffer never written). Tests `tests/patch_attack/test_monitor_backend.py`: 3 CPU-pure (hashes, freshness accept/reject, structural "not the camera-write path") + 1 GPU (`test_step_with_texture_feeds_the_fresh_post_upload_monitor`: 3 distinct stage hashes, action shape (7,), policy image reflects the uploaded monitor via a fresh render, +50 brightness margin vs black-monitor). ruff + project mypy clean.

**Files:**
- Create: `experiments/patch_attack/monitor_hijack_backend.py`
- Test: `tests/patch_attack/test_monitor_backend.py`

**What:** Subclass `OpenVLARolloutBackend`. Enforce the threat-model invariant: policy input derives **only** from a fresh render *after* the texture upload; camera buffer never written. Strict per-step ordering: upload → **fresh render** → build policy input → step.

**Interface:**
- `class MonitorHijackBackend(OpenVLARolloutBackend)`
- `step_with_texture(env, handle, texture, instruction) -> StepResult` — uploads, fresh-renders, feeds the exact post-upload frame.
- `canonical_hashes(env) -> tuple[str,str,str]` — S1 raw render `H×W×3`, S2 cropped/resized 224 input, S3 processor `pixel_values`.
- `assert_policy_input_fresh(...)` — S2/S3 derive from the post-upload S1; raise on stale.

**Test scenarios:**
- After an upload, the frame fed to the policy is the post-upload render (a deliberately-stale `obs` fails the guard — unit-test the assertion).
- The three hashes are logged every step; camera-buffer write path is never invoked (assert no perturbation applied to the model input tensor).

**Dependencies:** `evaluator.openvla_backend`, `rendering.monitor`.

**Notes:** This is what mechanizes the invariant — do NOT route through `hijack_backend.py`/`adaptive_attack.py` (those execute from a perturbed `pu8` = camera-buffer write).

**Commit:** `feat(patch_attack): MonitorHijackBackend with canonical-stage invariant`

- [ ] **Task 5: Neutral-teacher render + S0 sanity gate**

**Files:**
- Create: `experiments/patch_attack/monitor_attack.py` (teacher + S0 only for now)
- Test: `tests/patch_attack/test_monitor_backend.py`

**What:** Teacher tokens = `OpenVLA(neutral_frame, TARGET=salad_dressing)` where `neutral_frame` renders the same sim state with the monitor **present but showing neutral content** (same geometry as deployment — not "off"). S0 gate: confirm `TARGET` still *succeeds* with the neutral monitor present, across **seeds 0–4**; record any seed exclusions.

**Interface:**
- `teacher_tokens(backend, env, handle, neutral_rgb) -> Tensor[1,7]` (reuses `adaptive_attack._real_tokens` idiom).
- `s0_sanity(backend, seeds) -> dict[int,bool]` — target-success per seed with neutral monitor.

**Test scenarios (GPU-guarded):**
- Neutral monitor render matches deployment geometry (occlusion identical to attack render).
- S0 records per-seed target success; a failing seed is flagged for placement change, not silently used.

**Dependencies:** Tasks 1–4, `adaptive_attack` token helpers.

**Notes:** If S0 fails on a seed, the blank monitor itself perturbs the target policy → teacher tokens invalid → change placement/size (Task 2) before proceeding.

**Commit:** `feat(patch_attack): neutral-teacher render + S0 sanity gate`

- [ ] **Task 6: Texture design — masked-δ + homography + surrogate calibration**

**Files:**
- Create: `experiments/patch_attack/texture_surrogate.py`
- Test: `tests/patch_attack/test_texture_surrogate.py`

**What:** Design a texture that, when *rendered on the real monitor*, drives target progress. Optimize masked-δ in post-crop 224 space via `vla_diff`; invert homography to a texture; **calibrate a texture→policy-input surrogate from the real render** and iterate. Per-texture selection is **stateless** (target-token CE / logit margin on the post-upload render — no env step). Black-box/finite-difference is the fallback if the surrogate gap won't close.

**Interface:**
- `propose_texture(frame224, mask, teacher_tokens, homography) -> NDArray[uint8]`
- `calibrate_surrogate(proposal, real_render_policy_input) -> Surrogate`
- `select_texture(candidates, backend) -> NDArray[uint8]` — stateless proxy scoring.

**Test scenarios:**
- Masked-δ stays inside the mask (outside-mask unchanged).
- Homography inversion + real render reproduces the proposal within the surrogate's measured gap.
- Selection is stateless — env state identical before/after scoring N candidates.

**Dependencies:** `vla_diff`, `rendering.monitor`, Task 4 backend.

**Notes:** Additive δ is a *proposal*, not the deployed op (a rendered texture replaces pixels with shading). Optional cloned shadow-env lookahead with validated full-state restore — never mutate the committed rollout to test a candidate.

**Commit:** `feat(patch_attack): masked-δ texture design + real-render surrogate`

- [ ] **Task 7: Closed-loop oracle run (Stage 1) + record video + full logging**

**Files:**
- Modify: `experiments/patch_attack/monitor_attack.py`
- Test: `tests/patch_attack/test_texture_surrogate.py` (smoke)

**What:** Per step: teacher (Task 5) → design+select texture (Task 6) → `step_with_texture` (Task 4) → phased progress (Task 3). Record `texture_0..T` and per-step logs. Seeds 0–4. This is the oracle upper bound (assumes per-step attacker reaction) — not the deployment threat.

**Interface:**
- `run_oracle(backend, seed, max_steps) -> OracleResult` (targeted_success, commanded_success, phased trajectory, `texture_0..T`, per-step log rows).

**Test scenarios (GPU-guarded, short-horizon smoke):**
- One short seed runs end-to-end, 0 errored steps, video recorded, all log fields present.

**Logging (per step):** S1/S2/S3 hashes, rendered policy image, segmentation/visibility mask, inner token match, decoded action, phased target pose, upload status.

**Dependencies:** Tasks 3–6.

**Commit:** `exp(monitor): closed-loop oracle Stage 1 + recorded video`

- [ ] **Task 8: Open-loop replay (Stage 2) + controls + margin report (GATE B)**

**Files:**
- Create: `experiments/patch_attack/monitor_replay.py`
- Create: `runs/<run_id>/README.md` (headline)

**What:** Replay recorded `texture_0..T` **strictly time-indexed** (`texture_t = video[t]`, no re-optimization, no online frame selection) on the same seed. Run controls: **blank monitor** and **time-scrambled video**. Report the **full metric panel** for attack + both controls; the hijack claim is the **margin** on target progress. Headline = Stage-1→Stage-2 degradation. Predefine a trajectory-divergence metric + threshold; a same-seed replay failure is an explicit **no-deployment** result.

**Interface:**
- `run_replay(backend, seed, video) -> ReplayResult`
- `run_control(backend, seed, kind: Literal['blank','scrambled']) -> ReplayResult`
- `margin_report(attack, controls) -> dict`

**Test scenarios (GPU-guarded):**
- Replay is time-indexed only (no state-conditioned selection path reachable).
- Controls produce the full metric panel; margin computed over target displacement/containment.

**Dependencies:** Tasks 3, 4, 7.

**Notes:** Offline DAgger/EoT (baking one robust fixed video) is allowed at *precompute*; any deployment-time state-matched selection is a **separate labeled closed-loop ablation**, not this open-loop claim. **GATE B** decision made here.

**Commit:** `exp(monitor): open-loop replay Stage 2 + controls + margin`

- [ ] **Task 9: Living-docs update (mandatory, same change)**

**Files:**
- Modify: `docs/research/research-log.md` (dated entry + "START HERE" + Status at a glance)
- Modify: `CLAUDE.md` (status line — currently stale re: hijack), `docs/plans/2026-07-01-autoppia-vla.md` (Implementation Status)
- Create: `runs/<run_id>/README.md`, memory pointer under the auto-memory dir + `MEMORY.md` line

**What:** Record the outcome (hijack margin or boundary), the reality-gap measurement, GATE A/B decisions, and reproduce command. Keep snapshots consistent.

**Commit:** `docs(monitor): record physically-realizable hijack result + reality gap`

---

## Phase 1 — benchmark the loop as design-space search (coarse; ONLY if GATE B passes)

> Intentionally low-resolution: the action space and budget can only be right-sized *after* Phase 0 measures the reality gap and per-candidate cost. Do not pre-build.

- [ ] **Task 10 (coarse): loop-owned design-space benchmark** — freeze optimizer+renderer+evaluator as trusted tools; create a **new versioned `additionalProperties:false` optimizer-design schema** + validator (the current `attack_candidate.schema.json` rejects the extended spec); candidates carry **parameters only** (evaluator owns all artifact paths under the run dir); a **finite predeclared action space**; **all** conditions (`random_search`, `one_shot_llm`, `human_ppia`, loop) emit the same config type; budget equalized on **optimizer-calls/GPU-hours/seeds**, not candidate count; add **ledger file-locking / single idempotent scheduler** if jobs run in parallel. L2 cross-seed as the stretch; collapse-to-DoS = boundary result. Expand into per-task detail after GATE B.

---

## Out of scope
Physical robot / real users / sim-to-real (L3); the loop *computing* pixels (always the fixed optimizer); training-time poisoning; the no-gradient MSc-safe scope (this is explicitly white-box, as the existing adaptive hijack already is — the DoS-boundary result stays a separate regime).
