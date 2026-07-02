# Research Log

Living progress tracker. **Status at a glance** is kept current; dated entries are
appended chronologically. Detailed run artifacts live under `runs/`. The task-by-task
plan is `docs/plans/2026-07-01-autoppia-vla.md`.

## Status at a glance (updated 2026-07-02)

GPU-free core harness: implemented + tested — **116 tests, ruff + mypy `--strict` clean**,
run with `~/vla-injection/.venv/bin/python -m pytest` (`PYTHONPATH=~/LIBERO` for the
LIBERO-backed task-resolution/adjudication tests).

- [x] Env verified on the GPU host (reuse the proven `~/vla-injection/.venv`)
- [x] GPU stack pinned (OpenVLA `c8f03f4`, LIBERO `8f1084e`); `libero_object` checkpoint cached
- [x] Fixed evaluator contract (validate / metrics / score / budgets / `evaluate_candidate`)
- [x] Candidate schema + validation
- [x] Autoresearch loop scaffold (ledger, `run_search_condition`, memory, random generator)
- [x] Baselines / search-condition configs
- [x] Result aggregation
- [x] `targeted_success` adjudication design (benchmark predicates, independent labels)
- [x] Rendering **Option A**: text→texture + 3D visual-only geom injection — *verified on GPU*
- [x] Task-pair suite **locked: `libero_object`** (shared scene, distinct predicates)
- [x] Visibility gate (#2) + per-rollout logging (#3)
- [x] Presentation pipeline figure (`docs/figures/pipeline.svg`)
- [ ] `OpenVLARolloutBackend.run_rollouts` — the closed loop (inject → OpenVLA → predicates → visibility → log) — **IN PROGRESS** (plan `docs/plans/2026-07-02-openvla-rollout-backend.md`; CPU-pure seams **A** task-resolution + **B** predicate-adjudication done; GPU seams C–D + smoke E next)
- [ ] Async `submit_evaluation` job path
- [ ] Pilot study (plan Task 7)
- [ ] Task 1 threat-model / literature polish confirmed "dissertation-ready"

## 2026-07-02

- **Env / GPU stack (Phases A–B).** Verified the harness ports cleanly on the GPU host by
  reusing `~/vla-injection/.venv` (uv, torch 2.2.0+cu121, OpenVLA editable, LIBERO via
  `PYTHONPATH=~/LIBERO`, `MUJOCO_GL=egl`). Pinned third-party commits; updated CPU-only
  test guards to GPU reality. Booted a headless `libero_spatial` env + render as a stack smoke.
- **Literature.** Read PPIA (2601.17383) and TRAP (2603.23117); framed the two-sided
  vision-layer landscape (typographic vs adversarial-patch). Locked **scope (a)**:
  autoresearch discovery of PPIA-class typographic injection on OpenVLA/LIBERO; TRAP is the
  related-work boundary (no CoT victim, no patch optimisation).
- **Rendering Option A (Phase C, part 1).** Implemented text→texture, a CPU-pure geom-spec
  builder, and XML-based injection of a thin **visual-only** (`contype=0`) textured geom.
  Verified on GPU: injected into a live `libero_spatial` scene, the label renders in
  agentview with correct perspective + occlusion.
- **Suite decision.** Found `libero_spatial` unusable for hijack pairing (all 10 tasks share
  the identical `bowl→plate` goal). Locked **`libero_object`** (7 objects, one scene, distinct
  `In <object> basket` goals → independent user/target predicates). Downloaded the
  `openvla-7b-finetuned-libero-object` checkpoint; flipped backend defaults (unnorm_key,
  `max_steps=280`).
- **Decisions #2/#3.** Added the `prompt_visibility` gate (segmentation-based) and per-rollout
  artifact logging (`runs/<id>/candidates/<cid>/`: texture, first-frame PNGs, `rollouts.jsonl`).
- **Presentation.** Added `docs/figures/pipeline.svg` (+ README) for the MSc talk.
- GPU discipline: all work above was CPU/disk only except two brief EGL smokes; GPU 0 was
  running a concurrent job and was left untouched (rollout work will pin to a free card).
- **`run_rollouts` seams A+B (TDD, CPU-pure).** Started the closed-loop sub-plan. Added
  `evaluator/libero_tasks.py` (`resolve_task`: candidate free-text task → `libero_object`
  task_id/language/goal predicates via normalised match; raises on no-match/ambiguous/unknown
  suite) and `evaluator/adjudicate.py` (`eval_goal_state`: conjunction of a target goal's
  predicates over a live env's `object_states` via LIBERO `eval_predicate_fn`; raises
  `UnevaluableGoalError` on empty/missing-object — validates *all* objects before evaluating,
  so a missing object never hides behind a short-circuited `False`). Both compute verdicts
  **purely** (no sim/LLM/heuristic) and are unit-tested off-GPU. `goal_state` is an immutable
  tuple-of-tuples (post-review hardening). `python-reviewer` pass: 1 HIGH (mutable verdict
  data) + 3 MED addressed. Suite: 116 tests green.

## 2026-07-01

- Direction selected: autonomous discovery of physical prompt injection attacks against OpenVLA+LIBERO.
- Working name: AutoPPIA-VLA.
- Initial scope: readable visual prompt candidates, fixed evaluator, autoresearch-style loop comparison.
- GPU-free harness scaffold implemented and tested (evaluator, search loop, aggregation).
