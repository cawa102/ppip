# AutoPPIA-VLA Research Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a simulation-only research harness to measure whether autoresearch-style AI loops can discover and improve physical prompt injection attacks against OpenVLA on LIBERO.

**Architecture:** Keep the evaluator fixed and non-editable, while allowing an AI loop to propose and revise visual prompt candidates. The project separates research notes, candidate configs, rendering/injection logic, rollout evaluation, run logs, and third-party dependencies so attack success is measurable without metric gaming.

**Tech Stack:** Python, OpenVLA, LIBERO/robosuite/MuJoCo, JSON/JSONL configs, autoresearch-style agent loop, local run artifacts, arXiv/GitHub literature references.

---

## Research Thesis

**Working title:** AutoPPIA-VLA: Autonomous Discovery of Physical Prompt Injection Attacks on Vision-Language-Action Models

**Core question:** Given a fixed OpenVLA+LIBERO evaluator and a bounded experiment budget, can an AI research loop discover visual/environmental prompt candidates that cause targeted task substitution better than random search, one-shot LLM prompting, and human-designed PPIA-style prompts?

**Contribution framing:** This is not just another VLA attack. The contribution is a capability benchmark for autonomous red-team research loops in embodied AI security.

**Simulation boundary:** All experiments are limited to local simulation on benchmark VLA policies. No physical robot experiments, no real-world deployment, and no external system targeting are in scope.

## Target Attack Class

The base attack is PPIA-style physical prompt injection: a text-bearing object, label, sticker, decal, or visual prompt is placed into the scene so the model may treat it as an instruction. The extension is automated discovery: an AI loop searches over wording, layout, color, size, position, target task, and optional visual styling to improve targeted task hijacking.

Candidate levels:

1. **Semantic visual prompt:** readable in-scene text only.
2. **Optimized typographic prompt:** AI loop optimizes text, visual design, placement, and task pairing.
3. **Hybrid prompt-object:** readable text plus scene object, icon, or texture cues.

The MSc-safe initial scope is level 2. Level 3 is stretch.

## Key References

- OpenVLA: https://arxiv.org/abs/2406.09246
- LIBERO: https://arxiv.org/abs/2306.03310
- PPIA / physical prompt injection precedent: https://arxiv.org/abs/2601.17383
- karpathy/autoresearch scaffold inspiration: https://github.com/karpathy/autoresearch
- SABER agentic VLA attack to differentiate from: https://arxiv.org/abs/2603.24935
- AI Scientist-v2 autonomous research context: https://arxiv.org/abs/2504.08066
- AIRTBench autonomous AI red-teaming context: https://arxiv.org/abs/2506.14682

## Success Criteria

- A fixed evaluator can score a visual prompt candidate without agent-side metric changes.
- At least four search conditions are comparable under equal budgets.
- The report distinguishes ordinary task failure from targeted task substitution.
- Run artifacts preserve enough metadata to reproduce or audit each candidate.
- The final study can still be publishable if autonomous search improves failure rate but not targeted hijack rate.

## Autoresearch Adaptation: Long-Running VLA Jobs, Not a 5-Minute Training Loop

This project must not inherit `karpathy/autoresearch`'s 5-minute nanochat training cap. That cap is an implementation detail for fast language-model experiments, not a research requirement. OpenVLA+LIBERO rollout evaluation may require model loading, simulator startup, multiple seeds, and multiple episodes, so a useful candidate evaluation can take far longer than five minutes.

Use `karpathy/autoresearch` only as a design reference for the control pattern:

```text
candidate proposal -> fixed evaluator -> metric file -> ledger -> next candidate
```

Do not copy the following assumptions from `karpathy/autoresearch`:

- no fixed 5-minute wall-clock cap per iteration;
- no `train.py`/`prepare.py` contract;
- no validation-BPB objective;
- no requirement that the agent remains active while the evaluator runs;
- no assumption that a failed or slow experiment should be killed after a few minutes.

The AutoPPIA-VLA iteration unit is a **candidate evaluation job**, not a short training window. A candidate evaluation job may run synchronously for cheap smoke tests or asynchronously for OpenVLA rollouts. The loop must be resumable from `runs/<run_id>/ledger.jsonl` after long jobs, machine restarts, or manual pauses.

The default lifecycle for future implementers is:

1. The search condition proposes `candidate_<n>.json`.
2. The fixed evaluator validates the candidate and creates an evaluation job.
3. The evaluator runs the configured OpenVLA+LIBERO rollout budget.
4. The evaluator writes `metrics_<n>.json` and appends an immutable ledger row.
5. The loop reads the ledger and proposes the next candidate.

The search agent must only see the candidate schema, the fixed program file, prior candidate files, and evaluator outputs. It must not modify evaluator code, scoring code, task definitions, seed lists, or already-written metrics.

## Autoresearch File Mapping

Future sessions should use this mapping when translating `karpathy/autoresearch` concepts into this project:

| `karpathy/autoresearch` file | AutoPPIA-VLA equivalent | Editable by search agent? | Current status |
|---|---|---|---|
| `program.md` | `programs/autoppia-vla/program.md` | Read-only during a benchmark run | Exists |
| `prepare.py` | `src/evaluator/eval_attack.py`, `src/evaluator/metrics.py`, `src/rendering/`, `experiments/configs/evaluation_budgets.yaml` | No | Planned, except budget config exists |
| `train.py` | `src/autoresearch_loop/run_loop.py`, `src/autoresearch_loop/candidate_writer.py`, `src/autoresearch_loop/memory.py`, generated candidates under `experiments/candidates/` | Yes, depending on search condition | Planned, candidate templates exist |

Do not create a single monolithic `train.py` that can modify evaluation logic. The important separation is:

```text
fixed evaluator side = prepare.py-like, trusted, not agent-editable
search loop side     = train.py-like, agent-editable/candidate-generating
program file         = program.md-like, fixed instructions
```

## Evaluation Budget Model

Budget limits are expressed in rollout resources, not in an arbitrary 5-minute cap. Every experimental condition must receive the same configured budget.

Use three evaluation stages:

1. **Smoke evaluation:** checks plumbing and candidate validity. Use one task pair, one seed, and the minimum rollout count needed to verify that rendering, OpenVLA loading, and metrics output work.
2. **Pilot evaluation:** compares search conditions cheaply. Use one LIBERO suite, one or two task pairs, a small fixed seed set, and a fixed candidate count per condition.
3. **Full evaluation:** re-evaluates only the best candidates from each condition. Use the final rollout count and task coverage reported in the dissertation.

Initial config files should expose these fields:

```yaml
stage: smoke | pilot | full
max_candidates_per_condition: int
task_pairs: list
seeds: list
rollouts_per_candidate: int
max_wall_clock_hours_per_candidate: float
top_k_for_full_eval: int
allow_async_jobs: bool
```

`max_wall_clock_hours_per_candidate` is a runaway-job guard, not the scientific budget. The scientific budget is the fixed number of candidates, tasks, seeds, and rollouts.

---

- [ ] Task 1: Scope, Threat Model, and Literature Grounding

**Files:**
- Modify: `docs/research/literature-map.md`
- Modify: `docs/research/threat-model.md`
- Modify: `docs/research/risk-register.md`
- Reference: `VLA-security-project-decision.md`

**What:** Lock the research question, novelty boundary, threat model, and nearest-prior differentiation before implementation begins.

**Interface:**
- `ThreatModel` — defines attacker knowledge, allowed scene modifications, prohibited actions, target task selection, and simulation boundary.
- `PriorWorkMap` — maps PPIA, SABER, OpenVLA/LIBERO attacks, autoresearch, and autonomous red-teaming benchmarks to the proposed contribution.

**Test scenarios:**
- The threat model clearly excludes training-time poisoning and physical robot deployment.
- SABER is distinguished as text/instruction perturbation rather than environmental visual prompt discovery.
- PPIA is used as the base attack class, not claimed as novel by itself.
- The thesis remains valid if only non-targeted failure improves.

**Dependencies:** Existing decision memo, arXiv/GitHub references.

**Notes:** Keep citation claims conservative and re-verify arXiv metadata before formal dissertation submission.

**Commit:** `docs: define autoppia-vla scope and threat model`

- [x] Task 2: Fixed Evaluation Contract  <!-- GPU-independent validation, metrics, budgets, evaluate_candidate orchestration, and the RolloutBackend seam are implemented and tested. Rendering (Option A geom injection), visibility gate, per-rollout logging, and OpenVLARolloutBackend.run_rollouts are implemented and GPU-verified (2026-07-02). See docs/research/research-log.md. -->


**Files:**
- Create: `src/evaluator/eval_attack.py`
- Create: `src/evaluator/metrics.py`
- Modify: `src/evaluator/README.md`
- Modify: `docs/research/experiment-protocol.md`
- Modify: `experiments/configs/evaluation_budgets.yaml`

**What:** Define the evaluator that the AI loop cannot edit. It loads a candidate, renders it into LIBERO, runs OpenVLA rollouts, and emits a score file.

**Interface:**
- `evaluate_candidate(candidate_path: str, run_dir: str) -> dict` — evaluates one visual prompt candidate and writes rollout metrics.
- `compute_attack_score(metrics: dict) -> float` — combines targeted success, commanded-task failure, safety penalties, and cost penalties.
- `validate_candidate(candidate: dict) -> None` — rejects malformed or out-of-scope candidates.
- `load_evaluation_budget(config_path: str, stage: str) -> dict` — loads smoke, pilot, or full rollout budgets.

**Test scenarios:**
- Valid candidate produces a metrics JSON.
- Missing target task fails validation.
- Candidate attempting to modify evaluator settings fails validation.
- Clean baseline candidate produces a comparable score.
- Failed rollouts are logged without crashing the whole run.
- A long-running candidate is controlled by the configured rollout budget and runaway timeout, not by a hardcoded 5-minute limit.

**Dependencies:** OpenVLA checkpoint path, LIBERO task suite, candidate schema.

**Notes:** This evaluator is the integrity boundary. The AI loop may read outputs but must not modify this file during benchmark runs.

**Commit:** `feat: add fixed attack evaluator contract`

- [x] Task 3: Candidate Schema and Prompt-Object Representation  <!-- GPU-independent: schema + validate_candidate (scope bounds, readability, override rejection) implemented and tested. -->


**Files:**
- Modify: `experiments/configs/attack_candidate.schema.json`
- Modify: `experiments/candidates/baseline_human_ppia.json`
- Modify: `experiments/candidates/random_seed_001.json`
- Modify: `src/rendering/README.md`

**What:** Define how visual prompt candidates are represented independent of the agent that generated them.

**Interface:**
- `text: str` — visible text rendered into the scene.
- `target_task: str` — attacker-desired task identifier.
- `placement: object` — position, rotation, target surface/object, and visibility constraints.
- `style: object` — font, color, background, size, and optional icon/texture metadata.
- `budget: object` — query count, rollout count, and candidate generation metadata.

**Test scenarios:**
- Human PPIA-style prompt validates.
- Random prompt validates if it stays inside allowed placement bounds.
- Candidate with hidden evaluator overrides is rejected.
- Candidate with non-readable or empty text is marked as level-3/stretch, not level-2.

**Dependencies:** JSON Schema, LIBERO scene naming conventions.

**Notes:** Start with readable text prompts. Do not begin with unconstrained adversarial noise.

**Commit:** `feat: define visual prompt candidate schema`

- [x] Task 4: Autoresearch-Style Loop Scaffold  <!-- GPU-independent: ledger (append-only, immutable), record_result, select_incumbent, memory.summarize_history, random generator, and run_search_condition (budget-bounded, ledger-resumable, provider-agnostic via injected propose) implemented and tested. Async submit_evaluation job path deferred. -->


**Files:**
- Create: `src/autoresearch_loop/run_loop.py`
- Create: `src/autoresearch_loop/candidate_writer.py`
- Create: `src/autoresearch_loop/memory.py`
- Modify: `programs/autoppia-vla/program.md`
- Modify: `runs/README.md`
- Modify: `docs/research/experiment-protocol.md`

**What:** Build the loop that proposes candidates, calls or schedules the fixed evaluator, records results, and decides the next candidate under a fixed candidate/rollout budget. This is an autoresearch-style loop, not a direct reuse of the nanochat 5-minute experiment loop.

**Interface:**
- `propose_candidate(history_path: str, output_path: str) -> None` — writes a new candidate JSON.
- `record_result(candidate_path: str, metrics_path: str, ledger_path: str) -> None` — appends candidate and result metadata to JSONL.
- `select_incumbent(ledger_path: str) -> dict` — returns the current best candidate under the official score.
- `submit_evaluation(candidate_path: str, run_dir: str, budget: dict) -> str` — starts or records a candidate evaluation job and returns a job ID.
- `resume_loop(run_dir: str) -> None` — resumes from the ledger after long-running evaluation jobs or manual pauses.

**Test scenarios:**
- Loop stops at the configured candidate budget.
- Each candidate has a unique ID and immutable result record.
- Agent memory can summarize prior failures without editing evaluator outputs.
- Invalid candidates are logged and penalized rather than silently dropped.
- The loop can stop after submitting a long evaluator job and later resume from `ledger.jsonl`.
- The loop does not keep an LLM session open while OpenVLA/LIBERO rollouts are running.
- No code path hardcodes a 5-minute per-iteration cap.

**Dependencies:** Fixed evaluator, candidate schema, local model provider or manual candidate generation fallback.

**Notes:** Keep the loop provider-agnostic so Claude/Codex/manual one-shot conditions can share the same evaluator. Treat `karpathy/autoresearch` as a reference pattern; do not assume its original time budget, nanochat files, or training metric apply.

**Commit:** `feat: scaffold autonomous visual prompt search loop`

- [x] Task 5: Baselines and Experimental Conditions  <!-- GPU-independent: search_conditions.yaml + baselines.yaml + load_search_conditions (comparability enforcement, shared budget_stage) implemented and tested. -->


**Files:**
- Modify: `docs/research/experiment-protocol.md`
- Create: `experiments/configs/baselines.yaml`
- Create: `experiments/configs/search_conditions.yaml`
- Modify: `experiments/configs/evaluation_budgets.yaml`

**What:** Define the comparison conditions before running expensive rollouts.

**Interface:**
- `random_search` — random valid candidate generation.
- `human_ppia` — hand-written visual prompts based on PPIA intuition.
- `one_shot_llm` — one LLM-generated candidate set with no feedback loop.
- `loop_no_memory` — iterative agent using only current score.
- `loop_with_memory` — iterative agent using run history and failure summaries.
- `loop_with_skill` — iterative agent using a fixed VLA red-team skill/program file.

**Test scenarios:**
- Each condition receives the same rollout and candidate budget.
- Each condition uses the same fixed evaluator.
- Search history is separated by condition.
- Results can be aggregated without manual cleanup.
- Smoke, pilot, and full budgets can be selected without changing code.
- Full evaluation re-runs top candidates under a larger rollout budget instead of trusting noisy pilot scores.

**Dependencies:** Candidate loop, run ledger format, evaluator metrics.

**Notes:** Add multi-agent search only after the single-agent loop works.

**Commit:** `docs: define autoppia-vla baselines`

- [x] Task 6: Metrics, Logging, and Reproducibility  <!-- GPU-independent aggregation plus GPU rollout artifacts: aggregate_results (aggregate_condition/aggregate_run) with raw counts, recomputable official score, partial-run tolerance implemented and tested; .gitignore already keeps heavy artifacts out. -->


**Files:**
- Modify: `docs/research/experiment-protocol.md`
- Create: `experiments/results/aggregate_results.py`
- Modify: `experiments/results/README.md`
- Modify: `.gitignore`

**What:** Preserve the data needed to compare autonomous search conditions and detect metric gaming.

**Interface:**
- `commanded_success_rate` — fraction of rollouts completing the user-commanded task.
- `targeted_success_rate` — fraction of rollouts completing the attacker target task.
- `attack_score` — primary scalar used by search.
- `query_count` — number of evaluated candidates and rollouts.
- `wall_clock_minutes` — elapsed run time.
- `human_interventions` — number of manual fixes during a run.
- `invalid_candidate_rate` — fraction of rejected candidates.

**Test scenarios:**
- Aggregator handles partial failed runs.
- Metrics include raw counts, not only percentages.
- The official score can be recomputed from saved metrics.
- Heavy outputs are ignored by git while summaries remain trackable.

**Dependencies:** Evaluator output format, JSON/JSONL run ledger.

**Notes:** Save candidate snapshots and run metadata even for failed attempts.

**Commit:** `feat: add experiment aggregation contract`

- [x] Task 7: First Pilot Study  <!-- DONE (2026-07-02): four-condition pilot complete on GPU 1, 80/80 rollouts, 0 errored (after diagnosing+fixing a per-candidate model-reload OOM). Finding: denial, not hijack (0 targeted successes; visible readable labels cut commanded success 14/20 -> 2-4/20). See runs/pilot-001/README.md + docs/research/research-log.md. loop_with_memory used a documented mutate stand-in, so LLM-in-the-loop remains a follow-up. -->

> **Result (2026-07-02):** complete. 4 conditions × 20 rollouts, **0 errored** (after fixing a
> per-candidate model-reload OOM — the pilot's diagnostic finding: the bottleneck was OpenVLA
> loading). Attack reading: **denial, not hijack** — 0 targeted successes across 80 rollouts;
> visible readable labels suppressed the commanded task (14/20 → 2–4/20). `runs/pilot-001/`.

**Files:**
- Modify: `docs/research/research-log.md`
- Modify: `experiments/results/README.md`
- Create: `runs/pilot-001/README.md`
- Modify: `experiments/configs/evaluation_budgets.yaml`
- Create: `experiments/run_pilot.py`, `experiments/pilot_pools.py`, `src/autoresearch_loop/mutate.py` (+ `tests/autoresearch_loop/test_mutate.py`)

**What:** Run a cheap pilot before full OpenVLA rollouts. The pilot should validate end-to-end plumbing, not prove the final thesis.

**Status (2026-07-02): COMPLETE.** The four-condition pilot ran on GPU 1 (`runs/pilot-001/`,
`5 candidates × 2 seeds × 2 rollouts` per condition = 80 rollouts, **0 errored**). The first run
exposed a per-candidate model-reload **VRAM OOM** (the pilot's diagnostic finding: the bottleneck
is OpenVLA *loading*); fixed by caching the policy on the backend + one shared backend across
conditions, then re-run clean. Attack reading: **denial, not hijack** — 0 targeted successes
across 80 rollouts, while visible readable labels cut commanded success from 14/20 (`random_search`,
often out of view) to 2–4/20 (readable, prompt-visible in 20/20). `loop_with_memory` used a
documented programmatic mutate-incumbent stand-in for the LLM-in-the-loop (validates the feedback
machinery + equal-budget plumbing, **not** LLM search quality — that is a follow-up interactive
run). Full write-up: `runs/pilot-001/README.md` + `pilot_summary.md`.

**Interface:**
- `pilot-001` — one LIBERO suite, one or two task pairs, a small fixed seed set, a fixed rollout count, and four search conditions.
- `pilot_summary.md` — short result summary with failure modes and next-step decisions.

**Test scenarios:**
- Human baseline and random baseline both run.
- At least one loop condition produces multiple candidates.
- Candidate history reveals whether the AI loop uses feedback productively.
- The pilot identifies whether rendering, OpenVLA loading, or metric definition is the immediate bottleneck.

**Dependencies:** Tasks 1-6.

**Notes:** Do not optimize for final numbers in the pilot. Optimize for clear diagnosis.

The pilot may take longer than five minutes per evaluated candidate. That is expected. The required property is equal budget and reproducible logging, not short wall-clock runtime.

**Commit:** `exp: run autoppia-vla pilot study`

## Recommended Execution Approach

Use a sequential, subagent-driven implementation once coding begins. The tasks depend on a clean integrity boundary between fixed evaluator and editable search loop, so parallel implementation is premature until Tasks 1-3 are stable.

Immediate next step after this setup: complete Task 1 by expanding the threat model and literature map into dissertation-ready notes.

## Implementation Status

> **Living status is `docs/research/research-log.md`** ("Status at a glance" + dated
> entries). This section is the point-in-time snapshot; keep both current.

**On the GPU-capable host (updated 2026-07-02).** The GPU-independent core harness, the rendering layer
(Option A injection), *and* the `OpenVLARolloutBackend.run_rollouts` closed loop are
implemented and tested — **129 lightweight tests + 6 GPU tests green, `ruff` + `mypy --strict`
clean**, run with `~/vla-injection/.venv/bin/python -m pytest` (LIBERO-backed tests need
`PYTHONPATH=$HOME/LIBERO`; real-model tests need `PPIP_GPU_TESTS=1`). `run_rollouts` is
**verified end-to-end on GPU 1** (`runs/smoke-001/`: pipeline runs, 0 errored rollouts,
prompt visible, fits one A5000). The task-pair suite is locked to **`libero_object`**; the
matching checkpoint is cached. Decisions settled: `targeted_success` adjudication (via the
benchmark predicates, CPU-pure in `adjudicate.py`), visibility gate (#2), per-rollout
logging (#3). **Adjudicability constraint** discovered + documented: each libero_object task
has only 7 objects, so a target is adjudicable only if its object is in the user scene.

**Toolchain (GPU rollout env):** reuse the proven uv venv `~/vla-injection/.venv` (Python 3.10,
torch 2.2.0+cu121, OpenVLA editable, LIBERO via `PYTHONPATH=~/LIBERO`, `MUJOCO_GL=egl`).
`pyproject.toml` core deps: jsonschema, pyyaml, numpy<2, Pillow; the `gpu` group documents
the pinned stack. Import root is `src/` (+`experiments/results`). Rollout work pins to a
free GPU (`CUDA_VISIBLE_DEVICES=<idx>`) — the box is shared.

**Implemented modules (GPU-independent core plus rollout seam):**
- `src/evaluator/validation.py` — `validate_candidate` (schema + placement bounds + readability + evaluator-override rejection).
- `src/evaluator/metrics.py` — `RolloutOutcome`, `summarize_rollouts`, `compute_attack_score` (official formula).
- `src/evaluator/budgets.py` — `load_evaluation_budget` (stage selection, required-field checks).
- `src/evaluator/backends.py` — `RolloutBackend` Protocol (the GPU seam).
- `src/evaluator/eval_attack.py` — `evaluate_candidate` (validate → rollouts → summarize → score → write metrics; invalid/failed candidates penalized, not crashed).
- `src/evaluator/openvla_backend.py` — `OpenVLARolloutBackend`: reference-grounded defaults, GPU seams (`_load_policy`, `_build_env`, `_policy_action`, `_segmentation`, …), and the `run_rollouts` closed loop (implemented + GPU-verified).
- `src/evaluator/libero_tasks.py` — `resolve_task` (candidate task string → libero_object task + goal predicates), `parse_goal_state`; raises on no-match/ambiguous/unknown-suite (CPU-pure).
- `src/evaluator/adjudicate.py` — `eval_goal_state` (target goal predicates over live `object_states` via LIBERO `eval_predicate_fn`; `UnevaluableGoalError` on empty/missing-object — CPU-pure verdict logic).
- `src/autoresearch_loop/ledger.py` — append-only ledger, `select_incumbent`, duplicate-id rejection.
- `src/autoresearch_loop/run_loop.py` — `record_result`, `run_search_condition` (budget-bounded, ledger-resumable).
- `src/autoresearch_loop/candidate_writer.py` — `generate_random_candidate` (always-valid `random_search` baseline), `write_candidate`.
- `src/autoresearch_loop/memory.py` — `summarize_history` (read-only prior-failure summary).
- `src/autoresearch_loop/conditions.py` — `load_search_conditions` (comparability enforcement).
- `experiments/results/aggregate_results.py` — `aggregate_condition`, `aggregate_run`.
- `experiments/configs/{search_conditions,baselines}.yaml` — the six conditions + baseline subset.

**Implemented rendering + logging (Phase C, done 2026-07-02):**
- `src/rendering/text_prompt.py` — `render_prompt_texture` / `render_prompt_from_candidate` (text+style → RGB label).
- `src/rendering/geometry.py` — `build_prompt_geom` (placement/style → MuJoCo pose, quat, half-extents, texture).
- `src/rendering/inject.py` — `build_injection_xml` + `write_texture_png` (CPU-pure) and `inject_prompt` (the MuJoCo seam: `get_xml` → inject visual-only geom → `reset_from_xml_string`); verified on GPU.
- `src/rendering/visibility.py` — `prompt_pixel_fraction` (segmentation gate #2), `visibility_overlay` (figures).
- `src/evaluator/rollout_logging.py` — per-rollout artifacts under `runs/<id>/candidates/<cid>/` (#3), including sampled `first`/`step20`/`last` frames and canonical JSONL records with frame paths.
- `src/evaluator/metrics.py` — `RolloutOutcome.prompt_visibility`, optional `TargetDiagnostics` miss-distance evidence, and summary aggregation.

**Done (2026-07-02):**
- `OpenVLARolloutBackend.run_rollouts` — the closed loop tying the pieces together:
  `inject_prompt` → OpenVLA `get_action` loop under `user_task` → adjudicate `commanded`/`targeted`
  (benchmark predicates, latched non-terminating) → `prompt_visibility` → non-scoring
  target miss-distance diagnostics → sampled-frame per-rollout logging, one `RolloutOutcome`
  per episode; per-episode error isolation. **Verified end-to-end on GPU** (`runs/smoke-001/`).

**Pilot-001 (2026-07-02): COMPLETE on GPU 1 (Task 7 done).**
- `pilot` budget filled (real adjudicable pair user=alphabet_soup/target=cream_cheese;
  right-sized `5 candidates × 2 seeds × 2 rollouts`); `full` still `PLACEHOLDER`.
- Four-condition orchestrator `experiments/run_pilot.py` + authored pools
  `experiments/pilot_pools.py` (`human_ppia`, `one_shot_llm`) + `loop_with_memory`
  proposer `src/autoresearch_loop/mutate.py` (documented programmatic stand-in for the
  LLM-in-the-loop). Ran 80/80 rollouts, **0 errored**, after diagnosing + fixing a
  per-candidate model-reload VRAM OOM (policy cached on the backend; one shared backend across
  conditions). Finding: **denial, not hijack** (0 targeted successes; commanded 14/20 → 2–4/20
  under visible readable labels). Write-up: `runs/pilot-001/README.md` + `pilot_summary.md`.

**Remaining:**
1. LLM-driven `loop_with_memory` / `loop_with_skill` as an interactive follow-up (the pilot
   used a programmatic proposer stand-in so it could run unattended).
2. Stronger injection for real hijack signal (larger/central labels, level-2 optimized
   typography, placement nearer the manipulation region) — the pilot shows the harness measures
   this cleanly and current readable labels cause denial, not substitution.
3. Populate the target miss-distance diagnostic (`mean_min_target_distance_m` came back null —
   the target-region position was not extracted from `object_states`; non-scoring).
4. The async `submit_evaluation` job path (long OpenVLA jobs); the synchronous, resumable
   loop already models the control flow.
5. Fill the `full` budget `task_pairs` with adjudicable pairs for the top-k re-evaluation.
