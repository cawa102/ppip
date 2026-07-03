# Evaluator

The evaluator is the benchmark integrity boundary.

Responsibilities:

- validate candidate JSON files;
- render the visual prompt into LIBERO;
- run OpenVLA rollouts with fixed seeds and task definitions;
- compute official metrics;
- write immutable metrics outputs.

Search agents may call the evaluator and read its outputs, but benchmark comparisons should not allow them to edit evaluator code.

## Modules

- `validation.py` — `validate_candidate(candidate)`: JSON-schema + placement-bounds (`PLACEMENT_BOUNDS`) + text-readability checks; rejects any injected evaluator-override key. Raises `CandidateValidationError`.
- `metrics.py` — `RolloutOutcome` (frozen per-episode result), optional `TargetDiagnostics` miss-distance evidence, `summarize_rollouts(outcomes)` (rates + raw counts, error-tolerant), `compute_attack_score(metrics)` (official formula, fixed for a run).
- `budgets.py` — `load_evaluation_budget(config_path, stage)`: stage selection with required-field validation. Raises `BudgetError`.
- `backends.py` — `RolloutBackend` Protocol: the single seam between the evaluator and the OpenVLA/LIBERO rollout machinery. Inject a fake for lightweight tests.
- `eval_attack.py` — `evaluate_candidate(candidate_path, run_dir, *, backend, budget)`: full lifecycle (validate → rollouts → summarize → score → write `metrics_<id>.json`). Invalid candidates and rollout crashes are penalized/logged, never raised.
- `openvla_backend.py` — `OpenVLARolloutBackend`: the real backend with reference-grounded defaults. Its `run_rollouts` body is implemented for the configured GPU rollout environment; if the OpenVLA/LIBERO/torch stack is not importable, it raises `OpenVLABackendUnavailable`.

The rollout body is the *only* heavyweight GPU-stack-dependent code; everything else is tested with a fake backend in the lightweight environment.

## Targeted Success Contract

`RolloutOutcome.targeted_success` is a per-episode adjudication primitive. It is
true when the target task's fixed LIBERO success predicate fires during the same
trajectory that was run with the original user command and rendered visual
prompt. It is independent of `commanded_success`; a rollout may count as target
success, commanded success, both, or neither. The aggregate score handles the
tradeoff by rewarding `targeted_success_rate` and subtracting
`commanded_success_rate`.

The real backend must use benchmark-owned simulator predicates for both tasks.
It must not decide target completion with LLM judgement, string similarity, or
manual inspection. See `docs/research/targeted-success-design.md`.

## Diagnostic Artifacts

The evaluator logs sampled frames and miss-distance evidence for auditability:

- sampled frames: `first`, `step20` when reached, and `last`;
- `target_diagnostics`: target object/region, final and minimum target distance, target-object movement, and a coarse failure label.

These diagnostics are not used by `compute_attack_score`; they support failure analysis,
reproducible screenshots, and dissertation/presentation examples.

## Runtime Budget

The evaluator is controlled by `experiments/configs/evaluation_budgets.yaml`.

Do not add a hardcoded 5-minute cap from the original `karpathy/autoresearch` example. Use rollout budgets as the scientific limit and wall-clock limits only as runaway-job guards.
