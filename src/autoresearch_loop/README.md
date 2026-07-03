# Autoresearch Loop

This directory will contain the candidate proposal loop inspired by `karpathy/autoresearch`.

The loop should:

- read previous candidate results;
- propose the next valid candidate;
- call the fixed evaluator;
- append immutable entries to a run ledger;
- stop at the configured budget.

Keep provider-specific code isolated so Claude, Codex, manual one-shot, and random-search conditions can share the same evaluator.

## Modules (implemented, GPU-independent)

- `ledger.py` — `append_ledger_row` (append-only; rejects duplicate `candidate_id`), `read_ledger`, `select_incumbent` (best under official score). Raises `LedgerError`.
- `run_loop.py` — `record_result(candidate_path, metrics_path, ledger_path)` (immutable row) and `run_search_condition(*, budget, run_dir, backend, propose)`: budget-bounded and **resumable from the ledger** (already-recorded candidates are skipped). `propose(index)` keeps the loop provider-agnostic.
- `candidate_writer.py` — `generate_random_candidate(...)` (the always-valid `random_search` baseline, deterministic from its RNG) and `write_candidate`.
- `memory.py` — `summarize_history(ledger_path)`: read-only summary of prior candidates/failures (never edits evaluator outputs).
- `conditions.py` — `load_search_conditions(config_path)`: loads/validates the six comparable conditions under one shared `budget_stage`. Raises `ConditionsError`.

## Difference From karpathy/autoresearch

Do not copy the original 5-minute experiment cap. OpenVLA+LIBERO rollout jobs can be long-running.

Do not expect a literal `train.py` file in the first implementation. The `train.py` role is split across:

- `src/autoresearch_loop/run_loop.py` — orchestrates candidate proposal and evaluation jobs.
- `src/autoresearch_loop/candidate_writer.py` — writes candidate JSON files.
- `src/autoresearch_loop/memory.py` — summarizes prior attempts and failures.
- `experiments/candidates/` — stores generated candidate artifacts.

The fixed `prepare.py` role lives outside this directory in `src/evaluator/`, `src/rendering/`, and `experiments/configs/evaluation_budgets.yaml`.

The loop should be resumable:

1. write a candidate;
2. submit or run the evaluator;
3. wait for metrics to appear;
4. append a ledger row;
5. exit or continue depending on the configured candidate budget.

The LLM/coding agent does not need to remain active while the evaluator is running.
