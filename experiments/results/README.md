# Experiment Results

Aggregate summaries for reporting and metric-gaming checks. Heavy raw outputs live under
`runs/<run_id>/` (git-ignored); only summaries are tracked.

## Aggregator

`aggregate_results.py` rolls per-candidate `metrics_<id>.json` (via each run's
`ledger.jsonl`) up into per-condition summaries:

- `aggregate_run(run_dir)` → `{condition: summary}` for one ledger.
- `aggregate_condition(records)` → one condition summary.

Each summary reports **raw counts alongside rates** (`total_targeted_successes`,
`total_commanded_successes`, `total_completed_rollouts`, …) so the official
`attack_score = targeted_rate − commanded_rate − 0.05·invalid_rate` is always
recomputable from saved metrics. Partial/failed runs are tolerated (invalid or errored
candidates are counted, not dropped).

## Pilot-001 artifacts

The pilot orchestrator (`experiments/run_pilot.py`) runs each condition into its own
`runs/pilot-001/<condition>/` directory, then auto-writes on completion:

- `runs/pilot-001/pilot_summary.md` — per-condition comparison table (candidate/valid
  counts, best & mean `attack_score`, raw targeted vs commanded counts) + the loop
  stand-in caveat. **Tracked.**
- `runs/pilot-001/aggregate.json` — the machine-readable `{condition: summary}`. **Tracked.**

See `runs/pilot-001/README.md` for the pilot's configuration, conditions, and the
comparability invariant.

## Failure-mode analysis

For qualitative failure analysis, read the non-scoring `target_diagnostics` in each
`runs/<run_id>/<condition>/candidates/<id>/rollouts.jsonl` and the aggregate
miss-distance fields (`mean_miss_*`). Keep those diagnostic fields **separate** from
`attack_score`, which remains the only official search objective.
