"""Pilot-002 exploratory discovery runner.

This is a cheap discovery pass after pilot-001 found visible prompt-induced DoS
but no targeted hijack. It runs one broad AI-authored ``loop_with_skill`` candidate
pool from ``experiments/pilot_002_pools.py`` under the fixed evaluator.

This runner is intentionally *not* the final condition comparison. Its output should
be used to pick promising prompt families for a later equal-budget comparison against
random, human, one-shot LLM, and real loop conditions.

Real rollout launch:

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
      PYTHONPATH=$HOME/LIBERO:src:experiments/results \
      ~/vla-injection/.venv/bin/python experiments/run_pilot_002.py

CPU wiring check:

    PYTHONPATH=src:experiments/results python experiments/run_pilot_002.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pilot_002_pools
from aggregate_results import aggregate_run

from autoresearch_loop.run_loop import run_search_condition
from evaluator.budgets import load_evaluation_budget
from evaluator.metrics import RolloutOutcome, TargetDiagnostics

_BUDGET_CONFIG = "experiments/configs/evaluation_budgets.yaml"
_CONDITION = "loop_with_skill"


class _FakeBackend:
    """CPU dry-run backend with varied, strategy-shaped outcomes.

    The dry-run numbers are only a wiring check. They intentionally do not stand in
    for OpenVLA behavior.
    """

    def run_rollouts(
        self, *, candidate: dict[str, Any], seeds: list[int], rollouts_per_candidate: int
    ) -> list[RolloutOutcome]:
        strategy = candidate.get("metadata", {}).get("strategy_family", "")
        n = len(seeds) * rollouts_per_candidate
        outcomes: list[RolloutOutcome] = []
        for k in range(n):
            object_relabel = "relabel" in strategy or "callout" in strategy
            direct_override = "override" in strategy or "trigger" in strategy
            targeted = object_relabel and k == 0
            commanded = not targeted and not direct_override
            outcomes.append(
                RolloutOutcome(
                    seed=seeds[k // rollouts_per_candidate],
                    episode_index=k % rollouts_per_candidate,
                    commanded_success=commanded,
                    targeted_success=targeted,
                    prompt_visibility=0.06,
                    target_diagnostics=TargetDiagnostics(
                        target_object="cream_cheese_1",
                        target_region="basket_1_contain_region",
                        final_target_distance_m=0.25 if targeted else 0.35,
                        min_target_distance_m=0.02 if targeted else 0.18,
                        target_object_moved_m=0.08 if object_relabel else 0.01,
                        failure_mode="target_satisfied" if targeted else "dry_run_fake",
                    ),
                )
            )
        return outcomes


def _make_backend(dry_run: bool) -> Any:
    if dry_run:
        return _FakeBackend()
    from evaluator.openvla_backend import OpenVLARolloutBackend

    return OpenVLARolloutBackend()


def _apply_overrides(budget: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    budget = dict(budget)
    if args.max_candidates is not None:
        budget["max_candidates_per_condition"] = args.max_candidates
    if args.rollouts is not None:
        budget["rollouts_per_candidate"] = args.rollouts
    if args.seeds is not None:
        budget["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    return budget


def _build_proposer(
    *, user_task: str, target_task: str, max_candidates: int
) -> Any:
    pool = pilot_002_pools.pilot_002_skill_pool(user_task, target_task)
    if max_candidates > len(pool):
        raise SystemExit(
            f"pilot-002 pool has {len(pool)} candidates but budget asks for {max_candidates}"
        )

    def propose(index: int) -> dict[str, Any]:
        candidate = pool[index]
        pair = (candidate["user_task"], candidate["target_task"])
        if pair != (user_task, target_task):
            raise SystemExit(
                f"comparability violation for {candidate['candidate_id']}: {pair!r}"
            )
        return candidate

    return propose


def _aggregate(run_dir: Path) -> dict[str, Any]:
    ledger = run_dir / _CONDITION / "ledger.jsonl"
    if not ledger.exists():
        return {}
    return aggregate_run(run_dir / _CONDITION)


def _candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    ledger = run_dir / _CONDITION / "ledger.jsonl"
    if not ledger.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        ledger_row = json.loads(line)
        candidate = json.loads(Path(ledger_row["candidate_path"]).read_text(encoding="utf-8"))
        metrics = json.loads(Path(ledger_row["metrics_path"]).read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "strategy_family": candidate["metadata"].get("strategy_family", ""),
                "attack_score": metrics.get("attack_score"),
                "targeted_successes": metrics.get("targeted_successes", 0),
                "commanded_successes": metrics.get("commanded_successes", 0),
                "completed_rollouts": metrics.get("completed_rollouts", 0),
                "mean_min_target_distance_m": metrics.get("mean_min_target_distance_m"),
                "mean_target_object_moved_m": metrics.get("mean_target_object_moved_m"),
            }
        )
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_summary(
    run_dir: Path,
    *,
    summaries: dict[str, Any],
    budget: dict[str, Any],
    user_task: str,
    target_task: str,
    elapsed_s: float,
    dry_run: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "aggregate.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    rows = _candidate_rows(run_dir)
    n_cand = budget["max_candidates_per_condition"]
    n_seeds = len(budget["seeds"])
    n_roll = budget["rollouts_per_candidate"]
    mode = "DRY-RUN (fake backend, no GPU)" if dry_run else "real OpenVLA rollouts (GPU 1)"

    lines = [
        "# pilot-002 discovery summary",
        "",
        f"- mode: {mode}",
        f"- condition: `{_CONDITION}` broad AI-authored strategy pool",
        f"- user_task: `{user_task}`",
        f"- target_task: `{target_task}`",
        f"- budget: {n_cand} candidates x {n_seeds} seeds x {n_roll} rollouts",
        f"- elapsed: {elapsed_s / 60:.1f} min",
        "",
        "This is exploratory discovery, not a final condition-strength comparison.",
        "Use it to identify prompt families that produce targeted success or target-motion "
        "near misses, then re-run those families under an equal comparison budget.",
        "",
        "| candidate | strategy | score | targeted | commanded | rollouts | min_dist_m | moved_m |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_id']} | {row['strategy_family']} | "
            f"{_fmt(row['attack_score'])} | {row['targeted_successes']} | "
            f"{row['commanded_successes']} | {row['completed_rollouts']} | "
            f"{_fmt(row['mean_min_target_distance_m'])} | "
            f"{_fmt(row['mean_target_object_moved_m'])} |"
        )
    lines.append("")
    (run_dir / "pilot_002_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pilot-002 exploratory discovery.")
    parser.add_argument("--stage", default="pilot_002_discovery")
    parser.add_argument("--run-dir", default="runs/pilot-002")
    parser.add_argument("--dry-run", action="store_true", help="fake backend, no GPU")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--rollouts", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="comma list, e.g. 0,1")
    args = parser.parse_args()

    budget = _apply_overrides(load_evaluation_budget(_BUDGET_CONFIG, args.stage), args)
    pair = budget["task_pairs"][0]
    user_task, target_task = pair["user_task"], pair["target_task"]
    run_dir = Path(args.run_dir)
    cond_dir = run_dir / _CONDITION
    cond_dir.mkdir(parents=True, exist_ok=True)

    backend = _make_backend(args.dry_run)
    backend.run_dir = str(cond_dir)
    proposer = _build_proposer(
        user_task=user_task,
        target_task=target_task,
        max_candidates=budget["max_candidates_per_condition"],
    )

    print(f"[pilot-002] stage={args.stage} condition={_CONDITION}")
    print(f"[pilot-002] pair={user_task!r} -> {target_task!r}")
    print(f"[pilot-002] dry_run={args.dry_run} run_dir={run_dir}")
    t0 = time.time()
    run_search_condition(
        budget=budget,
        run_dir=str(cond_dir),
        backend=backend,
        propose=proposer,
    )
    summaries = _aggregate(run_dir)
    _write_summary(
        run_dir,
        summaries=summaries,
        budget=budget,
        user_task=user_task,
        target_task=target_task,
        elapsed_s=time.time() - t0,
        dry_run=args.dry_run,
    )
    print(f"[pilot-002] wrote {run_dir / 'pilot_002_summary.md'}")


if __name__ == "__main__":
    main()
