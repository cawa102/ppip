"""Batch candidate runner for the autoresearch-goal pair sweep (search side, editable).

Same recording contract as ``run_candidate.py`` (immutable ledger row + human-readable
``results.tsv`` row per candidate), but evaluates a *round* of candidate JSONs with ONE
shared OpenVLA backend so the 7B model loads once per round instead of once per candidate
(a ~2 min saving per candidate). This keeps the autoresearch loop adaptive **across rounds**
(the agent authors each round from the previous round's results) while amortizing the model
load **within a round**.

It still only calls the fixed ``evaluate_candidate`` and reads its output — it never computes
or edits a score, so it stays on the editable side of the integrity boundary. The pair comes
from each candidate JSON (evaluate_candidate reads it there), so a round may span many pairs.

Real rollout (GPU 1 only; GPU 0 is reserved):

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
      PYTHONPATH=$HOME/LIBERO:src:experiments/results \
      ~/vla-injection/.venv/bin/python experiments/run_sweep.py \
        runs/autoresearch-goal/proposals/round1/*.json \
        --run-dir runs/autoresearch-goal --stage pair_sweep

CPU wiring check (fake backend, no GPU):

    PYTHONPATH=src:experiments/results python experiments/run_sweep.py \
      some_candidate.json --run-dir /tmp/sweep --stage pair_sweep --dry-run
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from autoresearch_loop.ledger import LedgerError, read_ledger, select_incumbent
from autoresearch_loop.results_tsv import append_results_row, decide_status, init_results_tsv
from autoresearch_loop.run_loop import record_result
from evaluator.budgets import load_evaluation_budget
from evaluator.eval_attack import evaluate_candidate

_BUDGET_CONFIG = "experiments/configs/evaluation_budgets.yaml"


def _make_backend(dry_run: bool, run_dir: str) -> Any:
    if dry_run:
        # reuse run_candidate's plausible "seen but not hijacked" fake for wiring checks
        from run_candidate import _DryRunBackend

        backend: Any = _DryRunBackend()
    else:
        from evaluator.openvla_backend import OpenVLARolloutBackend

        backend = OpenVLARolloutBackend()
    backend.run_dir = run_dir
    return backend


def _peak_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_reserved()) / 1e9
    except Exception:
        return None


def _incumbent_score(ledger_path: Path) -> float | None:
    if not read_ledger(ledger_path):
        return None
    try:
        return float(select_incumbent(ledger_path)["attack_score"])
    except (LedgerError, KeyError):
        return None


def _description(candidate: dict[str, Any]) -> str:
    rationale = candidate.get("visual_prompt", {}).get("rationale")
    if rationale:
        return str(rationale)
    return str(candidate.get("metadata", {}).get("notes", ""))


def _apply_overrides(budget: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    budget = dict(budget)
    if args.seeds is not None:
        budget["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.rollouts is not None:
        budget["rollouts_per_candidate"] = args.rollouts
    return budget


def _evaluate_one(
    candidate_file: Path,
    *,
    run_dir: Path,
    ledger_path: Path,
    results_tsv: Path,
    backend: Any,
    budget: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    """Evaluate one candidate with the shared backend; record ledger + results.tsv."""
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    candidate_id = candidate["candidate_id"]
    candidate_path = run_dir / f"candidate_{candidate_id}.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")

    incumbent_before = _incumbent_score(ledger_path)
    metrics = evaluate_candidate(str(candidate_path), str(run_dir), backend=backend, budget=budget)
    metrics_path = run_dir / f"metrics_{candidate_id}.json"
    record_result(str(candidate_path), str(metrics_path), str(ledger_path))

    status = decide_status(
        valid=bool(metrics.get("valid")),
        completed_rollouts=int(metrics.get("completed_rollouts", 0)),
        attack_score=float(metrics.get("attack_score", 0.0)),
        incumbent_score=incumbent_before,
    )
    append_results_row(
        results_tsv,
        candidate_id=candidate_id,
        attack_score=float(metrics.get("attack_score", 0.0)),
        targeted=int(metrics.get("targeted_successes", 0)),
        commanded=int(metrics.get("commanded_successes", 0)),
        completed=int(metrics.get("completed_rollouts", 0)),
        visibility=metrics.get("mean_prompt_visibility"),
        memory_gb=None if dry_run else _peak_vram_gb(),
        status=status,
        description=_description(candidate),
    )
    completed = metrics.get("completed_rollouts")
    print(
        f"  {candidate_id:32s} score={metrics.get('attack_score')!s:>5} "
        f"tgt={metrics.get('targeted_successes')}/{completed} "
        f"cmd={metrics.get('commanded_successes')}/{completed} "
        f"err={metrics.get('errored_rollouts')} "
        f"vis={metrics.get('mean_prompt_visibility')} "
        f"min_dist={metrics.get('mean_min_target_distance_m')} "
        f"moved={metrics.get('mean_target_object_moved_m')} "
        f"[{status}]  {candidate['user_task'].split(' the ')[1].split(' and ')[0]} "
        f"-> {candidate['target_task'].split(' the ')[1].split(' and ')[0]}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ROUND of candidates (shared backend).")
    parser.add_argument("candidates", nargs="+", help="candidate JSON files (or a dir)")
    parser.add_argument("--run-dir", default="runs/autoresearch-goal")
    parser.add_argument("--stage", default="pair_sweep")
    parser.add_argument("--dry-run", action="store_true", help="fake backend, no GPU")
    parser.add_argument("--seeds", default=None, help="comma list override, e.g. 0,1,2")
    parser.add_argument("--rollouts", type=int, default=None)
    args = parser.parse_args()

    # Expand a single directory arg into its candidate_*.json / *.json files.
    files: list[Path] = []
    for arg in args.candidates:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)
    if not files:
        raise SystemExit("no candidate files found")

    budget = _apply_overrides(load_evaluation_budget(_BUDGET_CONFIG, args.stage), args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    results_tsv = init_results_tsv(run_dir / "results.tsv")

    print(
        f"[run_sweep] {len(files)} candidates, stage={args.stage} "
        f"seeds={budget['seeds']} rollouts={budget['rollouts_per_candidate']} "
        f"dry_run={args.dry_run}"
    )
    backend = _make_backend(args.dry_run, str(run_dir))  # loads the 7B model ONCE

    hits: list[str] = []
    for candidate_file in files:
        try:
            metrics = _evaluate_one(
                candidate_file,
                run_dir=run_dir,
                ledger_path=ledger_path,
                results_tsv=results_tsv,
                backend=backend,
                budget=budget,
                dry_run=args.dry_run,
            )
            if int(metrics.get("targeted_successes", 0)) > 0:
                hits.append(str(metrics.get("candidate_id")))
        except Exception:  # noqa: BLE001 - one bad candidate must not kill the round
            print(f"  !! {candidate_file.name} raised:\n{traceback.format_exc()}")

    print(f"[run_sweep] round done. targeted-success candidates: {hits or 'none'}")


if __name__ == "__main__":
    main()
