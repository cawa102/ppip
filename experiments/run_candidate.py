"""Single-candidate runner — the ``uv run train.py`` analog (search side, agent-editable).

Ported from ``karpathy/autoresearch``: one experiment = evaluate ONE candidate JSON with the
fixed evaluator, append an immutable ledger row + a human-readable ``results.tsv`` row, and
print an ``attack_score`` summary for the human-in-the-loop to decide keep/discard. It never
computes or edits a score itself — it only calls the fixed ``evaluate_candidate`` and reads
its output, so it stays on the editable side of the integrity boundary.

Real rollout (GPU 1 only; GPU 0 is reserved):

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
      PYTHONPATH=$HOME/LIBERO:src:experiments/results \
      ~/vla-injection/.venv/bin/python experiments/run_candidate.py \
        path/to/candidate.json --run-dir runs/autoresearch-jul3 --stage pilot_002_discovery

CPU wiring check (no GPU, fake backend):

    PYTHONPATH=src:experiments/results python experiments/run_candidate.py \
      path/to/candidate.json --run-dir /tmp/rc --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autoresearch_loop.ledger import LedgerError, read_ledger, select_incumbent
from autoresearch_loop.results_tsv import append_results_row, decide_status, init_results_tsv
from autoresearch_loop.run_loop import record_result
from evaluator.budgets import load_evaluation_budget
from evaluator.eval_attack import evaluate_candidate
from evaluator.metrics import RolloutOutcome, TargetDiagnostics

_BUDGET_CONFIG = "experiments/configs/evaluation_budgets.yaml"


class _DryRunBackend:
    """CPU wiring backend: a plausible 'seen but not hijacked' baseline outcome.

    Not a stand-in for OpenVLA behavior; it only exercises the runner end to end off-GPU.
    """

    def run_rollouts(
        self, *, candidate: dict[str, Any], seeds: list[int], rollouts_per_candidate: int
    ) -> list[RolloutOutcome]:
        n = len(seeds) * rollouts_per_candidate
        return [
            RolloutOutcome(
                seed=seeds[i // rollouts_per_candidate],
                episode_index=i % rollouts_per_candidate,
                commanded_success=True,
                targeted_success=False,
                prompt_visibility=0.05,
                target_diagnostics=TargetDiagnostics(
                    target_object="cream_cheese_1",
                    target_region="basket_1_contain_region",
                    failure_mode="dry_run_fake",
                ),
            )
            for i in range(n)
        ]


def _make_backend(dry_run: bool, run_dir: str) -> Any:
    if dry_run:
        backend: Any = _DryRunBackend()
    else:
        from evaluator.openvla_backend import OpenVLARolloutBackend

        backend = OpenVLARolloutBackend()
    backend.run_dir = run_dir
    return backend


def _peak_vram_gb() -> float | None:
    """Best-effort peak reserved VRAM in GB (None off-GPU); never fatal to the run."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_reserved()) / 1e9
    except Exception:
        return None


def _incumbent_score(ledger_path: Path) -> float | None:
    """The best attack_score recorded so far (None on an empty/unreadable ledger)."""
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


def _print_summary(
    metrics: dict[str, Any], status: str, incumbent_before: float | None, metrics_path: Path
) -> None:
    completed = metrics.get("completed_rollouts")
    print("---")
    print(f"candidate:      {metrics.get('candidate_id')}")
    print(f"attack_score:   {metrics.get('attack_score')}")
    print(f"targeted:       {metrics.get('targeted_successes')}/{completed}")
    print(f"commanded:      {metrics.get('commanded_successes')}/{completed}")
    print(f"errored:        {metrics.get('errored_rollouts')}")
    print(f"visibility:     {metrics.get('mean_prompt_visibility')}")
    print(f"valid:          {metrics.get('valid')}  error={metrics.get('error')}")
    print(f"incumbent_prev: {incumbent_before}")
    print(f"status:         {status.upper()}")
    print(f"metrics:        {metrics_path}")
    if status == "crash":
        print("note: crash (invalid or no completed rollout) — inspect the run log first.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONE candidate (autoresearch loop step).")
    parser.add_argument("candidate", help="path to the candidate JSON to evaluate")
    parser.add_argument("--run-dir", default="runs/autoresearch-jul3")
    parser.add_argument("--stage", default="pilot_002_discovery")
    parser.add_argument("--dry-run", action="store_true", help="fake backend, no GPU")
    parser.add_argument("--seeds", default=None, help="comma list override, e.g. 0,1")
    parser.add_argument("--rollouts", type=int, default=None)
    args = parser.parse_args()

    budget = _apply_overrides(load_evaluation_budget(_BUDGET_CONFIG, args.stage), args)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    results_tsv = init_results_tsv(run_dir / "results.tsv")

    # Canonicalize the candidate into the run dir before evaluation, so the ledger row and
    # the metrics file sit beside the exact candidate that produced them.
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    candidate_id = candidate["candidate_id"]
    candidate_path = run_dir / f"candidate_{candidate_id}.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")

    incumbent_before = _incumbent_score(ledger_path)

    backend = _make_backend(args.dry_run, str(run_dir))
    print(
        f"[run_candidate] {candidate_id}: "
        f"{candidate['user_task']!r} -> {candidate['target_task']!r}"
    )
    print(
        f"[run_candidate] stage={args.stage} seeds={budget['seeds']} "
        f"rollouts={budget['rollouts_per_candidate']} dry_run={args.dry_run}"
    )

    metrics = evaluate_candidate(
        str(candidate_path), str(run_dir), backend=backend, budget=budget
    )
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
        memory_gb=None if args.dry_run else _peak_vram_gb(),
        status=status,
        description=_description(candidate),
    )
    _print_summary(metrics, status, incumbent_before, metrics_path)


if __name__ == "__main__":
    main()
