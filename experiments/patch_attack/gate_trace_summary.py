"""Aggregate the per-step same-frame gate record into the word-gate mechanism number.

`ce_monitor_patch_attack`'s gated optimiser evaluates BOTH instructions on the SAME composite
image at every step (a free by-product of WP8's condition-blind selection) and writes the result
into each trace row's ``gate`` slot. This module turns those rows into the number the paper needs:

    **how often does ONE patch satisfy BOTH word conditions at once** --
    ``armed_match == 7`` (the target teacher under ``c⊕w``) *and*
    ``dormant_match == 7`` (the clean teacher under ``c``), on identical pixels.

Why this framing and not ``dormant_forced``: ``forced_fraction`` is measured on the *decisive*
dims, which are by definition the dims where the clean and target teachers disagree
(``ce_monitor_patch_attack.py:326``). So a dormant branch that reproduces the clean action scores
``dormant_forced == 0`` *by construction* -- the exact zero is a consequence, not an independent
observation. ``dormant_match`` (against the clean teacher) is the primitive fact, and
``both_perfect_fraction`` is the honest headline.

This is a **derived read of immutable artifacts**: it never re-judges a rollout and never touches
`targeted`/`commanded`. Those verdicts stay with the fixed evaluator (CLAUDE.md's invariant).

Run:

    python experiments/patch_attack/gate_trace_summary.py \
        --traces runs/monitor-stealth/word-gate/stage_c \
        --out runs/monitor-stealth/word-gate/analysis/stage_c_gate_diagnostic.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

#: OpenVLA emits exactly this many discretized action tokens per step.
ACTION_DIM: Final[int] = 7


@dataclass(frozen=True)
class GateStepSummary:
    """Aggregate of the same-frame armed/dormant comparison over a set of steps.

    ``both_perfect_fraction`` is the headline: the fraction of steps on which a single patch
    drove the target under ``c⊕w`` *and* reproduced the clean action under ``c``, exactly.
    """

    n_steps: int
    mean_armed_match: float
    mean_dormant_match: float
    armed_perfect_fraction: float
    dormant_perfect_fraction: float
    both_perfect_fraction: float
    mean_armed_forced: float
    mean_dormant_forced: float
    branches_differ_fraction: float
    decisive_dim_histogram: dict[int, int]


def gate_records(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten a trace into one record per gate-bearing step.

    Rows whose ``gate`` slot is ``None`` come from the ungated path and are dropped rather than
    counted as zeros. ``n_decisive_dims`` lives on the enclosing row, so it is copied in.
    """
    records: list[dict[str, Any]] = []
    for row in trace:
        gate = row.get("gate")
        if not isinstance(gate, Mapping):
            continue
        record = dict(gate)
        record["n_decisive_dims"] = int(row.get("n_decisive_dims", 0))
        records.append(record)
    return records


def _validate_match(record: Mapping[str, Any], key: str) -> int:
    value = int(record[key])
    if not 0 <= value <= ACTION_DIM:
        raise ValueError(f"{key} must be in [0, {ACTION_DIM}], got {value}")
    return value


def summarize_gate_steps(records: Sequence[Mapping[str, Any]]) -> GateStepSummary:
    """Aggregate gate records into a :class:`GateStepSummary`.

    Raises ``ValueError`` on an empty input -- a summary over no steps prints 0.0 everywhere and
    would read as a null result rather than as missing data.
    """
    if not records:
        raise ValueError("no gate-bearing steps to summarise (ungated run, or empty trace)")

    armed = [_validate_match(r, "armed_match") for r in records]
    dormant = [_validate_match(r, "dormant_match") for r in records]
    n = len(records)

    both_perfect = sum(
        1 for a, d in zip(armed, dormant, strict=True) if a == ACTION_DIM and d == ACTION_DIM
    )
    histogram = Counter(int(r["n_decisive_dims"]) for r in records)

    return GateStepSummary(
        n_steps=n,
        mean_armed_match=sum(armed) / n,
        mean_dormant_match=sum(dormant) / n,
        armed_perfect_fraction=sum(1 for a in armed if a == ACTION_DIM) / n,
        dormant_perfect_fraction=sum(1 for d in dormant if d == ACTION_DIM) / n,
        both_perfect_fraction=both_perfect / n,
        mean_armed_forced=sum(float(r["armed_forced"]) for r in records) / n,
        mean_dormant_forced=sum(float(r["dormant_forced"]) for r in records) / n,
        branches_differ_fraction=sum(1 for r in records if r["branches_differ"]) / n,
        decisive_dim_histogram=dict(sorted(histogram.items())),
    )


def summarize_trace_files(paths: Sequence[str]) -> dict[str, Any]:
    """Aggregate every ``trace_*.json`` in ``paths`` into one report with its provenance.

    The report carries the trace basenames so a table in the paper can be traced back to the
    exact episodes it came from. Raises ``ValueError`` if no files are given.
    """
    if not paths:
        raise ValueError("no trace files given")
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(gate_records(json.load(handle)))
    return {
        "n_traces": len(paths),
        "traces": sorted(os.path.basename(p) for p in paths),
        "summary": asdict(summarize_gate_steps(records)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--traces", required=True, help="directory holding trace_*.json")
    parser.add_argument("--out", required=True, help="path for the aggregate JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.traces, "trace_*.json")))
    report = summarize_trace_files(paths)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    summary = report["summary"]
    print(json.dumps(report, indent=2))
    print(
        f"\n[gate-trace] {summary['n_steps']} gate-bearing steps over {report['n_traces']} traces: "
        f"ONE patch satisfied BOTH conditions perfectly on "
        f"{summary['both_perfect_fraction'] * 100:.2f}% of them.",
        flush=True,
    )
    print(f"[gate-trace] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
