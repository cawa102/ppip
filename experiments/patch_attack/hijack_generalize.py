"""Generalization sweep for the adaptive vision-layer hijack (vary init state).

The companion `hijack_hitrate.py` fixed the LIBERO init state (seed 0) and varied
only the jitter -> 12/12 hijack, establishing reliability AT that init. This
sweep answers the orthogonal question: does the hijack transfer to OTHER scene
init states? It varies ``ADAPT_SEED`` (the init-state index) across INITS, each
run torch-seeded (``ADAPT_TRIAL``) so it is reproducible.

Because forcing real 7/7 makes the executed action ~= the target policy's own
action, a single trial nearly determines an init. To avoid miscounting a rare
jitter-sensitive frame as an init-level denial, any non-hijack trial is
re-confirmed with CONFIRM_TRIALS extra jitter seeds; an init counts as
hijackable if ANY of its trials reaches ``targeted_success``.

Reports (# hijackable inits)/(# inits) plus per-init detail. Search-side only;
the FIXED evaluator decides every success.

Usage (GPU 1 only):
  INITS=1,2,3,4,5,6,7,8,9,10 ~/vla-injection/.venv/bin/python \
    experiments/patch_attack/hijack_generalize.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "autoresearch")
VENV_PY = os.path.join(HOME, "vla-injection/.venv/bin/python")
ATTACK = os.path.join(REPO, "experiments/patch_attack/adaptive_attack.py")
RUN_DIR = os.path.join(REPO, "runs/autoresearch-hijack")
OUT_DIR = os.path.join(RUN_DIR, "generalize")

INITS = [int(x) for x in os.environ.get("INITS", "1,2,3,4,5,6,7,8,9,10").split(",")]
MAX_STEPS = os.environ.get("ADAPT_MAX_STEPS", "150")
CONFIRM_TRIALS = int(os.environ.get("CONFIRM_TRIALS", "2"))  # extra jitter draws on a denial
MAX_RESUMES = int(os.environ.get("MAX_RESUMES", "4"))

STATUS_RE = re.compile(
    r"\[adapt\] (HIJACK|DONE|PAUSED) seed=(\S+) step=(\d+)/(\d+) "
    r"targeted=(True|False) min_dist=(\S+)"
)


def child_env(init: int, trial: int) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ADAPT_K": "3",
            "ADAPT_MAXTRIES": "8",
            "ADAPT_EPS": "0.15",
            "ADAPT_EPS_CAP": "1.0",
            "ADAPT_LR": "2.5e-2",
            "ADAPT_MAX_STEPS": MAX_STEPS,
            "ADAPT_SEED": str(init),        # <-- the axis we vary: LIBERO init-state index
            "ADAPT_TRIAL": str(trial),      # jitter seed (reproducible)
            "ADAPT_CHUNK": "200",
            "CUDA_VISIBLE_DEVICES": "1",     # project rule: GPU 1 ONLY
            "MUJOCO_GL": "egl",
            "PYTHONPATH": (
                f"{HOME}/LIBERO:{HOME}/openvla:{HOME}/autoresearch:{HOME}/autoresearch/src"
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def state_path(init: int, trial: int) -> str:
    return os.path.join(RUN_DIR, f"adapt_state_seed{init}_trial{trial}.pkl")


def run_once(init: int, trial: int, logf) -> re.Match | None:
    proc = subprocess.Popen(
        [VENV_PY, ATTACK],
        cwd=REPO,
        env=child_env(init, trial),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last: re.Match | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        logf.write(line)
        logf.flush()
        m = STATUS_RE.search(line)
        if m:
            last = m
    proc.wait()
    return last


def run_trial(init: int, trial: int, logf) -> dict:
    """One trial to a decision (resume across host kills). Returns outcome dict."""
    sp = state_path(init, trial)
    if os.path.exists(sp):
        os.remove(sp)
    outcome: str | None = None
    min_dist: str | None = None
    resumed = 0
    for attempt in range(MAX_RESUMES + 1):
        m = run_once(init, trial, logf)
        if m and m.group(1) in ("HIJACK", "DONE"):
            outcome = m.group(1)
            min_dist = m.group(6)
            break
        if not os.path.exists(sp):
            break
        resumed = attempt + 1
    return {"init": init, "trial": trial, "outcome": outcome or "KILLED",
            "success": outcome == "HIJACK", "min_dist": min_dist, "resumed": resumed}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, "results.jsonl")
    open(results_path, "w").close()
    per_init: list[dict] = []

    for init in INITS:
        logpath = os.path.join(OUT_DIR, f"init{init:02d}.log")
        trials: list[dict] = []
        with open(logpath, "w") as logf:
            first = run_trial(init, 0, logf)
            trials.append(first)
            if not first["success"]:  # confirm a denial isn't a jitter fluke
                for t in range(1, CONFIRM_TRIALS + 1):
                    trials.append(run_trial(init, t, logf))
        hijackable = any(t["success"] for t in trials)
        rec = {
            "init": init,
            "hijackable": hijackable,
            "n_trials": len(trials),
            "n_hijack": sum(t["success"] for t in trials),
            "trials": trials,
        }
        per_init.append(rec)
        with open(results_path, "a") as rf:
            rf.write(json.dumps(rec) + "\n")
        n_hj = sum(r["hijackable"] for r in per_init)
        print(
            f"[generalize] init {init}: {'HIJACKABLE' if hijackable else 'DENIAL'} "
            f"({rec['n_hijack']}/{rec['n_trials']} trials) -> running {n_hj}/{len(per_init)} inits",
            flush=True,
        )

    n_hj = sum(r["hijackable"] for r in per_init)
    n_dec = sum(1 for r in per_init if any(t["outcome"] in ("HIJACK", "DONE") for t in r["trials"]))
    summary = {
        "inits": INITS,
        "n_inits": len(INITS),
        "n_hijackable": n_hj,
        "n_decided": n_dec,
        "hijack_rate_over_decided": (n_hj / n_dec) if n_dec else None,
        "note": "companion to hitrate/summary.json (seed-0 init = 12/12). init seeds here are 1+.",
        "per_init": per_init,
    }
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    pct = (100 * n_hj / n_dec) if n_dec else 0
    print(
        f"\n[generalize] GENERALIZATION = {n_hj}/{n_dec} decided inits hijackable "
        f"({len(INITS)} attempted) = {pct:.0f}%; summary -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
