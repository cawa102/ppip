"""Hit-rate harness for the adaptive vision-layer hijack (fixed init, varied jitter).

Runs N independent trials of ``adaptive_attack.py`` at a FIXED LIBERO init state
(``ADAPT_SEED``) while varying ONLY the EoT crop-jitter RNG (``ADAPT_TRIAL`` =
0..N-1, each ``torch.manual_seed``'d inside the attack). This isolates the sole
run-to-run randomness so the result is a genuine k/N reliability estimate of the
archived ``targeted_success`` hijack, not a confound of init-state + luck.

Each trial runs as its own continuous subprocess (``ADAPT_CHUNK >=
ADAPT_MAX_STEPS`` => no controller-resetting chunk boundaries). If the host kills
it mid-trajectory (no HIJACK/DONE decision line) the trial resumes from its
checkpoint up to ``MAX_RESUMES`` times; resumes are recorded so the caveat can
travel with the number.

Search-side only: the attack crafts camera perturbations; the FIXED evaluator
(`eval_goal_state` on the target predicate) decides success. No evaluator,
metric, or config file is touched.

Usage (GPU 1 only, per project rule):
  N=12 ~/vla-injection/.venv/bin/python experiments/patch_attack/hijack_hitrate.py
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
OUT_DIR = os.path.join(RUN_DIR, "hitrate")

N = int(os.environ.get("N", "12"))
SEED = os.environ.get("ADAPT_SEED", "0")            # FIXED init-state index (archived success = 0)
MAX_STEPS = os.environ.get("ADAPT_MAX_STEPS", "150")
MAX_RESUMES = int(os.environ.get("MAX_RESUMES", "4"))

# [adapt] HIJACK seed=0 step=122/150 targeted=True min_dist=0.0729 n_miss_this_chunk=2
STATUS_RE = re.compile(
    r"\[adapt\] (HIJACK|DONE|PAUSED) seed=(\S+) step=(\d+)/(\d+) "
    r"targeted=(True|False) min_dist=(\S+)"
)


def child_env(trial: int) -> dict:
    """Recipe env from HIJACK_SUCCESS.md, pinned to GPU 1, with the jitter seed set."""
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
            "ADAPT_SEED": SEED,
            "ADAPT_CHUNK": "200",          # >= MAX_STEPS => one continuous process
            "ADAPT_TRIAL": str(trial),     # determinizes the EoT jitter for this trial
            "CUDA_VISIBLE_DEVICES": "1",    # project rule: GPU 1 ONLY
            "MUJOCO_GL": "egl",
            "PYTHONPATH": (
                f"{HOME}/LIBERO:{HOME}/openvla:{HOME}/autoresearch:{HOME}/autoresearch/src"
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def state_path(trial: int) -> str:
    return os.path.join(RUN_DIR, f"adapt_state_seed{SEED}_trial{trial}.pkl")


def run_once(trial: int, logf) -> tuple[int, re.Match | None]:
    """Run one attack process, tee stdout to logf, return (returncode, last status match)."""
    proc = subprocess.Popen(
        [VENV_PY, ATTACK],
        cwd=REPO,
        env=child_env(trial),
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
    return proc.returncode, last


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, "results.jsonl")
    open(results_path, "w").close()  # fresh
    results: list[dict] = []

    for trial in range(N):
        sp = state_path(trial)
        if os.path.exists(sp):
            os.remove(sp)  # fresh trajectory for this trial
        logpath = os.path.join(OUT_DIR, f"trial{trial:02d}.log")
        outcome: str | None = None
        min_dist: str | None = None
        resumed = 0
        with open(logpath, "w") as logf:
            for attempt in range(MAX_RESUMES + 1):
                _rc, m = run_once(trial, logf)
                if m and m.group(1) in ("HIJACK", "DONE"):
                    outcome = m.group(1)
                    min_dist = m.group(6)
                    break
                # killed / PAUSED before a decision -> resume from checkpoint (state persists)
                if not os.path.exists(sp):
                    break  # died before first save; no checkpoint to resume from
                resumed = attempt + 1

        success = outcome == "HIJACK"
        rec = {
            "trial": trial,
            "outcome": outcome or "KILLED",
            "success": success,
            "min_dist": min_dist,
            "resumed": resumed,
        }
        results.append(rec)
        with open(results_path, "a") as rf:
            rf.write(json.dumps(rec) + "\n")
        k = sum(r["success"] for r in results)
        print(
            f"[hitrate] trial {trial}: {rec['outcome']} min_dist={min_dist} resumed={resumed} "
            f"-> running {k}/{len(results)} hijack",
            flush=True,
        )

    k = sum(r["success"] for r in results)
    decided = sum(1 for r in results if r["outcome"] in ("HIJACK", "DONE"))
    summary = {
        "n_attempted": N,
        "init_state_seed": int(SEED),
        "max_steps": int(MAX_STEPS),
        "hijack": k,
        "denial": decided - k,
        "undecided_killed": N - decided,
        "hit_rate_over_decided": (k / decided) if decided else None,
        "results": results,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as sf:
        json.dump(summary, sf, indent=2)
    pct = (100 * k / decided) if decided else 0
    print(
        f"\n[hitrate] HIT RATE = {k}/{decided} decided ({N} attempted) = {pct:.0f}% hijack "
        f"(init seed={SEED}); summary -> {os.path.join(OUT_DIR, 'summary.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
