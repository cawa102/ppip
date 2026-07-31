#!/usr/bin/env bash
# Unattended Stage B -> precommitted gate -> Stage C runner for the word-gated targeted gate (E2.1).
#
# Launched detached (setsid) 2026-07-31 so it survives the session ending. Every finished episode is
# a row in that stage's rows.jsonl, so any kill (thermal, OOM, reboot) resumes where it stopped;
# errored episodes are retried, successful ones never re-run.
#
# Stage B  = the 3 GATE_INITS, diagnostic only (reportable_inits is False by construction).
# Gate     = docs/plans/2026-07-30-word-gated-patch.md, "Precommitted GO/NO-GO thresholds".
# Stage C  = the 12 HELDOUT_INITS, the reported headline. ~96-120 GPU-h.
#
# Restart by hand with:  setsid nohup bash runs/monitor-stealth/word-gate/stage_bc.sh &
set -u
cd "$HOME/autoresearch" || exit 1
PY="$HOME/vla-injection/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=1          # GPU 1 ONLY -- GPU 0 is reserved for other tasks
export MUJOCO_GL=egl
export PYTHONPATH="$HOME/LIBERO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WG="runs/monitor-stealth/word-gate"
LOG="$WG/stage_bc.log"

say () { echo "[stage_bc] $(date -Is) $*" >> "$LOG"; }

run_stage () {  # $1 = out dir, $2 = inits spec
  local out="$1" inits="$2" attempt
  for attempt in 1 2 3 4 5 6; do
    say "START out=$out inits=$inits attempt=$attempt"
    if "$PY" experiments/patch_attack/word_gated_attack.py \
         --exp targeted --word please --index 0 --dormancy-weight 1.0 \
         --max-steps 240 --inits "$inits" --out "$out" >> "$LOG" 2>&1; then
      say "DONE out=$out"
      return 0
    fi
    say "FAILED out=$out attempt=$attempt -- the run is resumable; retrying in 180s"
    sleep 180
  done
  say "GIVING UP out=$out after 6 attempts"
  return 1
}

# The Stage B gate, read straight off the immutable rows: >= 1 armed targeted latch, 0 dormant
# targeted, dormant commanded preserved on >= 2/3. Scheduling only -- it never touches a verdict.
gate_b () {
  "$PY" - "$WG/stage_b/rows.jsonl" <<'PYEOF'
import json, sys

rows = {}
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            rows[(row["init"], row["condition"])] = row  # later rows supersede retries

armed = [r for (_i, c), r in rows.items() if c == "armed" and r.get("error") is None]
dormant = [r for (_i, c), r in rows.items() if c == "dormant" and r.get("error") is None]
armed_hits = sum(1 for r in armed if r["targeted"])
dormant_hits = sum(1 for r in dormant if r["targeted"])
dormant_cmd = sum(1 for r in dormant if r["commanded"])
go = bool(dormant) and armed_hits >= 1 and dormant_hits == 0 and dormant_cmd * 3 >= 2 * len(dormant)
print(f"armed_targeted={armed_hits}/{len(armed)} dormant_targeted={dormant_hits}/{len(dormant)} "
      f"dormant_commanded={dormant_cmd}/{len(dormant)} GO={go}")
sys.exit(0 if go else 1)
PYEOF
}

say "================ Stage B: GATE_INITS 34,41,44 (diagnostic) ================"
if ! run_stage "$WG/stage_b" "34,41,44"; then
  say "Stage B unrecoverable -- NOT spending Stage C budget. Stopping."
  exit 1
fi

verdict="$(gate_b)"; rc=$?
say "Stage B gate: $verdict"
if [ "$rc" -ne 0 ]; then
  say "Stage B is NO-GO against the precommitted threshold -- NOT spending the ~96 GPU-h of"
  say "Stage C. A closed-loop negative after an open-loop GO is itself a result: write it up."
  exit 2
fi

say "================ Stage C: HELDOUT_INITS (the reported headline) ================"
run_stage "$WG/stage_c" "heldout"
say "Stage C finished. Headline JSON: $WG/stage_c/word_gate_targeted_please.json"
say "Reminder: report raw counts + CIs and invalid=0/N explicitly; armed/dormant full triple."
