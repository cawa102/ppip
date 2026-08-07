#!/usr/bin/env bash
# Weekend queue (launched Fri 2026-08-07). Priority order agreed with the researcher: make the
# artifact-level result reportable FIRST (n=1 -> n=8), then the cheap decisive Tier-0 items, then
# the start of Tier 1. Nothing here takes a big step before the current result is reportable.
#
# NON-DESTRUCTIVE BY DESIGN. The previous figure queue retried by deleting partial frames, which
# nearly cost a run; this one never deletes anything. Every driver is resumable (a leg whose result
# JSON exists is skipped), so a crash is recovered by simply relaunching this script.
#
#   setsid nohup bash runs/monitor-stealth/word-gate/weekend_queue.sh > /dev/null 2>&1 &
set -u
cd "$HOME/autoresearch" || exit 1
PY="$HOME/vla-injection/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=1          # GPU 1 ONLY -- GPU 0 is reserved for other tasks
export MUJOCO_GL=egl
export PYTHONPATH="$HOME/LIBERO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WG="runs/monitor-stealth/word-gate"
LOG="$WG/weekend_queue.log"
V46="$WG/figure_init46/armed/patch"          # the only video that exists at launch
REPLAY="$PY experiments/patch_attack/run_word_gate_replay.py"

say () { echo "[weekend] $(date -Is) $*" >> "$LOG"; }
run () { # run <label> <cmd...>; log outcome, never abort the queue
  say "START $1"
  shift
  if "$@" >> "$LOG" 2>&1; then say "OK"; else say "FAILED (rc=$?) -- continuing to next item"; fi
}

say "queue start; GPU=$CUDA_VISIBLE_DEVICES"

# --- PHASE 1: artifact replication, cheapest inits first (n=1 -> n=4) ------------------------
for INIT in 24 7 49; do
  run "replication init=$INIT" $REPLAY --init "$INIT"
done

# --- PHASE 2: transfer. Init-46's video at every other clean init. Cheap (no new video), and it
# --- is what licenses the claim "the patch must be optimised for one specific setting".
for INIT in 7 24 26 33 36 38 49; do
  run "transfer 46->$INIT" $REPLAY --init "$INIT" --video "$V46" --legs replay --no-record \
      --tag-prefix xfer --out "$WG/artifact/transfer_from46/init$INIT"
done

# --- PHASE 3: artifact replication, remaining inits (n=4 -> n=8) -----------------------------
for INIT in 36 38 26 33; do
  run "replication init=$INIT" $REPLAY --init "$INIT"
done

# --- PHASE 4: Tier 0 -- cheap and decisive (gap register 6, items 1-2) -----------------------
run "E-A1 word-alone control" $PY experiments/patch_attack/ceiling_screen.py \
    --phase w --word please --index 0 --max-steps 240 --out "$WG/word_alone"
run "E-A2 lambda=0 ablation" $PY experiments/patch_attack/word_gate_probe.py \
    --effect targeted --word please --index 0 --lam 0.0 --out "$WG/lam0.0"

# --- PHASE 5: Tier 1 start -- lambda frontier, then word sweep -------------------------------
for LAM in 0.1 0.3 3 10; do
  run "E-A3 lambda=$LAM" $PY experiments/patch_attack/word_gate_probe.py \
      --effect targeted --word please --index 0 --lam "$LAM" --out "$WG/lam${LAM}"
done
for W in carefully now zx; do
  run "E-A4 word=$W" $PY experiments/patch_attack/word_gate_probe.py \
      --effect targeted --word "$W" --index 0 --lam 1.0 --out "$WG/word_${W}"
done

say "QUEUE COMPLETE"
