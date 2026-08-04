#!/usr/bin/env bash
# ASR for the eps=0.06 stealth patch: 5 precommitted HELD-OUT inits, identical effort.
#
# Effort is pinned to the escalated config (k=30, maxtries=10, restarts=3) that the free-range
# positive required at this cell -- at default effort the free-range attack scored
# targeted=False here, so a weaker budget would manufacture a false negative and the rate
# would measure search effort rather than the stealth constraint.
#
# Inits are the first five of shared_inits.HELDOUT_INITS in their declared order (no
# cherry-picking); init 0 is excluded by the precommit as selection-contaminated.
#
# Long GPU jobs on this host have been killed non-deterministically, so each seed is retried;
# run_confined_episode checkpoints env state, so a retry resumes rather than restarting.
set -u

SEEDS="${MC_ASR_SEEDS:-4 7 22 24 26}"
EPS="${MC_ASR_EPS:-0.06}"
GPU="${MC_ASR_GPU:-0}"
RUN_DIR="$HOME/autoresearch/runs/monitor-stealth/asr_eps${EPS}"
LOG_DIR="$RUN_DIR/logs"
MAX_TRIES=3

mkdir -p "$LOG_DIR"
echo "[asr] seeds=$SEEDS eps=$EPS gpu=$GPU -> $RUN_DIR"

for seed in $SEEDS; do
  result="$RUN_DIR/result_corner_BL_64_seed${seed}_stealth_asr_trial0.json"
  if [ -f "$result" ]; then
    echo "[asr] seed $seed already complete, skipping"
    continue
  fi
  for attempt in $(seq 1 $MAX_TRIES); do
    echo "[asr] seed $seed attempt $attempt $(date '+%F %T')"
    MC_CORNER=BL MC_SIZE=64 MC_SEED="$seed" MC_MAX_STEPS=220 \
      MC_RUN_DIR="$RUN_DIR" \
      MC_STEALTH_BASE=aurora MC_STEALTH_EPS="$EPS" \
      MC_K=30 MC_MAXTRIES=10 MC_RESTARTS=3 \
      MC_TAG_SUFFIX=_stealth_asr MC_RECORD=1 \
      CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl PYTHONPATH="$HOME/LIBERO" \
      "$HOME/vla-injection/.venv/bin/python" \
        "$HOME/autoresearch/experiments/patch_attack/corner_attack.py" \
        >> "$LOG_DIR/seed${seed}.log" 2>&1
    status=$?
    if [ -f "$result" ]; then
      echo "[asr] seed $seed DONE (exit $status)"
      break
    fi
    echo "[asr] seed $seed attempt $attempt failed (exit $status); retrying"
  done
done

echo "[asr] all seeds finished $(date '+%F %T')"
