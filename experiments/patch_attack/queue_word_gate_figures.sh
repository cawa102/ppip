#!/usr/bin/env bash
# Queue the remaining word-gate figure renders (init 24 and 7) behind the in-flight init-46 run,
# then build every GIF whose frames exist.
#
# Only one 7B policy fits on the shared card, so the renders are strictly sequential. Launched
# detached (setsid) so it survives the session ending:
#
#   setsid nohup bash runs/monitor-stealth/word-gate/queue_figures.sh &
set -u
cd "$HOME/autoresearch" || exit 1
PY="$HOME/vla-injection/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=1          # GPU 1 ONLY -- GPU 0 is reserved for other tasks
export MUJOCO_GL=egl
export PYTHONPATH="$HOME/LIBERO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WG="runs/monitor-stealth/word-gate"
LOG="$WG/queue_figures.log"
QUEUE="24 7"
ALL="46 24 7"

say () { echo "[figqueue] $(date -Is) $*" >> "$LOG"; }

say "waiting for the in-flight init-46 render to release the card"
while pgrep -f "render_figure_init46.py" > /dev/null; do sleep 60; done
say "card free -- starting queued renders: $QUEUE"

for init in $QUEUE; do
  for attempt in 1 2 3; do
    # A retry must not inherit frames from the killed attempt: a longer previous run would leave
    # stale high-numbered frames that the GIF would happily animate past the real ending.
    if [ "$attempt" -gt 1 ]; then
      say "clearing partial frames for init=$init before retry"
      rm -rf "$WG/figure_init$init/armed" "$WG/figure_init$init/dormant"
    fi
    say "render init=$init attempt=$attempt"
    if "$PY" experiments/patch_attack/render_word_gate_figure.py "$init" >> "$LOG" 2>&1; then
      say "render init=$init DONE"
      break
    fi
    say "render init=$init FAILED attempt=$attempt -- retrying in 180s"
    sleep 180
  done
done

for init in $ALL; do
  if [ -d "$WG/figure_init$init/armed/policy_input" ]; then
    say "building GIF init=$init"
    if "$PY" experiments/patch_attack/make_word_gate_gif.py "$WG/figure_init$init" "$init" \
         >> "$LOG" 2>&1; then
      say "GIF init=$init OK"
    else
      say "GIF init=$init FAILED (pairing check or missing frames -- see above)"
    fi
  else
    say "GIF init=$init SKIPPED -- no frames"
  fi
done
say "queue complete"
