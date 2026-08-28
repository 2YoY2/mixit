#!/usr/bin/env bash
# Pipeline watchdog. Phase 1: watch the 75 Hz pretrain's VAL lines and stop
# the run once validation stops improving (>=3 evals past the last best,
# minimum 6 evals seen) -- the HOURS cap is the backstop. Phase 2: wait for
# both PerceptAlign preps. Phase 3: archive best.pt, then run the three PA
# arms sequentially on the GPU:
#   arm1  PA-200 from scratch          (the null hypothesis)
#   arm2  PA-75  warm-started, WIN=256 (matched-rate transfer)
#   arm3  PA-200 warm-started          (mis-scaled warm start)
# Logs: $D/log_arm{1,2,3}.txt. After arm3: run eval_xpred on prep_pa{200,75}.
set -u
cd "$(dirname "$0")/.."
D=$HOME/zerdani/buffer/octonet
source ~/zerdani/phase1/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "[watchdog $(date +%H:%M:%S)] $*"; }

log "phase 1: watching pretrain VAL for stagnation"
while pgrep -f train_xpred >/dev/null; do
  read -r tot since <<< "$(awk '/VAL/{i++; if(/best/) l=i} END{print i+0, i-l+0}' \
                           "$D/log_xpred75.txt" 2>/dev/null || echo 0 0)"
  if [ "${tot:-0}" -ge 6 ] && [ "${since:-0}" -ge 3 ]; then
    log "VAL stagnant ($since evals past best, $tot total) -> stopping pretrain"
    pkill -f train_xpred; sleep 5; break
  fi
  sleep 60
done
log "pretrain over"

log "phase 2: waiting for PA preps"
until grep -aq "kept /" "$D/log_prep_pa400.txt" 2>/dev/null \
   && grep -aq "kept /" "$D/log_prep_pa75.txt" 2>/dev/null; do sleep 60; done
log "preps done"
mkdir -p "$D/archive"
cp "$D/xpred75_runs/best.pt" "$D/archive/xpred75_best_$(date +%Y%m%d).pt" \
  && log "pretrain best.pt archived"

log "phase 3: arm1 PA-400 scratch"
PREP_OUT=$D/prep_pa400 MIXIT_RUNS=$D/pa400_scratch_runs WIN=1024 STEPS=40000 HOURS=2 \
  python3 train/train_xpred.py > "$D/log_arm1.txt" 2>&1
log "arm1 done"

log "phase 3: arm2 PA-75 warm-start"
PREP_OUT=$D/prep_pa75 MIXIT_RUNS=$D/pa75_warm_runs INIT=$D/xpred75_runs/best.pt \
  WIN=256 STEPS=30000 HOURS=2 python3 train/train_xpred.py > "$D/log_arm2.txt" 2>&1
log "arm2 done"

log "phase 3: arm3 PA-400 warm-start"
PREP_OUT=$D/prep_pa400 MIXIT_RUNS=$D/pa400_warm_runs INIT=$D/xpred75_runs/best.pt \
  WIN=1024 STEPS=30000 HOURS=2 python3 train/train_xpred.py > "$D/log_arm3.txt" 2>&1
log "arm3 done -- all arms trained; next: eval_xpred on prep_pa400 and prep_pa75"
