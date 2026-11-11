#!/bin/bash
# Unattended chain for the faithful t59 evaluation (RTX 4090 pod). Never stops the pod.
#   1. wait for the faithful reference (seed 0), train the faithful task proxy (seed 0)
#   2. Step 2 checks (check_faithful.py); on failure write CHECK_FAIL and stop
#   3. stop the early (--no-faithful) server's workers cleanly, start 2 full servers x 3 workers on the queue
#   4. grow the drop bank as the source trials finish; cosine gate after 20 pps_f_g04 normal trials
#   5. when the queue is done: summarize
set -u
WT=/workspace/pps-selftrain-wt
ROOT=/workspace/runs/eval_t59_faithful
LOG=$ROOT/logs
REF=/workspace/runs/reference_t59_faithful_s0
TASK=/workspace/runs/task_t59_faithful_s0
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
source /workspace/env.sh
cd $WT
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/chain.log; }

log "waiting for $REF/final.pt"
while [ ! -f $REF/final.pt ] || pgrep -f "[d]istill_faithful.py --out $REF" > /dev/null; do sleep 20; done
if [ ! -f $TASK/final.pt ]; then
  log "training task proxy"
  PYTHONPATH=. $PY scripts/faithful/train_task_faithful.py --init $REF/final.pt --out $TASK > $LOG/train_task_s0.log 2>&1
fi
[ -f $TASK/final.pt ] || { log "task training failed"; touch $ROOT/TRAIN_FAIL; exit 1; }
log "task proxy done"

log "step 2 checks"
PYTHONPATH=. $PY scripts/faithful/check_faithful.py --out $ROOT/checks/check_faithful.json > $LOG/check_faithful.log 2>&1
if [ $? -ne 0 ]; then log "CHECK FAILED"; touch $ROOT/CHECK_FAIL; exit 1; fi
log "checks passed"

# drain the early server (port 8022) if it is running
if pgrep -f "[s]erve_faithful.py --port 8022" > /dev/null; then
  touch $ROOT/STOP_8022 $ROOT/repro_check/STOP_8022
  while pgrep -f "[r]un_faithful.py worker --port 8022" > /dev/null; do sleep 5; done
  pkill -f "[s]erve_faithful.py --port 8022"; sleep 10
  log "early server drained"
fi

for port in 8020 8021; do
  PYTHONPATH=. nohup $PY scripts/faithful/serve_faithful.py --port $port > $LOG/server_$port.log 2>&1 &
  sleep 60
done
for port in 8020 8021; do
  until grep -q '"gpu"' $LOG/server_$port.log; do sleep 5; done
done
sleep 20
log "servers up"
if [ ! -f $ROOT/repro_check/REUSE_OK ] && [ ! -f $ROOT/repro_check/REUSE_FAIL ]; then
  # reproduction check was not run early: run it first on 8020
  PYTHONPATH=.:/workspace/openpi/third_party/libero $LPY scripts/faithful/run_faithful.py worker --port 8020 \
      --out $ROOT/repro_check > $LOG/worker_repro.log 2>&1
  $PY scripts/faithful/compare_repro.py >> $LOG/chain.log 2>&1
fi
n=0
for port in 8020 8021; do
  for k in 1 2 3; do
    n=$((n+1))
    PYTHONPATH=.:/workspace/openpi/third_party/libero nohup $LPY scripts/faithful/run_faithful.py worker --port $port \
        --out $ROOT > $LOG/worker_${port}_$k.log 2>&1 &
    sleep 3
  done
done
log "workers up: $n"

COS_DONE=0
while true; do
  if [ -f $ROOT/bank/manifest.json ] && [ $(python3 -c "import json;print(len(json.load(open('$ROOT/bank/manifest.json'))['states']))") -lt 120 ]; then
    python3 scripts/faithful/build_bank.py >> $LOG/bank.log 2>&1 && { rm -f $ROOT/MORE; log "bank complete, MORE removed"; }
  fi
  if [ $COS_DONE -eq 0 ] && [ $(ls $ROOT/pps_f_g04/paired/trial_00{0,1}[0-9].npz 2>/dev/null | wc -l) -ge 20 ]; then
    c=$($PY -c "
import numpy as np
v=np.concatenate([np.load('$ROOT/pps_f_g04/paired/trial_%04d.npz'%t)['pps_ref_base_real'].ravel() for t in range(20)])
print('%.4f'%v.mean())")
    log "cosine check (ref vs base, real dims, steered path, 20 trials): $c"
    echo "$c" > $ROOT/checks/cosine_20trials.txt
    if python3 -c "import sys; sys.exit(0 if float('$c') >= 0.85 else 1)"; then COS_DONE=1; else
      log "COSINE BELOW 0.85: stopping"; touch $ROOT/STOP $ROOT/COSINE_FAIL; exit 1; fi
  fi
  if ! pgrep -f "[r]un_faithful.py worker" > /dev/null; then break; fi
  sleep 60
done
log "workers finished"
for port in 8020 8021; do pkill -f "[s]erve_faithful.py --port $port"; done
$PY scripts/faithful/summarize_faithful.py > $LOG/summarize.log 2>&1
touch $ROOT/DONE
log "done"
