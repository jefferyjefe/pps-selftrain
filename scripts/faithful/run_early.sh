#!/bin/bash
# Early phase: while the faithful task proxy trains (host-bound, GPU mostly idle), serve pi0.5 and v3 ours on one
# --no-faithful server (port 8022) for the jobs that do not need the faithful proxies: the pi0.5 reproduction check,
# then (main queue, cells base / pps_v3_g04 / task_v3) pi0.5 on the harder starts and base / v3 on bank states 0-39.
# run_chain.sh drains this server (STOP_8022) before it starts the two full servers.
set -u
WT=/workspace/pps-selftrain-wt
ROOT=/workspace/runs/eval_t59_faithful
LOG=$ROOT/logs
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
TASK_DONE=/workspace/runs/task_t59_faithful_s0/final.pt
source /workspace/env.sh
cd $WT
log() { echo "$(date -u +%FT%TZ) early: $*" | tee -a $LOG/chain.log; }
until grep -q "^step 1000 " $LOG/train_task_s0.log 2>/dev/null; do sleep 30; done
for i in $(seq 1 90); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 9500 ] && break
  [ -f $TASK_DONE ] && { log "task training finished before an early server could start"; exit 0; }
  sleep 20
done
log "task training in steady state, GPU free ${free} MiB"
if [ "$free" -lt 9500 ]; then log "not enough GPU memory for an early server, skipping"; exit 0; fi
PYTHONPATH=. nohup $PY scripts/faithful/serve_faithful.py --port 8022 --no-faithful > $LOG/server_8022.log 2>&1 &
until grep -q '"gpu"' $LOG/server_8022.log; do sleep 5; if ! pgrep -f "[s]erve_faithful.py --port 8022" >/dev/null; then log "early server died"; exit 1; fi; done
sleep 15
log "early server up"
PYTHONPATH=.:/workspace/openpi/third_party/libero $LPY scripts/faithful/run_faithful.py worker --port 8022 \
    --out $ROOT/repro_check > $LOG/worker_repro.log 2>&1
$PY scripts/faithful/compare_repro.py > $LOG/compare_repro.log 2>&1
log "repro: $(tail -1 $LOG/compare_repro.log)"
for k in 1 2 3; do
  PYTHONPATH=.:/workspace/openpi/third_party/libero nohup $LPY scripts/faithful/run_faithful.py worker --port 8022 \
      --out $ROOT --cells base,pps_v3_g04,task_v3 > $LOG/worker_8022_$k.log 2>&1 &
  sleep 3
done
log "early workers up"
