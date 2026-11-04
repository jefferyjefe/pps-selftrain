#!/bin/bash
# Experiment 1 eval. pi05_libero on 5 held-out LIBERO-90 tasks, 30 episodes each (init states 0..29, seed 7).
# Task t runs on its own server, split over 3 clients of 10 episodes.
source /workspace/env.sh
export PYTHONPATH=/workspace/openpi/third_party/libero
PY=/workspace/openpi/examples/libero/.venv/bin/python
OUT=/workspace/runs/m0/exp1_eval; mkdir -p $OUT/logs
TASKS=(28 30 59 64 38); PORTS=(8000 8002 8003 8004 8005)
for i in 0 1 2 3 4; do for s in 0 10 20; do
  $PY scripts/m0/eval_libero.py --suite libero_90 --task-ids ${TASKS[$i]} --episodes 10 --ep-start $s --port ${PORTS[$i]} \
    --out $OUT/t${TASKS[$i]}_e$s.jsonl --video-dir /workspace/runs/m0/videos/exp1_eval --videos 2 > $OUT/logs/t${TASKS[$i]}_e$s.log 2>&1 &
done; done
wait
