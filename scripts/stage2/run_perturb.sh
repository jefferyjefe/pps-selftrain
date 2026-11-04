#!/bin/bash
# Deliverable 3 runs. One condition per server, all in parallel.
#   run_perturb.sh EPISODES [VIDEOS]
# Episodes 0..EPISODES-1 (resumable, so 10 then 30 extends the same files). Servers on 8002, 8003, 8004.
set -u
EPS=$1; VIDS=${2:-2}
source /workspace/env.sh
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:/workspace/openpi/third_party/libero
PY=/workspace/openpi/examples/libero/.venv/bin/python
OUT=results/stage2/perturb; LOG=/workspace/runs/stage2/logs; VID=/workspace/runs/stage2/videos
mkdir -p $OUT $LOG $VID
i=0
for cond in none teleport release; do
  port=$((8002 + i)); i=$((i + 1))
  $PY scripts/stage2/eval_perturb.py --perturbation $cond --episodes $EPS --port $port --out $OUT/$cond.jsonl \
      --video-dir $VID --videos $VIDS > $LOG/perturb_$cond.log 2>&1 &
done
wait
