#!/bin/bash
# Background builder: training inputs for the candidates, one task at a time (CPU, plus EGL rendering), in the given
# order. While Stage A runs, run_stageB.sh builds anything still missing itself (per-task lock in build_task_inputs.sh).
# Usage: scripts/multitask/build_queue.sh 14 70 82 ...
cd "$(dirname "$0")/../.."
L=/workspace/runs/stage4_screen/logs/build
mkdir -p $L
for T in "$@"; do
  echo "$(date -u +%FT%TZ) build t$T" >> $L/queue.log
  bash scripts/multitask/build_task_inputs.sh $T > $L/t$T.log 2>&1
  echo "$(date -u +%FT%TZ) t$T exit $?" >> $L/queue.log
done
