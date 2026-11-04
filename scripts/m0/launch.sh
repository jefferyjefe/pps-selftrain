#!/bin/bash
# Shard LIBERO tasks across clients. Usage
#   launch.sh SUITE "TASK_IDS" EPISODES EP_START OUT_DIR "PORTS" CLIENTS_PER_PORT [extra eval_libero args]
# Each client writes OUT_DIR/shard_K.jsonl. Waits for all clients.
set -u
SUITE=$1; TASKS=($2); EPS=$3; EP0=$4; OUT=$5; PORTS=($6); CPP=$7; shift 7
source /workspace/env.sh
export PYTHONPATH=/workspace/openpi/third_party/libero
PY=/workspace/openpi/examples/libero/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
N=$(( ${#PORTS[@]} * CPP ))
for ((k=0; k<N; k++)); do
  mine=(); for ((i=k; i<${#TASKS[@]}; i+=N)); do mine+=(${TASKS[$i]}); done
  [ ${#mine[@]} -eq 0 ] && continue
  port=${PORTS[$(( k % ${#PORTS[@]} ))]}
  $PY "$HERE/eval_libero.py" --suite $SUITE --task-ids ${mine[@]} --episodes $EPS --ep-start $EP0 --port $port \
      --out "$OUT/shard_$k.jsonl" "$@" > "$OUT/logs/shard_$k.log" 2>&1 &
done
wait
