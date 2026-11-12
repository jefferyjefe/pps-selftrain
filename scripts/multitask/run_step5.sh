#!/bin/bash
# Step 5 (PROTOCOL_multitask.md section c): drop calibration on the chosen tasks, pi0.5 alone. Waits for Step 4.
#   pilot trials (offset 30000) in batches of 50 until 30 pilot states per task (at most 150 trials), then every
#   pilot state x levels 0-5, then calibrate.py summarize (frozen level). 2 servers x 3 workers. Writes DONE_STEP5.
# Usage: scripts/multitask/run_step5.sh [task ...]   (default: the qualifying tasks of stage B, shortlist order)
set -u
WT=/workspace/pps-stage4
B=/workspace/runs/stage4_screen/stageB
C=/workspace/runs/stage4_screen/step5
LOG=$C/logs
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
source /workspace/env.sh
cd $WT
mkdir -p $LOG
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/step5.log; }
until [ -f $B/DONE_STEP4 ]; do sleep 60; done
while pgrep -f "[s]erve_faithful.py|[d]istill_faithful.py|[t]rain_task_faithful.py" > /dev/null; do sleep 20; done
if [ $# -gt 0 ]; then TASKS="$*"; else
TASKS=$(python3 -c "
import json
sl = json.load(open('/workspace/runs/stage4_screen/stageA/shortlist.json'))['shortlist']
q = []
for t in sl:
    try:
        r = json.load(open('$B/t%d/result.json' % t))
    except FileNotFoundError:
        continue
    if r['qualifies']:
        q.append(t)
print(' '.join(str(t) for t in q[:3]))"); fi
log "chosen tasks: $TASKS"
[ -z "$TASKS" ] && { log "no chosen task"; touch $C/DONE_STEP5; exit 0; }
for port in 8020 8021; do
  PYTHONPATH=. nohup $PY scripts/faithful/serve_faithful.py --port $port --no-faithful --no-v3 > $LOG/server_$port.log 2>&1 &
  sleep 45
done
for port in 8020 8021; do until grep -q '"gpu"' $LOG/server_$port.log; do sleep 5; done; done
sleep 10
log "servers up"
run_workers() {
  for port in 8020 8021; do for k in 1 2 3; do
    PYTHONPATH=.:/workspace/openpi/third_party/libero nohup $LPY scripts/multitask/run_mt.py --port $port --out $C \
        > $LOG/worker_${port}_$k.log 2>&1 &
    sleep 2
  done; done
  sleep 5
  while pgrep -f "[r]un_mt.py --port 802[01] --out $C" > /dev/null; do sleep 20; done
}
for b in 0 1 2; do
  NEEDP=""
  for T in $TASKS; do
    n=0; [ -f $C/t$T/pilot_states.json ] && n=$(python3 -c "import json; print(json.load(open('$C/t$T/pilot_states.json'))['n_states'])")
    [ $n -ge 30 ] && continue
    NEEDP="$NEEDP $T"
    for j in $(seq $((50 * b)) $((50 * b + 49))); do
      echo "{\"kind\": \"pilot\", \"task\": $T, \"cell\": \"base\", \"trial\": $j}"; done >> $C/queue.jsonl
  done
  [ -z "$NEEDP" ] && break
  log "pilot batch $b for:$NEEDP"
  run_workers
  for T in $NEEDP; do log "t$T pilot states (found, trials done): $(python3 scripts/multitask/calibrate.py select --task $T)"; done
done
for T in $TASKS; do
  n=$(python3 -c "import json; print(json.load(open('$C/t$T/pilot_states.json'))['n_states'])")
  for s in $(seq 0 $((n - 1))); do for L in 0 1 2 3 4 5; do
    echo "{\"kind\": \"cal\", \"task\": $T, \"cell\": \"base\", \"state\": $s, \"level\": $L}"; done; done >> $C/queue.jsonl
done
log "calibration branches queued"
run_workers
for port in 8020 8021; do pkill -f "[s]erve_faithful.py --port $port"; done
for T in $TASKS; do python3 scripts/multitask/calibrate.py summarize --task $T > $LOG/cal_t$T.log 2>&1; cat $LOG/cal_t$T.log >> $LOG/step5.log; done
touch $C/DONE_STEP5
log "step 5 done"
