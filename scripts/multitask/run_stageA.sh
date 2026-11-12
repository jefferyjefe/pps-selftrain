#!/bin/bash
# Stage A (PROTOCOL_multitask.md section a): pi0.5 alone, 30 screening trials on each candidate, 2 servers x 3 workers.
# Then applies the shortlist rule (select_shortlist.py) and stops the servers. Writes <A>/DONE. Never stops the pod.
set -u
WT=/workspace/pps-stage4
A=/workspace/runs/stage4_screen/stageA
LOG=$A/logs
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
TASKS="2 14 19 31 38 43 50 56 60 64 70 77 82"
source /workspace/env.sh
cd $WT
mkdir -p $LOG
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/stageA.log; }
if [ ! -f $A/queue.jsonl ]; then
  for t in $TASKS; do for j in $(seq 0 29); do
    echo "{\"kind\": \"screen\", \"task\": $t, \"cell\": \"base\", \"trial\": $j}"; done; done > $A/queue.jsonl
fi
log "queue: $(wc -l < $A/queue.jsonl) jobs"
for port in 8020 8021; do
  PYTHONPATH=. nohup $PY scripts/faithful/serve_faithful.py --port $port --no-faithful --no-v3 > $LOG/server_$port.log 2>&1 &
  sleep 45
done
for port in 8020 8021; do until grep -q '"gpu"' $LOG/server_$port.log; do sleep 5; done; done
sleep 10
log "servers up: $(grep -h '"gpu"' $LOG/server_80*.log | head -1)"
for port in 8020 8021; do for k in 1 2 3; do
  PYTHONPATH=.:/workspace/openpi/third_party/libero nohup $LPY scripts/multitask/run_mt.py --port $port --out $A \
      > $LOG/worker_${port}_$k.log 2>&1 &
  sleep 2
done; done
log "workers up: 6"
while pgrep -f "[r]un_mt.py --port 802[01] --out $A" > /dev/null; do sleep 30; done
log "workers finished: $(cat $A/t*/base/screen.jsonl | wc -l) records"
for port in 8020 8021; do pkill -f "[s]erve_faithful.py --port $port"; done
sleep 5
python3 scripts/multitask/select_shortlist.py --stageA $A > $LOG/shortlist.log 2>&1
cat $LOG/shortlist.log | tee -a $LOG/stageA.log
touch $A/DONE
log "stage A done"
