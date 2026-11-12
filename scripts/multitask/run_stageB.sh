#!/bin/bash
# Stage B and Step 4 (PROTOCOL_multitask.md), unattended. Never stops the pod.
#   for each task of stageA/shortlist.json in order, until 3 qualify:
#     build inputs -> faithful reference s0 -> faithful task s0 -> check_faithful (A-D) ->
#     2 servers with the task's proxies: pi0.5 repro of screen trials 0 and 1 vs Stage A, then task_f 0-49 and
#     pi0.5 30-49 (0-49 if the repro differs) -> summarize_stageB.py (qualifies: task_f 20-42 of 50)
#   then Step 4: faithful reference and task seeds 1 and 2 on t59 (not evaluated)
# Training never overlaps a server. Writes <B>/DONE_B, <B>/DONE_STEP4.
set -u
WT=/workspace/pps-stage4
A=/workspace/runs/stage4_screen/stageA
B=/workspace/runs/stage4_screen/stageB
LOG=$B/logs
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
NEED=3
source /workspace/env.sh
cd $WT
mkdir -p $LOG
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG/chain.log; }
wait_gpu_free() { while pgrep -f "[s]erve_faithful.py" > /dev/null || pgrep -f "[d]istill_faithful.py|[t]rain_task_faithful.py" > /dev/null; do sleep 20; done; }

train_pair() {  # task seed data-env...
  local T=$1 S=$2; shift 2
  local REF=/workspace/runs/reference_t${T}_faithful_s${S} TASK=/workspace/runs/task_t${T}_faithful_s${S}
  if [ ! -f $REF/final.pt ]; then
    [ -d $REF ] && mv $REF ${REF}_incomplete_$(date +%s)
    log "t$T s$S reference: start"
    env "$@" PYTHONPATH=. $PY scripts/faithful/distill_faithful.py --out $REF --seed $S > $LOG/train_ref_t${T}_s${S}.log 2>&1
    [ -f $REF/final.pt ] || { log "t$T s$S reference FAILED (see train_ref_t${T}_s${S}.log)"; return 1; }
    log "t$T s$S reference: done, $(grep '^done in' $LOG/train_ref_t${T}_s${S}.log)"
  fi
  if [ ! -f $TASK/final.pt ]; then
    [ -d $TASK ] && mv $TASK ${TASK}_incomplete_$(date +%s)
    log "t$T s$S task: start"
    env "$@" PYTHONPATH=. $PY scripts/faithful/train_task_faithful.py --init $REF/final.pt --out $TASK --seed $S \
        > $LOG/train_task_t${T}_s${S}.log 2>&1
    [ -f $TASK/final.pt ] || { log "t$T s$S task FAILED (see train_task_t${T}_s${S}.log)"; return 1; }
    log "t$T s$S task: done, $(grep '^done in' $LOG/train_task_t${T}_s${S}.log)"
  fi
}

serve_task() {  # task: 2 servers with its s0 proxies
  local T=$1
  for port in 8020 8021; do
    PYTHONPATH=. nohup $PY scripts/faithful/serve_faithful.py --port $port --no-v3 --task-id $T \
        --ref /workspace/runs/reference_t${T}_faithful_s0/final.pt --task /workspace/runs/task_t${T}_faithful_s0/final.pt \
        > $LOG/server_t${T}_$port.log 2>&1 &
    sleep 45
  done
  for port in 8020 8021; do until grep -q '"gpu"' $LOG/server_t${T}_$port.log; do sleep 5;
    pgrep -f "[s]erve_faithful.py --port $port" > /dev/null || { log "server $port died"; return 1; }; done; done
  sleep 10
}

stop_servers() { for port in 8020 8021; do pkill -f "[s]erve_faithful.py --port $port"; done; sleep 10; }

workers() {  # root task n_per_server
  local R=$1 T=$2 n=$3
  for port in 8020 8021; do for k in $(seq 1 $n); do
    PYTHONPATH=.:/workspace/openpi/third_party/libero nohup $LPY scripts/multitask/run_mt.py --port $port --out $R \
        --tasks $T > $LOG/worker_t${T}_$(basename $R)_${port}_$k.log 2>&1 &
    sleep 2
  done; done
  sleep 5
  while pgrep -f "[r]un_mt.py --port 802[01] --out $R " > /dev/null; do sleep 20; done
}

until [ -f $A/DONE ]; do sleep 30; done
SHORT=$(python3 -c "import json; print(' '.join(str(t) for t in json.load(open('$A/shortlist.json'))['shortlist']))")
log "shortlist: $SHORT"
if [ -n "${ONLY_TASKS:-}" ]; then SHORT="$ONLY_TASKS"; log "ONLY_TASKS override (user change of plan, Nov 12): $SHORT"; fi
QUAL=0
for T in $SHORT; do
  QUAL=$(python3 -c "
import json, glob
print(sum(json.load(open(f))['qualifies'] for f in glob.glob('$B/t*/result.json')))")
  if [ $QUAL -ge $NEED ]; then log "$QUAL tasks qualify: no further candidates trained"; break; fi
  [ -f $B/t$T/result.json ] && { log "t$T already summarized"; continue; }
  mkdir -p $B/t$T
  log "t$T: inputs"
  bash scripts/multitask/build_task_inputs.sh $T > $LOG/build_t$T.log 2>&1 || { log "t$T input build FAILED"; continue; }
  wait_gpu_free
  DENV="PPS_FAITHFUL_DATA=/workspace/data/t${T}_faithful PPS_FAITHFUL_OBS=/workspace/data/t${T}_demos/policy_obs.npz PPS_FAITHFUL_TASK=/workspace/data/t${T}_task"
  train_pair $T 0 $DENV || { log "t$T training FAILED: not evaluated"; continue; }
  PROMPT=$(python3 -c "import json; print(json.load(open('/workspace/data/t${T}_faithful/real_dims.json'))['prompt'])")
  log "t$T: check_faithful ($PROMPT)"
  PYTHONPATH=. $PY scripts/faithful/check_faithful.py --images /workspace/data/t${T}_demos/policy_obs.npz \
      --prompt "$PROMPT" --ref /workspace/runs/reference_t${T}_faithful_s0/final.pt \
      --task /workspace/runs/task_t${T}_faithful_s0/final.pt --out $B/t$T/check_faithful.json > $LOG/check_t$T.log 2>&1
  if [ $? -ne 0 ]; then log "t$T CHECK FAILED: not evaluated"; touch $B/t$T/CHECK_FAIL; continue; fi
  log "t$T: checks passed"
  serve_task $T || { stop_servers; log "t$T servers FAILED"; continue; }
  log "t$T: servers up"
  RR=$B/repro; mkdir -p $RR
  printf '{"kind": "screen", "task": %d, "cell": "base", "trial": 0}\n{"kind": "screen", "task": %d, "cell": "base", "trial": 1}\n' $T $T >> $RR/queue.jsonl
  workers $RR $T 1
  python3 scripts/multitask/compare_repro.py --kind screen --new $RR/t$T/base --old $A/t$T/base \
      --out $B/t$T/repro_compare.json > $LOG/repro_t$T.log 2>&1
  log "t$T: pi0.5 repro vs Stage A: $(tail -1 $LOG/repro_t$T.log)"
  BASE0=30; grep -q IDENTICAL_ALL $LOG/repro_t$T.log || BASE0=0
  for j in $(seq 0 49); do
    echo "{\"kind\": \"screen\", \"task\": $T, \"cell\": \"task_f\", \"trial\": $j}"
    [ $j -ge $BASE0 ] && echo "{\"kind\": \"screen\", \"task\": $T, \"cell\": \"base\", \"trial\": $j}"
  done >> $B/queue.jsonl
  workers $B $T 3
  stop_servers
  python3 scripts/multitask/summarize_stageB.py --task $T > $LOG/summary_t$T.log 2>&1
  log "t$T: $(tail -1 $LOG/summary_t$T.log)"
done
QUAL=$(python3 -c "
import json, glob
print(sum(json.load(open(f))['qualifies'] for f in glob.glob('$B/t*/result.json')))")
log "stage B done: $QUAL qualify"
touch $B/DONE_B
if [ -n "${NO_STEP4:-}" ]; then log "NO_STEP4 set: stopping before step 4"; exit 0; fi

wait_gpu_free
for S in 1 2; do train_pair 59 $S || log "t59 s$S FAILED"; done
log "step 4 done"
touch $B/DONE_STEP4
