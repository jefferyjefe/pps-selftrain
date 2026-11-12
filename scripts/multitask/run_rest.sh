#!/bin/bash
# Nov 12 change of plan (time box: no training job or eval batch starts unless its estimate ends by 16:40 UTC).
#   Step A  t77 through the stage B chain (ONLY_TASKS=77 NO_STEP4=1)
#   Step B  harder-starts pilot (PROTOCOL_multitask.md addendum 1): t77 then t14, pi0.5 and task_f only, 2 servers
#   Step C  t77 task_f <= 42/50: t82 (train + eval) if it fits, then the t59 s1 reference if it fits
#           t77 task_f  > 42/50: t59 s1 reference, then t59 s1 task, then the t59 s2 reference, each if it fits
# Estimates come from the measured wall times of this pod's runs (chain.log), plus a margin.
set -u
WT=/workspace/pps-stage4
S=/workspace/runs/stage4_screen
B=$S/stageB
H=$S/hard
LOG=$H/logs
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
source /workspace/env.sh
cd $WT
mkdir -p $LOG
DEADLINE=$(date -u -d "2026-11-12T16:40:00Z" +%s)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $S/rest.log; }
fits() {  # minutes label
  local end=$(( $(date +%s) + $1 * 60 ))
  if [ $end -le $DEADLINE ]; then log "$2: estimate $1 min, ends $(date -u -d @$end +%H:%M) UTC <= 16:40: start"; return 0; fi
  log "$2: estimate $1 min, would end $(date -u -d @$end +%H:%M) UTC > 16:40: NOT started"; return 1
}
wall() {  # task what(reference|task|eval) -> minutes (last start to done), from chain.log
  python3 - "$1" "$2" <<'PY'
import sys, datetime as dt
t, what = sys.argv[1], sys.argv[2]
ts = lambda l: dt.datetime.strptime(l.split()[0], "%Y-%m-%dT%H:%M:%SZ")
lines = open("/workspace/runs/stage4_screen/stageB/logs/chain.log").read().splitlines()
if what == "eval":
    a = [ts(l) for l in lines if " t%s s0 task: done" % t in l]
    b = [ts(l) for l in lines if " t%s: t%s base " % (t, t) in l]
else:
    a = [ts(l) for l in lines if " t%s s0 %s: start" % (t, what) in l]
    b = [ts(l) for l in lines if " t%s s0 %s: done" % (t, what) in l]
print(int(round((b[-1] - a[-1]).total_seconds() / 60)) if a and b else 0)
PY
}
wait_gpu_free() { while pgrep -f "[s]erve_faithful.py|[d]istill_faithful.py|[t]rain_task_faithful.py" > /dev/null; do sleep 20; done; }

# ---------------------------------------------------------------- Step A
if [ ! -f $B/t77/result.json ]; then
  fits 250 "step A t77 (reference ~100 + task ~115 + checks/eval ~25 min)" || exit 1
  ONLY_TASKS=77 NO_STEP4=1 bash scripts/multitask/run_stageB.sh >> $S/stageB_chain.nohup 2>&1
fi
log "step A: $(tail -1 $B/logs/summary_t77.log 2>/dev/null)"

# ---------------------------------------------------------------- Step B
serve_task() {
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
hard_task() {
  local T=$1
  [ -f $H/t$T/manifest.json ] || { log "hard t$T: no manifest"; return 1; }
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$H/t$T/manifest.json'))['skipped'] else 1)"; then
    log "hard t$T: SKIPPED by the construction rule"; python3 scripts/multitask/summarize_hard.py --task $T >> $S/rest.log; return 0; fi
  [ -f /workspace/runs/task_t${T}_faithful_s0/final.pt ] || { log "hard t$T: no task_f checkpoint"; return 1; }
  local n=$(python3 -c "import json; print(json.load(open('$H/t$T/manifest.json'))['n_kept'])")
  fits 35 "hard t$T ($n starts x base, task_f)" || return 1
  wait_gpu_free
  serve_task $T || { stop_servers; log "hard t$T: servers FAILED"; return 1; }
  for j in $(seq 0 $((n - 1))); do for c in task_f base; do
    echo "{\"kind\": \"hard\", \"task\": $T, \"cell\": \"$c\", \"trial\": $j}"; done; done >> $H/queue.jsonl
  for port in 8020 8021; do for k in 1 2 3; do
    PYTHONPATH=.:/workspace/openpi/third_party/libero nohup $LPY scripts/multitask/run_mt.py --port $port --out $H \
        --tasks $T > $LOG/worker_t${T}_${port}_$k.log 2>&1 &
    sleep 2
  done; done
  sleep 5
  while pgrep -f "[r]un_mt.py --port 802[01] --out $H " > /dev/null; do sleep 20; done
  stop_servers
  log "hard t$T: $(python3 scripts/multitask/summarize_hard.py --task $T 2>&1 | tail -1)"
}
hard_task 77
hard_task 14

# ---------------------------------------------------------------- Step C
train_ref59() {  # seed
  local R=/workspace/runs/reference_t59_faithful_s$1
  [ -d $R ] && [ ! -f $R/final.pt ] && mv $R ${R}_incomplete_$(date +%s)
  log "t59 s$1 reference: start"
  PYTHONPATH=. $PY scripts/faithful/distill_faithful.py --out $R --seed $1 > $B/logs/train_ref_t59_s$1.log 2>&1
  [ -f $R/final.pt ] && log "t59 s$1 reference: done, $(grep '^done in' $B/logs/train_ref_t59_s$1.log)" || { log "t59 s$1 reference FAILED"; return 1; }
}
train_task59() {  # seed
  local R=/workspace/runs/reference_t59_faithful_s$1 K=/workspace/runs/task_t59_faithful_s$1
  [ -d $K ] && [ ! -f $K/final.pt ] && mv $K ${K}_incomplete_$(date +%s)
  log "t59 s$1 task: start"
  PYTHONPATH=. $PY scripts/faithful/train_task_faithful.py --init $R/final.pt --out $K --seed $1 \
      > $B/logs/train_task_t59_s$1.log 2>&1
  [ -f $K/final.pt ] && log "t59 s$1 task: done, $(grep '^done in' $B/logs/train_task_t59_s$1.log)" || { log "t59 s$1 task FAILED"; return 1; }
}
REF14=$(wall 14 reference); TASK14=$(wall 14 task)
REF77=$(wall 77 reference); TASK77=$(wall 77 task); EVAL77=$(wall 77 eval)
log "measured walls (min): t14 ref $REF14 task $TASK14; t77 ref $REF77 task $TASK77 checks+eval $EVAL77"
EST_REF59=$((REF14 + 8)); EST_TASK59=$((TASK14 + 5))
KF=$(python3 -c "import json; print(json.load(open('$B/t77/result.json'))['task_f']['successes'])" 2>/dev/null || echo NA)
wait_gpu_free
if [ "$KF" != NA ] && [ $KF -le 42 ]; then
  log "step C: t77 task_f $KF/50 <= 85%: t82 if it fits, then the t59 s1 reference"
  if fits $((REF77 + TASK77 + EVAL77 + 15)) "t82 train + eval"; then
    ONLY_TASKS=82 NO_STEP4=1 bash scripts/multitask/run_stageB.sh >> $S/stageB_chain.nohup 2>&1
    log "t82: $(tail -1 $B/logs/summary_t82.log 2>/dev/null)"
  fi
  wait_gpu_free
  fits $EST_REF59 "t59 s1 reference" && train_ref59 1
else
  log "step C: t77 task_f $KF/50 > 85% (or missing): stage B paused; t59 s1 reference, s1 task, s2 reference"
  if fits $EST_REF59 "t59 s1 reference" && train_ref59 1; then
    fits $EST_TASK59 "t59 s1 task" && train_task59 1 && fits $EST_REF59 "t59 s2 reference" && train_ref59 2
  fi
fi
touch $S/DONE_REST
log "run_rest done"
