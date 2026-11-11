#!/bin/bash
# Stage 3 full run, unattended. Launch with
#   nohup setsid bash scripts/stage3/run_all.sh > /workspace/runs/eval_t59/logs/run_all.nohup 2>&1 < /dev/null &
# 1. Keeps 3 dual method policy servers (8010-8012) up and WPS workers per server running while unclaimed jobs remain.
#    Workers take paired trials first (pi0.5 and PPS alternating), then drop branches once DROP_OK exists.
# 2. When the pi0.5 trials give a finished prefix with 40 mid-carry events, selects the pool and runs the branching
#    checks (exact replay of all 40 states; 3 drop branches run twice on different servers). Pass: DROP_OK. Fail: writes
#    SUMMARY.md with the failure and stops the pod.
# 3. When every job is done: stops servers and workers, writes SUMMARY.md, touches DONE, waits GRACE seconds (time
#    for a review of the summary) and stops the pod.
# Too many crashes (server restarts or worker relaunches over the caps) also end in a summary and a pod stop.
set -u
source /workspace/env.sh
cd /workspace/pps-selftrain
export PYTHONPATH=.:/workspace/openpi/third_party/libero
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CPY=/workspace/openpi/examples/libero/.venv/bin/python
SPY=/workspace/openpi/.venv/bin/python
D=scripts/stage3
R=/workspace/runs/eval_t59; L=$R/logs
PORTS="8010 8011 8012"; WPS=${WPS:-3}; GRACE=${GRACE:-900}
MAX_SERVER_RESTARTS=6; MAX_RELAUNCH=40
mkdir -p $L $R/checks
log(){ echo "$(date -u '+%F %T') $*" >> $L/orchestrator.log; }

finish(){  # $1 partial note ("" if complete), $2 stop reason
  log "finish: ${1:-complete}"
  pkill -f "$D/run_t59.py"; sleep 2; pkill -f "$D/serve_eval.py"; sleep 5
  if [ -n "$1" ]; then timeout 600 $SPY $D/summarize.py --partial-note "$1" >> $L/summarize.log 2>&1
  else timeout 600 $SPY $D/summarize.py >> $L/summarize.log 2>&1; fi
  log "summarize exit $?; GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  touch $R/DONE
  log "DONE written, waiting ${GRACE}s before stopping the pod"
  sleep $GRACE
  bash $D/pod_stop.sh "$2"
  exit 0
}

start_server(){
  ( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    nohup $SPY $D/serve_eval.py --port $1 >> $L/server_$1.log 2>&1 < /dev/null & )
}
healthy(){ curl -s -m 5 http://127.0.0.1:$1/healthz > /dev/null; }

server_restarts=0
ensure_servers(){
  for p in $PORTS; do
    if ! healthy $p && ! pgrep -f "serve_eval.py --port $p" > /dev/null; then
      server_restarts=$((server_restarts + 1)); log "server $p down, restart #$server_restarts"
      [ $server_restarts -gt $MAX_SERVER_RESTARTS ] && finish "Run aborted: policy servers kept crashing (over $MAX_SERVER_RESTARTS restarts), see logs/server_*.log." "server crashes"
      start_server $p
    fi
  done
  for p in $PORTS; do for i in $(seq 1 120); do healthy $p && break; sleep 5; done; done
}

relaunches=0
launch_workers(){
  for p in $PORTS; do
    n=$(pgrep -f "run_t59.py worker --port $p" | wc -l)
    while [ $n -lt $WPS ]; do
      relaunches=$((relaunches + 1)); n=$((n + 1))
      nohup $CPY $D/run_t59.py worker --port $p > $L/worker_${p}_$relaunches.log 2>&1 < /dev/null &
    done
  done
}

# exit 0 pass, 3 mismatch found, 1 crashed three times
replay_check(){
  NS=$(python3 -c "import json; print(len(json.load(open('$R/pool/manifest.json'))['states']))")
  for a in 1 2 3; do
    rm -f $R/checks/replay_check.jsonl
    $CPY $D/run_t59.py replay --port 8011 --states $(seq 0 $((NS - 1))) > $L/replay_check_$a.log 2>&1; rc=$?
    if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then return $rc; fi
    log "replay check process crashed (exit $rc), attempt $a"; sleep 60
  done
  return 1
}
repeat_check(){
  for a in 1 2 3; do
    $CPY $D/run_t59.py repeat --port 8010 --method base --pairs 0:0 1:0 2:0 --tag a >> $L/repeat_a.log 2>&1
    $CPY $D/run_t59.py repeat --port 8012 --method base --pairs 0:0 1:0 2:0 --tag b >> $L/repeat_b.log 2>&1
    $SPY $D/compare_repeat.py > $L/repeat_cmp.log 2>&1; rc=$?
    if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then return $rc; fi
    log "repeat check incomplete (exit $rc), attempt $a"; sleep 60
  done
  return 1
}

checks_started=0
POOL_TRIALS=200
run_checks(){
  log "pool selected: $(cat $L/select_pool.out)"
  replay_check; rc=$?
  if [ $rc -eq 1 ]; then
    touch $R/DROP_FAIL; log "replay check crashed 3 times"
    printf '%s\n' "## STOPPED: the branching check could not run" "" \
      "The replay check process crashed three times (logs/replay_check_*.log), so branching could not be verified and" \
      "the drop test was not run. The pod was stopped. The normal paired trials finished before the stop are summarized" \
      "above." >> $R/SUMMARY_NOTES.md
    finish "Stopped because the branching check crashed repeatedly (see the notes at the end)." "replay check crashed"
  elif [ $rc -ne 0 ]; then
    touch $R/DROP_FAIL; log "replay check FAILED"
    printf '%s\n' "## STOPPED: branching is not reproducible" "" \
      "Restoring a saved mid-carry state and replaying pi0.5 with the saved action queue and noise generator did not" \
      "reproduce the original trajectory bit for bit for at least one state. Per the instructions the drop test was not" \
      "run and the pod was stopped. Per state comparisons are in /workspace/runs/eval_t59/checks/replay_check.jsonl" \
      "(first differing step, max abs difference per quantity). The normal paired trials finished before the stop are" \
      "summarized above." >> $R/SUMMARY_NOTES.md
    finish "Stopped because the branching check failed (see the notes at the end)." "branching check failed"
  fi
  repeat_check; rc=$?
  if [ $rc -ne 0 ]; then
    touch $R/DROP_FAIL; log "drop branch repeat check FAILED or crashed (rc $rc)"
    printf '%s\n' "## STOPPED: drop branches are not reproducible (or the check could not run)" "" \
      "Three drop branches (states 0 to 2, branch 0, pi0.5) were run twice on different server processes (exit code $rc:" \
      "3 = trajectories differ, 1 = the check crashed three times; logs/repeat_cmp.log, logs/repeat_*.log). The replay" \
      "check had passed. Per the instructions the drop test was not run and the pod was stopped. The normal paired" \
      "trials finished before the stop are summarized above." >> $R/SUMMARY_NOTES.md
    finish "Stopped because the drop branch repeat check failed (see the notes at the end)." "drop repeat check failed"
  fi
  touch $R/DROP_OK; log "branching checks passed, DROP_OK"
}

log "run_all start, pid $$"
ensure_servers
while true; do
  if [ -f $R/DROP_FAIL ]; then log "check failed, main loop stands down (run_checks finishes and stops the pod)"; wait; exit 0; fi
  read pu pc du dc < <(python3 $D/jobs_status.py --cleanup)
  if [ ! -f $R/DROP_OK ] && [ $checks_started -eq 0 ]; then
    python3 $D/select_pool.py --n-trials $POOL_TRIALS > $L/select_pool.out 2>&1; rc=$?
    if [ $rc -eq 1 ] && [ $POOL_TRIALS -eq 200 ]; then
      # Fewer than 40 qualifying events in the 200 normal trials: workers run extra pi0.5 trials 200..259 (same
      # conventions) only to fill the pool. summarize.py keeps them out of the normal trial statistics.
      POOL_TRIALS=260; touch $R/POOL_EXTRA
      log "fewer than 40 qualifying events in 200 pi0.5 trials, workers run pool-only trials 200..259"
      pkill -f "$D/run_t59.py worker"; sleep 3   # restart workers so all run the code that knows POOL_EXTRA
      rc=2
    elif [ $rc -eq 1 ]; then python3 $D/select_pool.py --n-trials 260 --allow-fewer > $L/select_pool.out 2>&1; rc=0
      log "fewer than 40 qualifying events even in 260 pi0.5 trials, using all of them"; fi
    if [ $rc -eq 0 ]; then checks_started=1; rm -f $R/POOL_EXTRA; run_checks & fi
  fi
  if [ "$pu" = "0" ] && [ -f $R/DROP_OK ] && [ "$du" = "0" ]; then finish "" "run complete"; fi
  if [ "$pc" != "0" ] || { [ -f $R/DROP_OK ] && [ "$dc" != "0" ]; } || { [ ! -f $R/DROP_OK ] && [ ! -f $R/DROP_FAIL ]; }; then
    ensure_servers
    [ $relaunches -gt $((MAX_RELAUNCH + 3 * WPS)) ] && finish "Run aborted: workers kept crashing ($relaunches launches), see logs/worker_*.log." "worker crashes"
    launch_workers
  fi
  sleep 30
done
