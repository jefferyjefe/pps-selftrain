#!/bin/bash
# RTX 4090 rerun of the t59 evaluation (2026-11-11): step 5 (pi0.5 alone, PPS gamma 0.4) and step 6 (task proxy alone)
# on the same 200 paired normal trials and the same 40 saved drop states x 5 branches as the Nov 10 RTX PRO 4500 run.
# Unattended. Unlike run_all.sh it NEVER stops the pod and has no watchdog. Launch with
#   nohup setsid bash scripts/stage3/run_4090.sh > /workspace/runs/eval_t59_4090/logs/run_4090.nohup 2>&1 < /dev/null &
# Phase A, before the full run:
#   1. check_gamma0.py (gamma 0 bit exact with pi0.5). Fails: nothing is run, summary written, stop (pod stays up).
#   2. check_task_only.py (task proxy fast path bit exact with the reference hook path). Fails: servers use the hook.
#   3. two dual servers (8010, 8011; a third does not fit in 24 GB).
#   4. branching checks on the copied Nov 10 pool: sim side open loop replay (40 states, must be bit exact), policy
#      replay vs the PRO 4500 trajectories (must be identical up to the first fresh 4090 policy call), and 3 drop
#      branches run twice on different servers (must be identical). Pass: DROP_OK. Fail: DROP_FAIL, normal trials only.
# Phase B: WPS workers per server, groups "base,pps" then "task" (step 5 first), paired trials then drop branches.
# Phase C: summarize_4090.py, DONE. Create $R/STOP to end early (a summary is still written).
set -u
source /workspace/env.sh
cd /workspace/pps-selftrain
export PYTHONPATH=.:/workspace/openpi/third_party/libero
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CPY=/workspace/openpi/examples/libero/.venv/bin/python
SPY=/workspace/openpi/.venv/bin/python
D=scripts/stage3
R=/workspace/runs/eval_t59_4090; L=$R/logs; OLD=/workspace/runs/eval_t59
PORTS="8010 8011"; WPS=${WPS:-3}
MAX_SERVER_RESTARTS=6; MAX_RELAUNCH=40
COMMON="--out $R --pool-dir $R/pool --orig-root $OLD"
mkdir -p $L $R/checks $R/pool_own4090
log(){ echo "$(date -u '+%F %T') $*" >> $L/orchestrator.log; }
note(){ printf '%s\n' "$@" >> $R/SUMMARY_NOTES.md; }

finish(){  # $1 partial note ("" if complete)
  log "finish: ${1:-complete}"
  pkill -f "[r]un_t59.py worker"; sleep 2; pkill -f "[s]erve_eval.py --port 801"; sleep 5
  if [ -n "$1" ]; then timeout 900 $SPY $D/summarize_4090.py --partial-note "$1" >> $L/summarize.log 2>&1
  else timeout 900 $SPY $D/summarize_4090.py >> $L/summarize.log 2>&1; fi
  log "summarize exit $?; GPU processes left: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  touch $R/DONE
  log "DONE written. The pod is left running."
  exit 0
}

TASK_IMPL=fast
start_server(){
  ( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    nohup $SPY $D/serve_eval.py --port $1 --task-impl $TASK_IMPL >> $L/server_$1.log 2>&1 < /dev/null & )
}
healthy(){ curl -s -m 5 http://127.0.0.1:$1/healthz > /dev/null; }
server_restarts=0
ensure_servers(){
  for p in $PORTS; do
    if ! healthy $p && ! pgrep -f "[s]erve_eval.py --port $p" > /dev/null; then
      server_restarts=$((server_restarts + 1)); log "server $p down, (re)start #$server_restarts"
      [ $server_restarts -gt $((MAX_SERVER_RESTARTS + 2)) ] && finish "Run aborted: policy servers kept crashing, see logs/server_*.log."
      start_server $p
    fi
  done
  for p in $PORTS; do for i in $(seq 1 120); do healthy $p && break; sleep 5; done; done
}
relaunches=0
launch_workers(){
  for p in $PORTS; do
    n=$(pgrep -f "[r]un_t59.py worker --port $p" | wc -l)
    while [ $n -lt $WPS ]; do
      relaunches=$((relaunches + 1)); n=$((n + 1))
      nohup $CPY $D/run_t59.py worker --port $p $COMMON --save-pool-dir $R/pool_own4090 --groups base,pps task \
        > $L/worker_${p}_$relaunches.log 2>&1 < /dev/null &
    done
  done
}

# a check split over both servers; exit 0 all pass, 3 some mismatch, 1 crashed three times
split_check(){  # $1 mode, $2 jsonl name
  for a in 1 2 3; do
    rm -f $R/checks/$2
    $CPY $D/run_t59.py $1 --port 8010 $COMMON --states $(seq 0 19) > $L/${1}_check_8010_$a.log 2>&1 & p1=$!
    $CPY $D/run_t59.py $1 --port 8011 $COMMON --states $(seq 20 39) > $L/${1}_check_8011_$a.log 2>&1 & p2=$!
    wait $p1; r1=$?; wait $p2; r2=$?
    if { [ $r1 -eq 0 ] || [ $r1 -eq 3 ]; } && { [ $r2 -eq 0 ] || [ $r2 -eq 3 ]; }; then
      [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && return 0; return 3; fi
    log "$1 check crashed (exit $r1 / $r2), attempt $a"; sleep 30
  done
  return 1
}

log "run_4090 start, pid $$, GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[ -f $R/pool/manifest.json ] || finish "Stopped: the copied Nov 10 pool is missing ($R/pool)."

# ---- A1 gamma 0
if [ ! -f $R/checks/gamma0_check.json ]; then
  ( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    $SPY $D/check_gamma0.py --out $R/checks/gamma0_check.json > $L/check_gamma0.log 2>&1 )
fi
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/checks/gamma0_check.json'))['bit_exact_all'] else 1)" 2>/dev/null; then
  touch $R/GAMMA0_FAIL; log "gamma 0 check FAILED or crashed"
  note "## STOPPED: gamma = 0 is not bit exact with pi0.5 on the 4090 (or the check crashed)" "" \
    "See checks/gamma0_check.json and logs/check_gamma0.log. Per the instructions nothing else was run."
  finish "Stopped before the full run: the gamma = 0 check did not pass."
fi
log "gamma 0 check passed"

# ---- A2 task proxy alone, fast path vs reference path
if [ ! -f $R/checks/task_only_check.json ]; then
  ( export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    $SPY $D/check_task_only.py --out $R/checks/task_only_check.json > $L/check_task_only.log 2>&1 )
fi
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/checks/task_only_check.json'))['bit_exact_all'] else 1)" 2>/dev/null; then
  TASK_IMPL=fast; log "task only fast path bit exact, servers use it"
else
  TASK_IMPL=hook; log "task only fast path NOT bit exact (or check crashed), servers use the hook path"
  note "- The task proxy alone ran through the reference hook path (pps_real.sampler with the base velocity replaced)," \
       "  because the fast path did not pass check_task_only.py."
fi

# ---- A3 servers
ensure_servers
log "servers up"

# ---- A4 branching checks on the copied Nov 10 pool
if [ ! -f $R/DROP_OK ] && [ ! -f $R/DROP_FAIL ]; then
  split_check simreplay simreplay_check.jsonl; rs=$?
  log "sim replay check exit $rs"
  split_check replay replay_check.jsonl; rp=$?
  log "policy replay check exit $rp (3 expected: not identical to the PRO 4500 after the first fresh call)"
  python3 - "$R" <<'EOF'; rg=$?
import json, sys
R = sys.argv[1]
sim = [json.loads(l) for l in open(R + "/checks/simreplay_check.jsonl")]
rep = [json.loads(l) for l in open(R + "/checks/replay_check.jsonl")]
ok = (len(sim) == 40 and all(r["identical"] for r in sim) and len(rep) == 40
      and all(r["identical_before_first_fresh_call"] for r in rep))
print("sim identical %d/%d, policy replay identical before the first fresh call %d/%d" % (
    sum(r["identical"] for r in sim), len(sim), sum(r["identical_before_first_fresh_call"] for r in rep), len(rep)))
sys.exit(0 if ok else 3)
EOF
  log "replay gate exit $rg"
  rr=1
  if [ $rg -eq 0 ]; then
    for a in 1 2 3; do
      $CPY $D/run_t59.py repeat --port 8010 --method base --pairs 0:0 1:0 2:0 --tag a $COMMON >> $L/repeat_a.log 2>&1 & q1=$!
      $CPY $D/run_t59.py repeat --port 8011 --method base --pairs 0:0 1:0 2:0 --tag b $COMMON >> $L/repeat_b.log 2>&1 & q2=$!
      wait $q1 $q2
      $SPY $D/compare_repeat.py $R > $L/repeat_cmp.log 2>&1; rr=$?
      { [ $rr -eq 0 ] || [ $rr -eq 3 ]; } && break
      log "repeat check incomplete (exit $rr), attempt $a"; sleep 30
    done
    log "drop repeat check exit $rr"
  fi
  if [ $rg -eq 0 ] && [ $rr -eq 0 ]; then
    touch $R/DROP_OK; log "branching checks passed, DROP_OK"
  else
    touch $R/DROP_FAIL; log "branching checks FAILED (sim $rs, gate $rg, repeat $rr): drop test not run"
    note "## Drop test NOT run: a branching check failed on the 4090" "" \
      "Sim side replay exit $rs, replay gate exit $rg, drop repeat exit $rr (0 = pass). Per state results are in" \
      "checks/simreplay_check.jsonl and checks/replay_check.jsonl, the repeat comparison in logs/repeat_cmp.log." \
      "The normal paired trials ran as planned."
  fi
fi

# ---- B workers
log "workers start (WPS $WPS, task impl $TASK_IMPL)"
while true; do
  [ -f $R/STOP ] && finish "Stopped early by the STOP file."
  read pu pc du dc < <(python3 $D/jobs_status.py --cleanup --root $R --methods base pps task)
  if [ "$pu" = "0" ] && { [ -f $R/DROP_FAIL ] || [ "$du" = "0" ]; }; then
    if [ -f $R/DROP_FAIL ]; then finish "The drop test was not run because a branching check failed (see the notes)."; fi
    finish ""
  fi
  if [ "$pc" != "0" ] || { [ -f $R/DROP_OK ] && [ "$dc" != "0" ]; }; then
    ensure_servers
    [ $relaunches -gt $((MAX_RELAUNCH + 2 * WPS)) ] && finish "Run aborted: workers kept crashing ($relaunches launches), see logs/worker_*.log."
    launch_workers
  fi
  sleep 30
done
