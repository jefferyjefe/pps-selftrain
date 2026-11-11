#!/bin/bash
# Safety net. Stops the pod at 2026-11-10 14:00 US Eastern (EST, UTC-5) = 19:00 UTC no matter what.
# Five minutes before, it tries (with a time limit) to write a partial summary so results on disk are summarized.
STOP_EPOCH=1794337200   # TZ=America/New_York date -d "2026-11-10 14:00" +%s  ->  Tue Nov 10 19:00:00 UTC 2026
DIR=/workspace/pps-selftrain/scripts/stage3
LOG=/workspace/runs/eval_t59/logs/watchdog.log
echo "$(date -u) watchdog pid $$ armed, stop at $(date -u -d @$STOP_EPOCH)" >> $LOG
while [ $(date +%s) -lt $((STOP_EPOCH - 300)) ]; do sleep 30; done
if [ ! -f /workspace/runs/eval_t59/DONE ]; then
  echo "$(date -u) run not finished, writing partial summary" >> $LOG
  source /workspace/env.sh
  timeout 200 /workspace/openpi/.venv/bin/python $DIR/summarize.py --partial-note "Pod stopped by the 14:00 ET watchdog before the run finished." >> $LOG 2>&1
fi
while [ $(date +%s) -lt $STOP_EPOCH ]; do sleep 5; done
echo "$(date -u) stopping pod" >> $LOG
bash $DIR/pod_stop.sh "watchdog 14:00 ET"
