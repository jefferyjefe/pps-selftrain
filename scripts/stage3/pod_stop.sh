#!/bin/bash
# Stop this RunPod pod. runpodctl first, GraphQL podStop with the pod's own API key as fallback.
#   pod_stop.sh REASON
# runpodctl get pod is Unauthorized with the pod scoped key (it lists the account's pods), but the podStop mutation is
# authorized (checked with an invalid pod id on 2026-11-10, which returned POD_NOT_FOUND, not Unauthorized).
LOG=/workspace/runs/eval_t59/logs/pod_stop.log
echo "$(date -u) stop requested: $*" >> $LOG
sync
if runpodctl stop pod "$RUNPOD_POD_ID" >> $LOG 2>&1; then
  echo "$(date -u) runpodctl stop returned 0" >> $LOG
else
  echo "$(date -u) runpodctl stop failed, trying GraphQL" >> $LOG
  curl -s -m 30 -H "Content-Type: application/json" "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" >> $LOG 2>&1
  echo >> $LOG
fi
sync
