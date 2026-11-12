#!/bin/bash
# Build the faithful proxies' training inputs for one libero_90 task from its 50 demos, with the same scripts and the
# same pipeline as t59 (stage 2 export, render at eval resolution, pi05_libero action targets, faithful inputs).
#   1. scripts/stage2/export_demo_states.py      hdf5 -> /workspace/data/t<T>_demos/states.npz        (openpi venv, CPU)
#   2. demo_idx.npy / frame_idx.npy from states.npz (build_task_data.py reads them from its --distill dir)
#   3. scripts/multitask/render_demo_obs_mt.py   -> policy_obs.npz (+ .json fidelity checks)     (LIBERO venv, EGL)
#   4. scripts/stage2/build_task_data.py         -> /workspace/data/t<T>_task/actions.npy ...     (openpi venv, CPU)
#   5. scripts/faithful/build_faithful_data.py   -> /workspace/data/t<T>_faithful/                 (openpi venv, CPU)
# Writes /workspace/data/t<T>_faithful/BUILD_OK on success. Skips a task that already has BUILD_OK.
# Usage: scripts/multitask/build_task_inputs.sh 50
set -eu
T=$1
source /workspace/env.sh
cd "$(dirname "$0")/../.."
PY=/workspace/openpi/.venv/bin/python
LPY=/workspace/openpi/examples/libero/.venv/bin/python
D=/workspace/data/t${T}_demos
K=/workspace/data/t${T}_task
F=/workspace/data/t${T}_faithful
exec 9>/workspace/data/.build_t${T}.lock
flock 9  # one build per task at a time (the background builder and the stage B chain)
[ -f $F/BUILD_OK ] && { echo "t$T already built"; exit 0; }
NAME=$(PYTHONPATH=/workspace/openpi/third_party/libero $LPY -c "
from libero.libero import benchmark
print(benchmark.get_benchmark_dict()['libero_90']().get_task($T).name)" 2>/dev/null | tail -1)
H=/workspace/data/libero_datasets/libero_90/${NAME}_demo.hdf5
[ -f "$H" ] || { echo "missing $H"; exit 2; }
mkdir -p $D
echo "$(date -u +%FT%TZ) t$T $NAME: export"
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY scripts/stage2/export_demo_states.py --hdf5 "$H" --out $D/states.npz
CUDA_VISIBLE_DEVICES= $PY -c "
import numpy as np
z = np.load('$D/states.npz')
np.save('$D/demo_idx.npy', z['demo_idx']); np.save('$D/frame_idx.npy', z['frame_idx'])
print('demos', len(set(z['demo_idx'].tolist())), 'frames', len(z['demo_idx']))"
echo "$(date -u +%FT%TZ) t$T: render"
PYTHONPATH=.:/workspace/openpi/third_party/libero $LPY scripts/multitask/render_demo_obs_mt.py --task-id $T \
    --states $D/states.npz --out $D/policy_obs.npz 2>&1 | grep -v -i warn | tail -12
echo "$(date -u +%FT%TZ) t$T: task targets"
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY scripts/stage2/build_task_data.py --states $D/states.npz --obs $D/policy_obs.npz \
    --distill $D --out $K | tail -5
echo "$(date -u +%FT%TZ) t$T: faithful inputs"
CUDA_VISIBLE_DEVICES= PYTHONPATH=. PPS_FAITHFUL_OBS=$D/policy_obs.npz PPS_FAITHFUL_STATES=$D/states.npz \
    PPS_FAITHFUL_TASK=$K PPS_FAITHFUL_DATA=$F $PY scripts/faithful/build_faithful_data.py | tail -4
touch $F/BUILD_OK
echo "$(date -u +%FT%TZ) t$T: BUILD_OK"
