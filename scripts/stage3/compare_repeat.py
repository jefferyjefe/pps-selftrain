"""Drop branch determinism check: states 0 to 2, branch 0, pi0.5, run twice (tags a and b) on different servers.

Exit 0 if every pair is identical (actions, qpos, qvel, bottle pose at every step), 3 if any differs, 2 if a run is
missing (the run crashed).
"""

import os
import sys

import numpy as np

ok = True
for k in ("state_00_branch_0", "state_01_branch_0", "state_02_branch_0"):
    fa, fb = ["/workspace/runs/eval_t59/base/drop/%s%s.npz" % (k, t) for t in "ab"]
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(k, "missing")
        sys.exit(2)
    a, b = np.load(fa), np.load(fb)
    same = all(np.array_equal(a[x], b[x]) for x in ("step_action", "step_qpos", "step_qvel", "step_obj"))
    print(k, "identical" if same else "DIFFERENT")
    ok &= same
sys.exit(0 if ok else 3)
