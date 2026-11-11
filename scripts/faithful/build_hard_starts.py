"""Harder starts for t59: the 50 official init states with the bottle moved outside the range of the 50 training demos.
LIBERO venv (Python 3.8).

Declared shift: tomato_sauce_1 (the object to pick) is moved by dy = -0.07 m (world y, away from the tray) from its
position in official init state i, for i = 0..49. Nothing else changes (other objects, tray, arm, time, velocities).
In the 50 demos (frame 0) the bottle's x is in [-0.124, -0.077] and y in [0.025, 0.074]; shifted y is in
[-0.045, 0.004], i.e. 2.1 to 7.0 cm below the demo minimum (the declared amount: >= 2 cm outside the demo range, 7 cm
from the original spot). The bottle stays within reach and in view; the nearest object in that direction is the
alphabet soup (y <= -0.125, about 8 cm center to center).

Each start is checked after openpi's 10 settle steps (dummy action), and a start is kept only if: the bottle rests on
the table (bottle-table contact, no contact with any other object), its orientation is within 5 deg of the unshifted start's after settling, it moved less than
5 mm in xy while settling, and every other object ends within 2 mm (xy) of where it ends from the unshifted state.
Trial j of the harder set uses kept start j, env.seed(7 + 10000 + j) and noise default_rng((7, 10000 + j)).

Output /workspace/runs/eval_t59_faithful/hard/: init_states.npy (n, 97), manifest.json.
Usage
  PYTHONPATH=/workspace/pps-selftrain-wt:/workspace/openpi/third_party/libero:/workspace/pps-selftrain-wt/scripts/stage3 \
      /workspace/openpi/examples/libero/.venv/bin/python scripts/faithful/build_hard_starts.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "stage3"))
import t59_sim as S  # noqa: E402
from pps_real import libero_obs  # noqa: E402
from pps_real import perturb as _perturb  # noqa: E402

OUT = pathlib.Path("/workspace/runs/eval_t59_faithful/hard")
OBJECTS = ["alphabet_soup_1", "cream_cheese_1", "tomato_sauce_1", "ketchup_1", "butter_1", "wooden_tray_1"]
BOTTLE_QPOS = 9 + 7 * 2  # qpos address of tomato_sauce_1's free joint (xyz, quat wxyz)
DY = -0.07
DEMO_BOTTLE = {"x": [-0.124, -0.077], "y": [0.025, 0.074]}


def obj_xy(env):
    q = env.env.sim.data.qpos
    return np.stack([q[9 + 7 * i:9 + 7 * i + 2].copy() for i in range(len(OBJECTS))])


def settle(env, state, seed):
    env.seed(seed)
    env.reset()
    env.set_init_state(state)
    for _ in range(S.SETTLE):
        env.step(libero_obs.DUMMY_ACTION)


def main():
    env, task, init_states = S.make_env()
    OUT.mkdir(parents=True, exist_ok=True)
    kept, rows = [], []
    for i in range(len(init_states)):
        base = np.array(init_states[i], dtype=np.float64)
        settle(env, base, S.SEED + i)
        ref_xy = obj_xy(env)
        ref_quat = env.env.sim.data.qpos[BOTTLE_QPOS + 3:BOTTLE_QPOS + 7].copy()
        mod = base.copy()
        mod[1 + BOTTLE_QPOS + 1] += DY
        settle(env, mod, S.SEED + i)
        xy = obj_xy(env)
        sim = env.env.sim
        q = sim.data.qpos
        quat = q[BOTTLE_QPOS + 3:BOTTLE_QPOS + 7]
        # orientation change vs the same start settled without the shift (the asset's own frame is not z-up)
        dot = abs(float(np.dot(quat / np.linalg.norm(quat), ref_quat / np.linalg.norm(ref_quat))))
        tilt = float(np.degrees(2 * np.arccos(min(1.0, dot))))
        pert = _perturb.Perturber(env, perturbation="none")
        cs = _perturb.object_contacts(sim, pert.obj_geoms)
        on_table = any(c.startswith(_perturb.TABLE_PREFIX) or c.startswith("body:" + _perturb.TABLE_PREFIX) for c in cs)
        others = [c for c in cs if not (c.startswith(_perturb.TABLE_PREFIX) or c.startswith("body:" + _perturb.TABLE_PREFIX))]
        moved = float(np.linalg.norm(xy[2] - (mod[1 + BOTTLE_QPOS:1 + BOTTLE_QPOS + 2])))
        other_shift = float(max(np.linalg.norm(xy[j] - ref_xy[j]) for j in range(len(OBJECTS)) if j != 2))
        ok = on_table and not others and tilt < 5 and moved < 0.005 and other_shift < 0.002
        r = {"base_init_state": i, "bottle_xy_spawn": mod[1 + BOTTLE_QPOS:1 + BOTTLE_QPOS + 2].tolist(),
             "bottle_xy_settled": xy[2].tolist(), "tilt_deg": tilt, "settle_xy_move_m": moved,
             "other_objects_max_xy_shift_m": other_shift, "on_table": on_table, "other_contacts": others,
             "below_demo_min_y_m": DEMO_BOTTLE["y"][0] - float(xy[2][1]), "kept": bool(ok)}
        rows.append(r)
        if ok:
            kept.append(mod)
        print(json.dumps(r), flush=True)
    trials = [{"trial": j, "base_init_state": int(r["base_init_state"]), "env_seed": S.SEED + 10000 + j,
               "noise_seed": [S.SEED, 10000 + j]} for j, r in enumerate([r for r in rows if r["kept"]])]
    np.save(OUT / "init_states.npy", np.stack(kept))
    man = {"rule": __doc__.split("\n\n")[1] + "\n\n" + __doc__.split("\n\n")[2], "dy_m": DY,
           "demo_bottle_range": DEMO_BOTTLE, "n_kept": len(kept), "n_built": len(rows), "trials": trials,
           "rows": rows}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print("kept %d of %d" % (len(kept), len(rows)))
    env.close()


if __name__ == "__main__":
    main()
