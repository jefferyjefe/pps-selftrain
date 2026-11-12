"""Harder starts for any libero_90 task (PROTOCOL_multitask.md, addendum 1): the t59 construction
(scripts/faithful/build_hard_starts.py) with the shift direction and size given by a fixed rule. LIBERO venv (3.8).

Rule (fixed before any harder-start rollout):
  object   the task's carried object (sim_mt.task_spec: first (On|In obj target) goal predicate).
  demo range  the carried object's xy at frame 0 of each of the 50 training demos (/workspace/data/t<T>_demos/
           states.npz, frame_idx == 0), read after env.set_init_state.
  goal location  the goal target's xy in the same 50 frame-0 states (sim_mt.Probe.target_xy: the target site's centre
           for a site, else the target object's body position); its mean over the 50 demos.
  direction 1  "directly away from the goal", along a world table axis as on t59 (-y there): take the xy vector from
           the mean goal location to the mean demo start of the object, and use the world axis (+x, -x, +y or -y)
           closest to it (the larger component, with its sign). On t59 this gives -y and 7 cm (the shift used there);
           the exact vector would give (-0.40, -0.92) and 9 cm, not the t59 construction.
  directions 2, 3  the perpendiculars, in this fixed order: direction 1 rotated by +90 deg (counter-clockwise seen
           from above), then by -90 deg.
  shift size  along direction u: width of the demo range along u (max - min of the 50 demo starts projected on u)
           plus 2 cm, rounded up to a whole cm. Every start then sits at least 2 cm beyond the demo range along u
           when its unshifted position lies in that range (the per start margin is reported).
  starts   the 50 official init states i = 0..49 with the object's free joint xy moved by shift * u; nothing else
           changes (other objects, fixtures, arm, time, velocities).
  settle check (t59's)  after openpi's 10 settle steps with the dummy action, a start passes if the object touches
           the table and nothing else, its orientation is within 5 deg of the unshifted start's after the same
           settle, it moved < 5 mm in xy while settling, and every other object ends within 2 mm (xy) of where it
           ends from the unshifted start.
  choice   direction 1 if >= 40 of 50 pass, else the first of directions 2, 3 with >= 40, else the task is skipped.
  trials   trial j uses kept start j, env.seed(7 + 40000 + j), noise default_rng((7, 40000 + j)) (HARD_OFFSET 40000,
           disjoint from eval 0-49, t59 harder starts 10000+, screening 20000+ and drop pilots 30000+).

--geometry-only computes the direction and shift sizes and writes nothing (used for t59 to show the rule's value).
Output: /workspace/runs/stage4_screen/hard/t<T>/init_states.npy (n, D) and manifest.json.

Usage
  PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
      scripts/multitask/build_hard_starts_mt.py --task 14
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stage3"))
import sim_mt as M  # noqa: E402
import t59_sim as S  # noqa: E402
from pps_real import libero_obs  # noqa: E402
from pps_real import perturb as _perturb  # noqa: E402

OUT = pathlib.Path("/workspace/runs/stage4_screen/hard")
HARD_OFFSET = 40000
MARGIN_M = 0.02
MIN_PASS = 40
N_STARTS = 50


def settle(env, state, seed):
    env.seed(seed)
    env.reset()
    env.set_init_state(state)
    for _ in range(S.SETTLE):
        env.step(libero_obs.DUMMY_ACTION)


def others_xy(raw, obj):
    sim = raw.sim
    return {n: sim.data.body_xpos[sim.model.body_name2id(o.root_body)][:2].copy()
            for n, o in raw.objects_dict.items() if n != obj}


def shift_size(proj):
    w = float(proj.max() - proj.min())
    return w, math.ceil(round((w + MARGIN_M) * 100, 9)) / 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--geometry-only", action="store_true")
    a = ap.parse_args()
    T = a.task
    spec = M.task_spec(T)
    env, task, init_states = M.make_env(T)
    raw = env.env
    env.reset()
    probe = M.Probe(env, spec)
    adr = probe.pert.qpos_adr
    z = np.load("/workspace/data/t%d_demos/states.npz" % T)
    f0 = z["states"][z["frame_idx"] == 0]
    assert len(f0) == 50, len(f0)
    P, G = [], []
    for s in f0:
        env.reset()
        env.set_init_state(np.asarray(s, dtype=np.float64))
        P.append(raw.sim.data.qpos[adr:adr + 2].copy())
        G.append(probe.target_xy())
    P, G = np.array(P), np.array(G)
    away = P.mean(0) - G.mean(0)
    ax = int(np.argmax(np.abs(away)))
    d1 = np.zeros(2)
    d1[ax] = float(np.sign(away[ax]))
    dirs = [("away", d1), ("perp_plus90", np.array([-d1[1], d1[0]])), ("perp_minus90", np.array([d1[1], -d1[0]]))]
    geo = {"task": T, "name": spec["name"], "obj": spec["obj"], "target": spec["target"],
           "demo_obj_xy_min": P.min(0).tolist(), "demo_obj_xy_max": P.max(0).tolist(),
           "goal_xy_mean": G.mean(0).tolist(), "demo_obj_xy_mean": P.mean(0).tolist(),
           "away_exact_unit_xy": (away / np.linalg.norm(away)).tolist(), "directions": []}
    for name, u in dirs:
        w, sh = shift_size(P @ u)
        geo["directions"].append({"name": name, "unit_xy": u.tolist(), "demo_width_m": w, "shift_m": sh})
    print(json.dumps(geo), flush=True)
    if a.geometry_only:
        env.close()
        return
    out = OUT / ("t%d" % T)
    out.mkdir(parents=True, exist_ok=True)
    tried, chosen = [], None
    for (name, u), g in zip(dirs, geo["directions"]):
        sh = g["shift_m"]
        demo_max = float((P @ u).max())
        rows, kept = [], []
        for i in range(N_STARTS):
            base = np.array(init_states[i], dtype=np.float64)
            settle(env, base, S.SEED + i)
            ref_others = others_xy(raw, spec["obj"])
            ref_quat = raw.sim.data.qpos[adr + 3:adr + 7].copy()
            unshifted_proj = float(base[1 + adr:1 + adr + 2] @ u)
            mod = base.copy()
            mod[1 + adr:1 + adr + 2] += sh * u
            settle(env, mod, S.SEED + i)
            sim = raw.sim
            q = sim.data.qpos
            quat = q[adr + 3:adr + 7]
            dot = abs(float(np.dot(quat / np.linalg.norm(quat), ref_quat / np.linalg.norm(ref_quat))))
            tilt = float(np.degrees(2 * np.arccos(min(1.0, dot))))
            pert = _perturb.Perturber(env, perturbation="none")
            assert pert.obj is raw.objects_dict[spec["obj"]]
            cs = _perturb.object_contacts(sim, pert.obj_geoms)
            on_table = any("table" in c for c in cs)
            other_c = [c for c in cs if "table" not in c]
            xy = q[adr:adr + 2].copy()
            moved = float(np.linalg.norm(xy - mod[1 + adr:1 + adr + 2]))
            now = others_xy(raw, spec["obj"])
            other_shift = float(max(np.linalg.norm(now[n] - ref_others[n]) for n in now)) if now else 0.0
            ok = on_table and not other_c and tilt < 5 and moved < 0.005 and other_shift < 0.002
            rows.append({"base_init_state": i, "obj_xy_spawn": mod[1 + adr:1 + adr + 2].tolist(),
                         "obj_xy_settled": xy.tolist(), "tilt_deg": tilt, "settle_xy_move_m": moved,
                         "other_objects_max_xy_shift_m": other_shift, "on_table": on_table,
                         "other_contacts": other_c, "unshifted_inside_demo_range": bool(
                             (P @ u).min() - 1e-9 <= unshifted_proj <= demo_max + 1e-9),
                         "beyond_demo_range_m": float(xy @ u) - demo_max, "kept": bool(ok)})
            if ok:
                kept.append(mod)
        n_ok = len(kept)
        tried.append({"direction": name, "unit_xy": u.tolist(), "shift_m": sh, "n_pass": n_ok, "rows": rows})
        print("t%d %s shift %.2f m: %d/%d pass" % (T, name, sh, n_ok, N_STARTS), flush=True)
        if n_ok >= MIN_PASS:
            chosen = (name, kept, rows)
            break
    man = {"rule": __doc__.split("\n\n")[1], "geometry": geo, "tried": tried, "hard_offset": HARD_OFFSET,
           "min_pass": MIN_PASS, "skipped": chosen is None}
    if chosen is None:
        man["skip_reason"] = "no direction with >= %d of %d starts passing the settle check" % (MIN_PASS, N_STARTS)
    else:
        name, kept, rows = chosen
        krows = [r for r in rows if r["kept"]]
        man.update({"direction": name, "n_kept": len(kept),
                    "beyond_demo_range_m": [min(r["beyond_demo_range_m"] for r in krows),
                                            max(r["beyond_demo_range_m"] for r in krows)],
                    "trials": [{"trial": j, "base_init_state": r["base_init_state"],
                                "env_seed": S.SEED + HARD_OFFSET + j, "noise_seed": [S.SEED, HARD_OFFSET + j]}
                               for j, r in enumerate(krows)]})
        np.save(out / "init_states.npy", np.stack(kept))
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("t%d: %s" % (T, "SKIPPED" if chosen is None else "%s, %d kept" % (chosen[0], len(chosen[1]))))
    env.close()


if __name__ == "__main__":
    main()
