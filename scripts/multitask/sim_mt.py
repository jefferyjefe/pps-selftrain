"""Simulator side for the stage 4 multi-task screen: any libero_90 task, same conventions as scripts/stage3/t59_sim.py.

Runs in openpi's examples/libero venv (Python 3.8), so it stays 3.8 compatible.

What is the same as t59 (imported, not copied): the env (OffScreenRenderEnv at 256 px), the episode loop of
scripts/faithful/run_faithful.py (10 settle steps with the dummy action, then t < 410, replan every 5 actions from a 10
step chunk, success = the `done` LIBERO's env.step returns), image and state processing (pps_real.libero_obs), the
snapshot / restore code and the drop helpers of t59_sim.

What is per task: the carried object and the goal target, both read from the task's BDDL goal (first predicate of the
form (On|In obj target)), and the scene's table (any contact geom, or the body of an unnamed geom, whose name contains
"table"). Probe.read returns the same keys as t59_sim.Probe.read, so run_t59.STEP_KEYS and the npz layout are unchanged:
  obj, dz            carried object's position, height above its post-settle height
  grasp, a6, w, tw   both finger geom groups touch the carried object (robosuite _check_grasp), executed action[6],
                     finger opening and target opening (pps_real.perturb.Perturber helpers)
  table              the carried object touches the table
  tray               the carried object touches the goal target (its object or fixture; never for table regions)
  tray_dxy           horizontal distance from the carried object to the goal target: to the target site's box when the
                     target is a site (LIBERO's in_box box), else to the target object's body position
  finger             the carried object touches any gripper geom
(The key names "tray" and "tray_dxy" are kept from t59 so that the shared loop and npz layout need no change.)

Screening trials (stage 4 protocol, results/multitask/PROTOCOL_multitask.md section d): screen trial j uses LIBERO init
state j % 50, env.seed(7 + SCREEN_OFFSET + j) and noise default_rng((7, SCREEN_OFFSET + j)), SCREEN_OFFSET = 20000.
"""

import pathlib
import re

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from pps_real import perturb as _perturb

SUITE = "libero_90"
SEED = 7
SCREEN_OFFSET = 20000
PILOT_OFFSET = 30000  # drop calibration pilot trials (Step 5)

GOAL_RE = re.compile(r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)")


def task_spec(task_id):
    """Task name, prompt, scene, carried object and goal target from LIBERO's benchmark and the task's BDDL."""
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    text = bddl.read_text()
    goal = text.split("(:goal", 1)[1]
    m = GOAL_RE.search(goal)
    assert m is not None, (task_id, goal)
    scene = task.name.split("_")
    scene = "_".join(scene[: next(i for i, w in enumerate(scene) if w.startswith("SCENE")) + 1])
    return {"task_id": int(task_id), "name": task.name, "prompt": str(task.language), "scene": scene,
            "bddl": str(bddl), "goal_predicates": GOAL_RE.findall(goal), "obj": m.group(2), "target": m.group(3),
            "goal_kind": m.group(1)}


def make_env(task_id):
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    return env, task, init_states


def trial_setup(trial, n_init, offset):
    """offset 0: eval trial convention (t59_sim.trial_setup); SCREEN_OFFSET: screening; PILOT_OFFSET: drop pilots."""
    i = int(offset) + int(trial)
    return {"trial": int(trial), "trial_index": i, "seed_offset": int(offset), "init_state": int(trial % n_init),
            "env_seed": SEED + i, "noise_seed": [SEED, i]}


class Probe(object):
    """t59_sim.Probe for any task (see module doc). Wraps a Perturber for its finger helpers, retargeted to the task's
    carried object."""

    def __init__(self, env, spec):
        self.env = env
        self.raw = env.env
        self.spec = spec
        _perturb.OBJECT = spec["obj"]  # Perturber reads the module constant in __init__
        self.pert = _perturb.Perturber(env, perturbation="none")
        assert self.pert.obj is self.raw.objects_dict[spec["obj"]]
        tgt = spec["target"]
        self.target_site = tgt if tgt in self.raw.object_sites_dict else None
        owner = None
        if self.target_site is None:
            owner = tgt
        else:
            for name in list(self.raw.objects_dict) + list(self.raw.fixtures_dict):
                if tgt.startswith(name + "_"):
                    owner = name
        self.target_owner = owner  # object / fixture whose geoms count as "tray" contact (None for table regions)
        self.target_body = None
        if self.target_site is None:
            self.target_body = self.raw.sim.model.body_name2id(self.raw.objects_dict[tgt].root_body)
        self.z0 = None

    def start(self):
        self.pert.start()
        self.z0 = self.pert.z0

    def target_dist_xy(self, p):
        sim = self.raw.sim
        if self.target_site is not None:
            site = self.raw.object_sites_dict[self.target_site]
            pos = sim.data.get_site_xpos(self.target_site).copy()
            mat = sim.data.get_site_xmat(self.target_site).copy()
            half = np.abs(mat @ site.size)
            lb, ub = pos - half, pos + half
            d = np.maximum(np.maximum(lb[:2] - p[:2], p[:2] - ub[:2]), 0.0)
            return float(np.linalg.norm(d))
        return float(np.linalg.norm(sim.data.body_xpos[self.target_body][:2] - p[:2]))

    def read(self, action):
        p = self.pert.obj_pos()
        cs = _perturb.object_contacts(self.raw.sim, self.pert.obj_geoms)
        own = self.target_owner
        return {
            "obj": p, "dz": float(p[2] - self.z0), "grasp": self.pert.grasping(), "a6": float(action[6]),
            "w": self.pert.width(), "tw": self.pert.target_width(),
            "table": any("table" in c for c in cs),
            "tray": bool(own) and any(own in c for c in cs),
            "finger": any(("finger" in c) or ("gripper" in c) for c in cs),
            "tray_dxy": self.target_dist_xy(p),
        }
