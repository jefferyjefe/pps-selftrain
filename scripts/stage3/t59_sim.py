"""Stage 3 simulator side for t59: env setup, per step probe, mid-carry event, exact snapshots and the drop.

Runs in openpi's examples/libero venv (Python 3.8), so it stays 3.8 compatible. Reuses pps_real.libero_obs for the
observation processing (identical to openpi's LIBERO client and to the images the proxies were trained on) and
pps_real.perturb.Perturber for the grasp, finger width and gripper target helpers and for the forced open gripper.

Trial and episode conventions (same as milestone 0 and stage 2, which follow openpi examples/libero/main.py)
  - 10 settle steps with the dummy action, then up to 400 policy steps (openpi's libero_90 budget). The loop runs while
    t < 400 + 10, and success is the `done` LIBERO's env.step returns, i.e. env._check_success() on the BDDL goal
    (In tomato_sauce_1 wooden_tray_1_contain_region).
  - Replan every 5 actions from a 10 step chunk.
  - Trial i uses LIBERO init state i % n_init, env.seed(7 + i) before reset, and flow matching noise for its j-th
    policy call from the j-th (10, 32) standard normal draw of numpy default_rng((7, i)).

Mid-carry event (thresholds below, all must hold after an env step, first such step fires)
  1. Gripper closed on the bottle. Both fingers touch the bottle (Perturber.grasping, any finger collision geom) and
     the executed action commands the gripper closed (action[6] > 0).
  2. Lifted clearly off the table. Bottle body z at least LIFT_M above its height at the end of the settle steps, and
     no contact between the bottle and the table.
  3. Not at the tray yet. Horizontal distance from the bottle body to the tray's contain region rectangle (the box
     LIBERO's success check uses) at least TRAY_MARGIN_M, and no contact between the bottle and the tray.

Snapshot (everything the next env step and the policy client read)
  - MuJoCo integration state via mj_getState(mjSTATE_INTEGRATION): time, qpos, qvel, act, qacc_warmstart, ctrl,
    applied forces, eq_active, mocap, userdata, plugin state.
  - Plain data attributes (owned numpy arrays, scalars, robosuite buffers) of the robosuite env, robot, OSC
    controller, gripper (its integrated finger target current_action) and every observable (sampling timers and the
    last observed values, which are what env._get_observations returns), plus the env's obs cache.
  - numpy's global RNG state (env.seed sets it).
  - Model body_pos / body_quat (LIBERO's reset writes fixture poses into the model), checked on restore.
  - Client state: step counter, remaining action queue, the noise generator's state, policy call count, bottle start
    height, trial identifiers, and the processed policy input at that step.
"""

import copy
import pathlib
import pickle

import mujoco
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import buffers as _rs_buffers

import pps_real
from pps_real import libero_obs
from pps_real import perturb as _perturb

SEED = 7
REPLAN_STEPS = 5
MAX_STEPS = libero_obs.MAX_STEPS_LIBERO_90  # 400
SETTLE = libero_obs.SETTLE_STEPS  # 10
T_END = MAX_STEPS + SETTLE

LIFT_M = 0.05
TRAY_MARGIN_M = 0.05
TRAY_SITE = "wooden_tray_1_contain_region"
TRAY_PREFIX = "wooden_tray_1"
DROP_FRAMES = 10
MJ_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def make_env():
    suite = benchmark.get_benchmark_dict()[pps_real.T59_SUITE]()
    task = suite.get_task(pps_real.T59_TASK_ID)
    assert task.name == pps_real.T59_NAME
    init_states = suite.get_task_init_states(pps_real.T59_TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    return env, task, init_states


def trial_setup(trial, n_init):
    return {"trial": int(trial), "init_state": int(trial % n_init), "env_seed": SEED + int(trial),
            "noise_seed": [SEED, int(trial)]}


class Probe(object):
    """Per step quantities read from the sim. Wraps a Perturber (stage 2) for its helpers."""

    def __init__(self, env):
        self.env = env
        self.raw = env.env
        self.pert = _perturb.Perturber(env, perturbation="none")
        sim = self.raw.sim
        self.tray_body = sim.model.body_name2id(TRAY_PREFIX + "_main") if _has_body(sim, TRAY_PREFIX + "_main") else None
        self.z0 = None

    def start(self):
        self.pert.start()
        self.z0 = self.pert.z0

    def tray_box(self):
        site = self.raw.object_sites_dict[TRAY_SITE]
        pos = self.raw.sim.data.get_site_xpos(TRAY_SITE).copy()
        mat = self.raw.sim.data.get_site_xmat(TRAY_SITE).copy()
        half = np.abs(mat @ site.size)
        lb, ub = pos - half, pos + half
        lb[2] -= 0.01  # as in SiteObject.in_box
        return lb, ub

    def tray_dist_xy(self, p):
        lb, ub = self.tray_box()
        d = np.maximum(np.maximum(lb[:2] - p[:2], p[:2] - ub[:2]), 0.0)
        return float(np.linalg.norm(d))

    def contacts(self):
        return _perturb.object_contacts(self.raw.sim, self.pert.obj_geoms)

    def read(self, action):
        p = self.pert.obj_pos()
        cs = self.contacts()
        return {
            "obj": p, "dz": float(p[2] - self.z0), "grasp": self.pert.grasping(), "a6": float(action[6]),
            "w": self.pert.width(), "tw": self.pert.target_width(),
            "table": any(c.startswith(_perturb.TABLE_PREFIX) or c.startswith("body:" + _perturb.TABLE_PREFIX) for c in cs),
            "tray": any(TRAY_PREFIX in c for c in cs),
            "finger": any(("finger" in c) or ("gripper" in c) for c in cs),
            "tray_dxy": self.tray_dist_xy(p),
        }


def _has_body(sim, name):
    try:
        sim.model.body_name2id(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def mid_carry(r):
    """The mid-carry event predicate on one Probe.read() record."""
    return bool(r["grasp"] and r["a6"] > 0 and r["dz"] >= LIFT_M and not r["table"]
                and r["tray_dxy"] >= TRAY_MARGIN_M and not r["tray"])


EVENT_DEF = {
    "name": "mid_carry",
    "gripper_closed_on_bottle": "both fingers' collision geoms touch the bottle (pps_real.perturb.Perturber.grasping) "
                                "and the executed action[6] > 0 (commanded closed)",
    "lifted": "bottle body z - z_settle >= %.3f m and no bottle-table contact" % LIFT_M,
    "not_at_tray": "xy distance from bottle body to the tray contain region rectangle (LIBERO's success box) >= %.3f m "
                   "and no bottle-tray contact" % TRAY_MARGIN_M,
    "fires": "first env step after the settle steps at which all hold",
    "LIFT_M": LIFT_M, "TRAY_MARGIN_M": TRAY_MARGIN_M,
}


# ---------------------------------------------------------------- snapshots

_PLAIN = (float, int, bool, str, type(None), np.floating, np.integer, np.bool_)
_BUFS = tuple(getattr(_rs_buffers, n) for n in ("RingBuffer", "DeltaBuffer", "DelayBuffer") if hasattr(_rs_buffers, n))


def _plain_attrs(obj):
    out = {}
    for k, v in vars(obj).items():
        if isinstance(v, np.ndarray):
            if v.base is None:  # owned data, not a view into mjData
                out[k] = v.copy()
        elif isinstance(v, _PLAIN):
            out[k] = v
        elif isinstance(v, _BUFS):
            out[k] = copy.deepcopy(v)
    return out


def _set_attrs(obj, attrs):
    for k, v in attrs.items():
        setattr(obj, k, copy.deepcopy(v))


def _objects(env):
    raw = env.env
    robot = raw.robots[0]
    objs = {"env": raw, "robot": robot, "controller": robot.controller, "gripper": robot.gripper}
    for name, ob in raw._observables.items():  # noqa: SLF001
        objs["obs:" + name] = ob
    return objs


def mj_state(env):
    m, d = env.env.sim.model._model, env.env.sim.data._data  # noqa: SLF001
    s = np.empty(mujoco.mj_stateSize(m, MJ_SPEC), dtype=np.float64)
    mujoco.mj_getState(m, d, s, MJ_SPEC)
    return s


def take_snapshot(env, client):
    sim = env.env.sim
    return {
        "mj_spec": MJ_SPEC, "mj_state": mj_state(env),
        "attrs": {k: _plain_attrs(o) for k, o in _objects(env).items()},
        "obs_cache": copy.deepcopy(getattr(env.env, "_obs_cache", None)),
        "np_random": np.random.get_state(),
        "body_pos": sim.model.body_pos.copy(), "body_quat": sim.model.body_quat.copy(),
        "client": client,
    }


def restore_snapshot(env, init_states, snap):
    """Puts env exactly in the snapshot's state. Returns the observation dict (the observables' restored values)."""
    c = snap["client"]
    env.seed(c["env_seed"])
    env.reset()
    env.set_init_state(init_states[c["init_state"]])
    sim = env.env.sim
    assert np.array_equal(sim.model.body_pos, snap["body_pos"]) and np.array_equal(sim.model.body_quat, snap["body_quat"]), \
        "model body poses after reset differ from the snapshot's"
    m, d = sim.model._model, sim.data._data  # noqa: SLF001
    mujoco.mj_setState(m, d, snap["mj_state"], snap["mj_spec"])
    objs = _objects(env)
    for k, attrs in snap["attrs"].items():
        _set_attrs(objs[k], attrs)
    if snap["obs_cache"] is not None:
        env.env._obs_cache = copy.deepcopy(snap["obs_cache"])  # noqa: SLF001
    np.random.set_state(snap["np_random"])
    # No sim.forward() here. mj_forward rewrites qacc_warmstart, so an extra call gives the constraint solver a
    # different warm start than the original trajectory had (seen as 1e-8 qpos differences on the next step). The
    # next env.step starts with sim.forward() itself, exactly as on the original trajectory, and the observation
    # returned below is the observables' restored values, not a fresh read of mjData.
    return env.env._get_observations()  # noqa: SLF001


def save_snapshot(path, snap):
    with open(path, "wb") as fh:
        pickle.dump(snap, fh, protocol=4)


def load_snapshot(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def force_open(gripper):
    """Gripper's integrated finger target fully open (see pps_real.perturb release)."""
    gripper.current_action = np.array([1.0, -1.0])


DROP_ACTION = [0.0] * 6 + [-1.0]
