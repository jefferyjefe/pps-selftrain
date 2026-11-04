"""Perturbation wrapper for LIBERO-90 t59 recovery tests. Runs in the LIBERO venv and stays Python 3.8 compatible.

An event detector watches every env step and fires once per episode. When it fires, one perturbation is applied.

Events
  grasp   first env step at which the gripper has closed on the tomato sauce bottle. All four must hold after the
          step.
          1. Both fingers touch the bottle (robosuite _check_grasp with the finger geom groups, so any collision
             geom of the left finger and any of the right finger, pads included. The pads alone almost never touch
             this bottle).
          2. The policy commands the gripper closed (action[6] > 0).
          3. The gripper's internal finger target is at least SQUEEZE_M narrower than the actual opening, so the
             fingers are pressing rather than still travelling to an open target.
          4. The opening shrank by less than STALL_M on this step, so the bottle has stopped the fingers. This rules
             out fingers that brush the bottle while sweeping shut past it.
          Only steps after the settle steps are considered.

Perturbations
  none      detect and log the event, change nothing. The paired control.
  teleport  move the bottle --teleport-dist (default 0.05 m) in a random horizontal direction, keeping its height
            and orientation and zeroing its velocity. A direction whose new pose would touch anything but the
            table (another object, the tray, the gripper) is rejected and a new one drawn, up to max_tries.
  release   open the gripper for --release-frames (default 3) env steps, starting with the step right after the
            event. robosuite's Panda gripper integrates its action (the finger target moves 0.25 of its range per
            env step toward the sign of action[6]), so sending -1 for 3 steps after a close would only relax the
            squeeze from +1 to +0.25 and never open the fingers. Instead the gripper's internal target is set fully
            open on each of those steps and action[6] is -1. The arm part of the policy's action is kept. After the
            release the policy's own close command needs about 8 steps to close the target again.

The policy is not told. The action chunk it is executing carries on, and it sees the perturbed scene at its next
replanning step, as it would see any disturbance.
"""

import numpy as np

EVENTS = ("grasp",)
SQUEEZE_M = 0.005
STALL_M = 0.002
PERTURBATIONS = ("none", "teleport", "release")
OBJECT = "tomato_sauce_1"
TABLE_PREFIX = "living_room_table"


def _geom_names(sim, ids):
    """Geom names, falling back to the owning body's name for unnamed geoms."""
    out = []
    for i in ids:
        n = sim.model.geom_id2name(int(i))
        out.append(n if n else "body:" + (sim.model.body_id2name(int(sim.model.geom_bodyid[int(i)])) or "?"))
    return out


def object_contacts(sim, object_geoms):
    """Names of geoms touching any geom of the object."""
    ids = set(sim.model.geom_name2id(g) for g in object_geoms)
    other = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if c.geom1 in ids and c.geom2 not in ids:
            other.append(c.geom2)
        elif c.geom2 in ids and c.geom1 not in ids:
            other.append(c.geom1)
    return sorted(set(_geom_names(sim, other)))


def closes_on(grasp, a6, width, prev_width, target_width):
    """The grasp event predicate on logged quantities (widths in metres). See the module docstring."""
    if prev_width is None:
        return False
    return bool(grasp and a6 > 0 and target_width < width - SQUEEZE_M and width - prev_width > -STALL_M)


class Perturber(object):
    def __init__(self, env, event="grasp", perturbation="none", rng=None, teleport_dist=0.05, release_frames=3,
                 max_tries=50):
        assert event in EVENTS, event
        assert perturbation in PERTURBATIONS, perturbation
        self.env = env
        self.raw = env.env  # the robosuite env inside LIBERO's OffScreenRenderEnv
        self.event = event
        self.perturbation = perturbation
        self.rng = rng if rng is not None else np.random.default_rng()
        self.teleport_dist = float(teleport_dist)
        self.release_frames = int(release_frames)
        self.max_tries = int(max_tries)
        self.gripper = self.raw.robots[0].gripper
        ig = self.gripper.important_geoms
        self.finger_groups = [ig["left_finger"], ig["right_finger"]]
        sim = self.raw.sim
        act = [sim.model.actuator_name2id(a) for a in self.gripper.actuators]
        cr = sim.model.actuator_ctrlrange[act]
        self._ctrl_bias = 0.5 * (cr[:, 1] + cr[:, 0])
        self._ctrl_weight = 0.5 * (cr[:, 1] - cr[:, 0])
        self.prev_width = None
        self.obj = self.raw.objects_dict[OBJECT]
        self.obj_geoms = list(self.obj.contact_geoms)
        jname = self.obj.joints[0]
        self.qpos_adr = self.raw.sim.model.get_joint_qpos_addr(jname)[0]
        self.qvel_adr = self.raw.sim.model.get_joint_qvel_addr(jname)[0]
        self.fired_step = None
        self.info = None
        self._release_left = 0
        self.z0 = None
        self.prev_grasp = False

    def start(self):
        """Call once after reset, settle and before the first policy step."""
        self.z0 = float(self.obj_pos()[2])

    def obj_pos(self):
        return self.raw.sim.data.qpos[self.qpos_adr:self.qpos_adr + 3].copy()

    def width(self):
        """Finger opening in metres, left finger qpos minus right finger qpos."""
        q = self.raw.sim.data.qpos[self.raw.robots[0]._ref_gripper_joint_pos_indexes]  # noqa: SLF001
        return float(q[0] - q[1])

    def target_width(self):
        """Opening the finger position actuators are driving toward, from the gripper's integrated command."""
        tgt = self._ctrl_bias + self._ctrl_weight * np.asarray(self.gripper.current_action, dtype=float)
        return float(tgt[0] - tgt[1])

    def grasping(self):
        return bool(self.raw._check_grasp(gripper=self.finger_groups, object_geoms=self.obj_geoms))  # noqa: SLF001

    def pad_grasping(self):
        """robosuite's default grasp test, both fingerpads. Logged for comparison."""
        return bool(self.raw._check_grasp(gripper=self.gripper, object_geoms=self.obj_geoms))  # noqa: SLF001

    def filter_action(self, action):
        """Apply any action override. Call on the action about to be sent to env.step."""
        action = list(action)
        if self._release_left > 0:
            action[6] = -1.0
            # fully open target. Closing adds [-1, 1] * speed, so open is [1, -1].
            self.gripper.current_action = np.array([1.0, -1.0])
            self._release_left -= 1
            self.info.setdefault("release_actions_overridden", 0)
            self.info["release_actions_overridden"] += 1
        return action

    @property
    def overriding(self):
        return self._release_left > 0

    def after_step(self, t, action, obs):
        """Call after every env step past the settle steps with the action that was executed and the obs it returned.

        Returns (obs, fired). obs is refreshed if the perturbation changed the scene."""
        if self.fired_step is not None:
            return obs, False
        grasp = self.grasping()
        prev_grasp = self.prev_grasp
        self.prev_grasp = grasp
        w, w_prev, w_tgt = self.width(), self.prev_width, self.target_width()
        self.prev_width = w
        if not closes_on(grasp, float(action[6]), w, w_prev, w_tgt):
            return obs, False
        self.fired_step = int(t)
        pos = self.obj_pos()
        self.info = {
            "event": self.event, "perturbation": self.perturbation, "step": int(t),
            "contact_on_previous_step": bool(prev_grasp),
            "gripper_qpos": [float(v) for v in obs["robot0_gripper_qpos"]],
            "gripper_action": float(action[6]),
            "width_m": w, "width_prev_m": w_prev, "target_width_m": w_tgt,
            "obj_pos": [float(v) for v in pos], "obj_height_above_start_m": float(pos[2] - self.z0),
            "eef_pos": [float(v) for v in obs["robot0_eef_pos"]],
            "obj_contacts": object_contacts(self.raw.sim, self.obj_geoms),
            "pad_grasp": self.pad_grasping(),
        }
        if self.perturbation == "teleport":
            obs = self._teleport()
        elif self.perturbation == "release":
            self._release_left = self.release_frames
        return obs, True

    def _teleport(self):
        sim = self.raw.sim
        qpos0 = sim.data.qpos.copy()
        qvel0 = sim.data.qvel.copy()
        p0 = qpos0[self.qpos_adr:self.qpos_adr + 3].copy()
        tries = []
        for k in range(self.max_tries):
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            d = self.teleport_dist * np.array([np.cos(theta), np.sin(theta)])
            qpos = qpos0.copy()
            qpos[self.qpos_adr:self.qpos_adr + 2] = p0[:2] + d
            sim.data.qpos[:] = qpos
            qvel = qvel0.copy()
            qvel[self.qvel_adr:self.qvel_adr + 6] = 0.0
            sim.data.qvel[:] = qvel
            sim.forward()
            contacts = object_contacts(sim, self.obj_geoms)
            bad = [c for c in contacts if not (c.startswith(TABLE_PREFIX) or c.startswith("body:" + TABLE_PREFIX))]
            tries.append({"theta": theta, "blocking_contacts": bad})
            if not bad:
                break
        ok = not tries[-1]["blocking_contacts"]  # if no direction was clear, the last one is kept and logged
        p1 = self.obj_pos()
        self.raw._update_observables(force=True)  # noqa: SLF001
        obs = self.raw._get_observations()  # noqa: SLF001
        self.info.update({
            "teleport_theta": tries[-1]["theta"], "teleport_tries": len(tries), "teleport_clear": bool(ok),
            "teleport_rejected": tries[:-1], "obj_pos_after": [float(v) for v in p1],
            "teleport_horizontal_m": float(np.linalg.norm(p1[:2] - p0[:2])),
            "teleport_vertical_m": float(p1[2] - p0[2]),
            "grasp_after_teleport": self.grasping(),
        })
        return obs
