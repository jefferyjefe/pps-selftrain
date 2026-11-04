"""Observation processing shared by every LIBERO client in stage 2. Python 3.8 compatible.

Matches openpi examples/libero/main.py and scripts/m0/eval_libero.py exactly. Images are rotated
180 degrees, resized with padding to 224 and converted to uint8. State is eef pos, eef axis angle and
gripper qpos. pi05_libero ignores state (discrete_state_input is False and pi0.5 puts no state token in
the suffix) but openpi's input transform still expects it, so it is passed through.
"""

import math

import numpy as np

DUMMY_ACTION = [0.0] * 6 + [-1.0]
NOISE_SHAPE = (10, 32)  # pi05_libero action_horizon x action_dim
MAX_STEPS_LIBERO_90 = 400
SETTLE_STEPS = 10


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def process_images(obs):
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    return img, wrist


def policy_state(obs):
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))


def policy_input(obs, prompt):
    img, wrist = process_images(obs)
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": policy_state(obs),
        "prompt": str(prompt),
    }


def episode_noise_rng(seed, episode):
    """Per episode flow matching noise stream, same as milestone 0."""
    return np.random.default_rng((seed, episode))
