from __future__ import annotations

import numpy as np
import gymnasium as gym
import sys

ROOT = __file__.split("act_project")[0] + "panda_mujoco_gym_ref"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import panda_mujoco_gym  # noqa: F401


def diagnose_observation(obs: dict) -> None:
    o = obs["observation"]
    labels = [
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_vx",
        "ee_vy",
        "ee_vz",
        "finger_width",
        "obj_x",
        "obj_y",
        "obj_z",
        "obj_rot_x",
        "obj_rot_y",
        "obj_rot_z",
        "obj_vx",
        "obj_vy",
        "obj_vz",
        "obj_wx",
        "obj_wy",
        "obj_wz",
    ]
    print("obs len", len(o))
    for i, (lab, val) in enumerate(zip(labels, o)):
        print(f"[{i:02d}] {lab:12s}: {val:.5f}")
    print("achieved_goal", obs["achieved_goal"])
    print("desired_goal", obs["desired_goal"])


def test_gripper(env: gym.Env, steps: int = 100) -> None:
    obs, _ = env.reset()
    readings = []
    # open wide first
    for _ in range(20):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
    open_width = obs["observation"][6]

    for _ in range(steps):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0], dtype=np.float32))
        readings.append(obs["observation"][6])
    print("gripper open_width", open_width, "final", readings[-1], "delta", readings[-1] - open_width)


def test_movement(env: gym.Env, steps: int = 10) -> None:
    obs, _ = env.reset()
    start = obs["observation"][:3].copy()
    for _ in range(steps):
        obs, _, _, _, _ = env.step(np.array([0, 0, -1.0, 0], dtype=np.float32))
    end = obs["observation"][:3]
    print("move start", start, "end", end, "delta", end - start, "per_step", (end - start) / steps)


def test_simple_grasp(env: gym.Env) -> None:
    obs, _ = env.reset()
    ee = obs["observation"][:3]
    obj = obs["observation"][7:10]
    print("init ee", ee, "obj", obj, "xy_err", np.linalg.norm(ee[:2] - obj[:2]), "z_err", ee[2] - obj[2])
    # align xy
    for _ in range(80):
        ee = obs["observation"][:3]
        obj = obs["observation"][7:10]
        dx = np.clip((obj[0] - ee[0]) * 10, -1, 1)
        dy = np.clip((obj[1] - ee[1]) * 10, -1, 1)
        obs, _, _, _, _ = env.step(np.array([dx, dy, 0, 1.0], dtype=np.float32))
        if np.linalg.norm(obs["observation"][:2] - obj[:2]) < 0.02:
            break
    # descend
    obj = obs["observation"][7:10]
    target_z = obj[2] + 0.02
    for _ in range(80):
        ee = obs["observation"][:3]
        z_err = ee[2] - target_z
        if z_err < 0.01:
            break
        dz = np.clip(-z_err * 10, -1, 1)
        obs, _, _, _, _ = env.step(np.array([0, 0, dz, 1.0], dtype=np.float32))
    # close
    for _ in range(80):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, -1.0], dtype=np.float32))
    obj_before = obs["observation"][9]
    # lift
    for _ in range(60):
        obs, _, _, _, _ = env.step(np.array([0, 0, 1.0, -1.0], dtype=np.float32))
    obj_after = obs["observation"][9]
    print("obj z change", obj_before, "->", obj_after, "lifted", obj_after > obj_before + 0.05)


def main() -> None:
    env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode=None)
    obs, _ = env.reset()
    diagnose_observation(obs)
    test_gripper(env)
    test_movement(env)
    test_simple_grasp(env)
    env.close()


if __name__ == "__main__":
    main()
