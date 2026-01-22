from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np

# Ensure local packages (panda_mujoco_gym_ref) are importable without installation.
ROOT = Path(__file__).resolve().parents[1]
PANDA_ROOT = ROOT.parent / "panda_mujoco_gym_ref"
for candidate in (ROOT, PANDA_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from panda_mujoco_gym.envs.panda_env import FrankaEnv

CUSTOM_MODEL = str((ROOT / "assets" / "pick_and_place_with_table.xml").resolve())


class FrankaPickAndPlaceTableEnv(FrankaEnv):
    def __init__(self, reward_type: str = "sparse", **kwargs):
        super().__init__(
            model_path=CUSTOM_MODEL,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=False,
            distance_threshold=0.05,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.2,
            **kwargs,
        )


class PickPlaceTableEnv(gym.Wrapper):
    """
    Wrap FrankaPickAndPlaceEnv to place goal on a target table at (0.3, 0.4).
    Success requires the block to stay near the tabletop for several steps to
    avoid early success during fly-over.
    """

    TABLE_POS = np.array([0.3, 0.4, 0.0])
    TABLE_TOP_Z = 0.10
    TABLE_HALF_SIZE = 0.075
    OBJECT_HALF_HEIGHT = 0.02
    TARGET_JITTER = 0.06

    SUCCESS_DISTANCE_THRESHOLD = 0.05
    SUCCESS_HOLD_STEPS = 6
    PLACED_Z_TOLERANCE = 0.03

    def __init__(self, render_mode: str | None = None, randomize_target: bool = False):
        base_env = FrankaPickAndPlaceTableEnv(reward_type="sparse", render_mode=render_mode)
        super().__init__(base_env)
        self.randomize_target = randomize_target
        self.table_center = np.array(
            [
                self.TABLE_POS[0],
                self.TABLE_POS[1],
                self.TABLE_TOP_Z + self.OBJECT_HALF_HEIGHT,
            ]
        )
        self.success_counter = 0

    def reset(self, **kwargs) -> Tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        self.success_counter = 0
        new_goal = self._sample_goal_on_table()
        self._apply_goal(obs, new_goal)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._apply_goal(obs, self.env.unwrapped.goal)

        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        xy_distance = np.linalg.norm(achieved[:2] - desired[:2])
        obj_z = achieved[2]
        target_z = desired[2]

        xy_close = xy_distance < self.SUCCESS_DISTANCE_THRESHOLD
        z_on_table = abs(obj_z - target_z) < self.PLACED_Z_TOLERANCE

        if xy_close and z_on_table:
            self.success_counter += 1
        else:
            self.success_counter = 0

        success_now = self.success_counter >= self.SUCCESS_HOLD_STEPS
        info["is_success"] = success_now
        info["success_counter"] = self.success_counter
        info["xy_distance"] = float(xy_distance)
        info["z_on_table"] = bool(z_on_table)

        return obs, reward, terminated, truncated, info

    def _sample_goal_on_table(self) -> np.ndarray:
        if self.randomize_target:
            max_jitter = min(self.TARGET_JITTER, self.TABLE_HALF_SIZE - 0.01)
            offset = np.array(
                [
                    self.env.np_random.uniform(-max_jitter, max_jitter),
                    self.env.np_random.uniform(-max_jitter, max_jitter),
                    0.0,
                ]
            )
            return self.table_center + offset
        return self.table_center.copy()

    def _apply_goal(self, obs: dict, goal: np.ndarray) -> None:
        self.env.unwrapped.goal = goal
        obs["desired_goal"] = goal


def make_env(render_mode: str | None = None, randomize_target: bool = False) -> PickPlaceTableEnv:
    return PickPlaceTableEnv(render_mode=render_mode, randomize_target=randomize_target)


if __name__ == "__main__":
    print(f"Using custom model: {CUSTOM_MODEL}")
    env = make_env(render_mode="human", randomize_target=False)
    for ep in range(3):
        obs, info = env.reset()
        print(f"Episode {ep}: obj {obs['observation'][7:10]}, goal {obs['desired_goal']}")
        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success"):
                print(f" success at step {step}")
                break
            if terminated or truncated:
                break
    env.close()
