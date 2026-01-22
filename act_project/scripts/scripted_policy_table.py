from __future__ import annotations

import enum
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env  # noqa: E402


class GraspState(enum.Enum):
    APPROACH = enum.auto()
    DESCEND = enum.auto()
    CLOSE = enum.auto()
    LIFT = enum.auto()
    TRANSPORT = enum.auto()
    LOWER = enum.auto()
    RELEASE = enum.auto()
    STABILIZE = enum.auto()
    RETREAT = enum.auto()
    RETURN_HOME = enum.auto()
    DONE = enum.auto()


class ScriptedPickAndPlaceTablePolicy:
    HOME_POSITION = np.array([0.64, 0.0, 0.25])

    def __init__(
        self,
        safe_height: float = 0.22,
        grasp_height: float = 0.02,
        place_height: float = 0.14,
        xy_tol: float = 0.018,
        z_tol: float = 0.012,
        pos_gain: float = 10.0,
        max_action: float = 1.0,
        close_steps: int = 25,
        release_steps: int = 20,
        stabilize_steps: int = 30,
        min_lift_delta: float = 0.05,
        home_position: np.ndarray | None = None,
        home_gain: float = 6.0,
        home_xy_tol: float = 0.02,
        home_z_tol: float = 0.02,
        close_push: float = -0.03,
        max_grasp_retries: int = 3,
    ) -> None:
        self.safe_height = safe_height
        self.grasp_height = grasp_height
        self.place_height = place_height
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.pos_gain = pos_gain
        self.max_action = max_action
        self.close_steps = close_steps
        self.release_steps = release_steps
        self.stabilize_steps = stabilize_steps
        self.min_lift_delta = min_lift_delta
        self.home_position = home_position if home_position is not None else self.HOME_POSITION.copy()

        self.grip_open = 1.0
        self.grip_close = -1.0
        self.home_gain = home_gain
        self.home_xy_tol = home_xy_tol
        self.home_z_tol = home_z_tol
        self.close_push = close_push
        self.max_grasp_retries = max_grasp_retries

        self.state: GraspState = GraspState.APPROACH
        self.step_counter = 0
        self.initial_obj_height: float | None = None
        self.initial_ee_pos: np.ndarray | None = None
        self.grasp_confirmed = False
        self.grasp_retries = 0

    def reset(self) -> None:
        self.state = GraspState.APPROACH
        self.step_counter = 0
        self.initial_obj_height = None
        self.initial_ee_pos = None
        self.grasp_confirmed = False
        self.grasp_retries = 0

    def act(self, obs: dict) -> np.ndarray:
        ee_pos, finger_width, obj_pos, goal = self._extract(obs)

        if self.initial_ee_pos is None:
            self.initial_ee_pos = ee_pos.copy()
        if self.initial_obj_height is None:
            self.initial_obj_height = obj_pos[2]

        if self.state == GraspState.APPROACH:
            target = np.array([obj_pos[0], obj_pos[1], self.safe_height])
            action = self._move_to(ee_pos, target, self.grip_open)
            if self._at_xy(ee_pos, obj_pos) and abs(ee_pos[2] - target[2]) < 0.03:
                self.state = GraspState.DESCEND
                self.step_counter = 0

        elif self.state == GraspState.DESCEND:
            target = np.array([obj_pos[0], obj_pos[1], self.grasp_height])
            action = self._move_to(ee_pos, target, self.grip_open)
            if self._at_xyz(ee_pos, target):
                self.state = GraspState.CLOSE
                self.step_counter = 0

        elif self.state == GraspState.CLOSE:
            action = np.array([0.0, 0.0, self.close_push, self.grip_close], dtype=np.float32)
            self.step_counter += 1
            if self.step_counter >= self.close_steps:
                self.state = GraspState.LIFT
                self.step_counter = 0

        elif self.state == GraspState.LIFT:
            target = np.array([ee_pos[0], ee_pos[1], self.safe_height])
            action = self._move_to(ee_pos, target, self.grip_close)
            obj_lifted = self.initial_obj_height is not None and (obj_pos[2] - self.initial_obj_height) > self.min_lift_delta
            if obj_lifted:
                self.grasp_confirmed = True
            if self._at_xyz(ee_pos, target):
                if obj_lifted:
                    self.state = GraspState.TRANSPORT
                    self.grasp_retries = 0
                else:
                    self.grasp_retries += 1
                    if self.grasp_retries > self.max_grasp_retries:
                        self.state = GraspState.DONE
                    else:
                        self.state = GraspState.APPROACH
                self.step_counter = 0

        elif self.state == GraspState.TRANSPORT:
            target = np.array([goal[0], goal[1], self.safe_height])
            action = self._move_to(ee_pos, target, self.grip_close)
            if self._at_xy(ee_pos, target):
                self.state = GraspState.LOWER
                self.step_counter = 0

        elif self.state == GraspState.LOWER:
            target = np.array([goal[0], goal[1], self.place_height])
            action = self._move_to(ee_pos, target, self.grip_close)
            if self._at_xyz(ee_pos, target):
                self.state = GraspState.RELEASE
                self.step_counter = 0

        elif self.state == GraspState.RELEASE:
            action = np.array([0.0, 0.0, 0.0, self.grip_open], dtype=np.float32)
            self.step_counter += 1
            if self.step_counter >= self.release_steps:
                self.state = GraspState.STABILIZE
                self.step_counter = 0

        elif self.state == GraspState.STABILIZE:
            action = np.array([0.0, 0.0, 0.0, self.grip_open], dtype=np.float32)
            self.step_counter += 1
            if self.step_counter >= self.stabilize_steps:
                self.state = GraspState.RETREAT
                self.step_counter = 0

        elif self.state == GraspState.RETREAT:
            target = np.array([goal[0], goal[1], self.safe_height])
            action = self._move_to(ee_pos, target, self.grip_open)
            if self._at_xyz(ee_pos, target):
                self.state = GraspState.RETURN_HOME
                self.step_counter = 0

        elif self.state == GraspState.RETURN_HOME:
            target = np.array([self.home_position[0], self.home_position[1], self.safe_height])
            if self._at_xyz_home(ee_pos, target):
                target = self.home_position.copy()
            action = self._move_to_gain(ee_pos, target, self.grip_open, self.home_gain)
            if self._at_xyz_home(ee_pos, self.home_position):
                self.state = GraspState.DONE

        else:
            action = np.array([0.0, 0.0, 0.0, self.grip_open], dtype=np.float32)

        return action

    def _extract(self, obs: dict) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        obs_vec = obs["observation"]
        ee_pos = obs_vec[0:3]
        finger_width = obs_vec[6]
        obj_pos = obs_vec[7:10]
        goal = obs["desired_goal"]
        return ee_pos, float(finger_width), obj_pos, goal

    def _at_xy(self, current: np.ndarray, target: np.ndarray) -> bool:
        return np.linalg.norm(current[:2] - target[:2]) < self.xy_tol

    def _at_xyz(self, current: np.ndarray, target: np.ndarray) -> bool:
        return self._at_xy(current, target) and abs(current[2] - target[2]) < self.z_tol

    def _move_to(self, current: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        return self._move_to_gain(current, target, gripper, self.pos_gain)

    def _move_to_gain(self, current: np.ndarray, target: np.ndarray, gripper: float, gain: float) -> np.ndarray:
        delta = target - current
        scaled = np.clip(delta * gain, -self.max_action, self.max_action)
        return np.concatenate([scaled, np.array([gripper], dtype=np.float32)], axis=0)

    def _at_xyz_home(self, current: np.ndarray, target: np.ndarray) -> bool:
        return (
            np.linalg.norm(current[:2] - target[:2]) < self.home_xy_tol
            and abs(current[2] - target[2]) < self.home_z_tol
        )


if __name__ == "__main__":
    env = make_env(render_mode="human", randomize_target=False)
    policy = ScriptedPickAndPlaceTablePolicy()

    for ep in range(3):
        obs, info = env.reset()
        policy.reset()
        print(f"\nEpisode {ep}: obj {obs['observation'][7:10]}, goal {obs['desired_goal']}, home {policy.home_position}")
        for step in range(220):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if policy.state == GraspState.DONE:
                print(f"  DONE at step {step}, success={info.get('is_success', False)}")
                break
            if terminated or truncated:
                print(f"  terminated early at step {step}")
                break

    env.close()
