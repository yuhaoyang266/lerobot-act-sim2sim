from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env  # noqa: E402
from act_project.scripts.scripted_policy_table import ScriptedPickAndPlaceTablePolicy  # noqa: E402


DATASET_DIR = ROOT.parent / "lerobot" / "act" / "datasets" / "pnp-sim2sim"
EPISODES_DIR = DATASET_DIR / "episodes"
VIDEOS_DIR = DATASET_DIR / "videos"
GOAL_MANIFEST = DATASET_DIR / "goal_manifest.json"
DATASET_MANIFEST = DATASET_DIR / "dataset_manifest.json"
COLLECTION_REPORT = DATASET_DIR / "collection.json"

GOAL_COUNT = 50
DEMOS_PER_GOAL = 10
MAX_ATTEMPTS_PER_GOAL = 30
JITTER_RANGE = 0.04
VIDEO_SAMPLES = 10  # record first attempt of these goals
RNG_SEED = 123


def ensure_dirs() -> None:
    for d in (DATASET_DIR, EPISODES_DIR, VIDEOS_DIR):
        os.makedirs(d, exist_ok=True)


def sample_goals(center: np.ndarray, rng: np.random.Generator) -> List[Dict]:
    goals = []
    for gid in range(GOAL_COUNT):
        offset = rng.uniform(-JITTER_RANGE, JITTER_RANGE, size=2)
        pos = center.copy()
        pos[0] += offset[0]
        pos[1] += offset[1]
        goals.append({"goal_id": gid, "position": pos.tolist()})
    return goals


def save_goal_manifest(center: np.ndarray, goals: List[Dict]) -> None:
    manifest = {
        "table_center": center.tolist(),
        "jitter_range": JITTER_RANGE,
        "goal_count": GOAL_COUNT,
        "goals": goals,
    }
    with GOAL_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_episode_hdf5(path: Path, observations: np.ndarray, actions: np.ndarray, timestamps: np.ndarray, attrs: Dict) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("observations", data=observations)
        h5.create_dataset("actions", data=actions)
        h5.create_dataset("timestamps", data=timestamps)
        for k, v in attrs.items():
            h5.attrs[k] = v


def collect() -> None:
    ensure_dirs()
    env = make_env(render_mode="rgb_array", randomize_target=False)
    policy = ScriptedPickAndPlaceTablePolicy()
    rng = np.random.default_rng(RNG_SEED)

    _obs0, _info0 = env.reset()
    center = env.table_center if hasattr(env, "table_center") else _obs0["desired_goal"]

    goals = sample_goals(center, rng)
    save_goal_manifest(center, goals)

    video_goal_ids = set(np.linspace(0, GOAL_COUNT - 1, VIDEO_SAMPLES, dtype=int).tolist())

    stats_per_goal = []
    total_success = 0
    total_attempts = 0
    episode_idx = 0

    for goal in goals:
        gid = goal["goal_id"]
        goal_pos = np.array(goal["position"])
        goal_success = 0
        attempts = 0
        retries = 0
        while goal_success < DEMOS_PER_GOAL and attempts < MAX_ATTEMPTS_PER_GOAL:
            attempts += 1
            seed = int(rng.integers(0, 1_000_000_000))
            obs, info = env.reset(seed=seed)
            env.unwrapped.goal = goal_pos
            obs["desired_goal"] = goal_pos
            policy.reset()

            frames = []
            obs_list = []
            act_list = []
            timestamps = []
            success = False

            for step in range(env.spec.max_episode_steps if env.spec and env.spec.max_episode_steps else 220):
                obs_with_goal = np.concatenate([obs["observation"], obs["desired_goal"]])
                obs_list.append(obs_with_goal.astype(np.float32, copy=False))
                action = policy.act(obs)
                act_list.append(action.copy())
                timestamps.append(step)

                obs, reward, terminated, truncated, info = env.step(action)

                if gid in video_goal_ids and attempts == 1:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                if info.get("is_success"):
                    success = True
                    break
                if truncated:
                    break

            if success:
                goal_success += 1
                total_success += 1
                ep_path = EPISODES_DIR / f"episode_{episode_idx:06d}.hdf5"
                observations = np.array(obs_list, dtype=np.float32)
                actions = np.array(act_list, dtype=np.float32)
                timestamps_arr = np.array(timestamps, dtype=np.float32)
                attrs = {
                    "goal_id": gid,
                    "seed": seed,
                    "success": True,
                    "steps": len(actions),
                    "goal_position": goal_pos.tolist(),
                }
                write_episode_hdf5(ep_path, observations, actions, timestamps_arr, attrs)
                episode_idx += 1

                if frames and (goal_success == 1) and (gid in video_goal_ids):
                    import imageio.v2 as imageio

                    video_path = VIDEOS_DIR / f"goal_{gid:02d}_attempt_{attempts:02d}.mp4"
                    imageio.mimwrite(video_path, frames, fps=20, quality=8)
            else:
                retries += 1

            if attempts >= MAX_ATTEMPTS_PER_GOAL and goal_success < DEMOS_PER_GOAL:
                break

        total_attempts += attempts
        stats_per_goal.append(
            {
                "goal_id": gid,
                "goal_position": goal_pos.tolist(),
                "success_count": goal_success,
                "attempt_count": attempts,
                "retry_count": retries,
                "success_rate": goal_success / attempts if attempts else 0.0,
            }
        )

    env.close()

    dataset_manifest = {
        "total_episodes": episode_idx,
        "goals": GOAL_COUNT,
        "demos_per_goal": DEMOS_PER_GOAL,
        "success_count": total_success,
        "attempt_count": total_attempts,
        "success_rate": total_success / total_attempts if total_attempts else 0.0,
        "goal_manifest": str(GOAL_MANIFEST),
    }
    with DATASET_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    with COLLECTION_REPORT.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": dataset_manifest,
                "per_goal": stats_per_goal,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    collect()
