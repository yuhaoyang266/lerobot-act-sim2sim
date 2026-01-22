from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_ROOT = ROOT.parent / "panda_mujoco_gym_ref"
if str(PANDA_ROOT) not in sys.path:
    sys.path.insert(0, str(PANDA_ROOT))
import panda_mujoco_gym  # noqa: F401

DEFAULT_DATASET_ROOT = ROOT.parent / "lerobot" / "act" / "datasets" / "pnp-sim2sim"


def inspect_env(env_id: str) -> None:
    np.set_printoptions(precision=4, suppress=True)
    env = gym.make(env_id)
    obs, _ = env.reset()
    print("=" * 50)
    print("Environment observation diagnostics")
    print("=" * 50)
    print(f"keys: {list(obs.keys())}")
    obs_arr = obs["observation"]
    print(f"observation shape: {obs_arr.shape}")
    print(f"observation[:10]: {obs_arr[:10]}")
    if len(obs_arr) >= 10:
        obj_pos = obs_arr[7:10]
        print(f"obs[7:10] (obj_pos): {obj_pos}")
    if "achieved_goal" in obs:
        print(f"achieved_goal (object pos): {obs['achieved_goal']}")
    if "desired_goal" in obs:
        print(f"desired_goal (goal pos): {obs['desired_goal']}")
    if len(obs_arr) >= 10 and "desired_goal" in obs:
        delta = np.linalg.norm(obs["desired_goal"] - obs_arr[7:10])
        print(f"|desired_goal - obj_pos|: {delta:.4f}")
    env.close()


def _pick_episode(dataset_path: Path, episode_index: int) -> tuple[Path, int]:
    if dataset_path.is_file():
        return dataset_path, 1
    episodes_dir = dataset_path / "episodes"
    files = sorted(episodes_dir.glob("episode_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no episodes found under {episodes_dir}")
    idx = min(max(episode_index, 0), len(files) - 1)
    return files[idx], len(files)


def inspect_demo(dataset_root: Path, episode_index: int) -> None:
    np.set_printoptions(precision=4, suppress=True)
    ep_path, total = _pick_episode(dataset_root, episode_index)
    print("\n" + "=" * 50)
    print("Demo dataset diagnostics")
    print("=" * 50)
    print(f"Using episode file: {ep_path.name} (index {episode_index}, total {total})")

    with h5py.File(ep_path, "r") as f:
        obs = f["observations"][:]
        act = f["actions"][:]
        print(f"observations shape: {obs.shape}")
        print(f"actions shape: {act.shape}")
        if obs.shape[0] != act.shape[0]:
            print("WARNING: obs/action length mismatch")
        if obs.shape[0] == 0:
            print("WARNING: empty observations")
            return
        first = obs[0]
        last = obs[-1]
        if obs.shape[1] >= 10:
            obj0 = first[7:10]
            obj1 = last[7:10]
            diff = np.linalg.norm(obj1 - obj0)
            print(f"obs[7:10] first: {obj0}")
            print(f"obs[7:10] last:  {obj1}")
            print(f"object position delta: {diff:.4f}")
            if diff < 0.01:
                print("WARNING: object position hardly changes; may be missing")
            else:
                print("OK: object position varies")
        if obs.shape[1] >= 3:
            goal0 = first[-3:]
            print(f"obs[-3:] (likely desired_goal): {goal0}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether observations/demos include object position.")
    parser.add_argument("--env-id", default="FrankaPickAndPlaceDense-v0")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to dataset root containing episodes/ or a single .hdf5 file",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    inspect_env(args.env_id)
    inspect_demo(args.dataset_root, args.episode_index)


if __name__ == "__main__":
    main()
