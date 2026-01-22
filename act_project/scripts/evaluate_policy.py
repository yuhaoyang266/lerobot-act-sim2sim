from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

# Ensure local packages (panda_mujoco_gym_ref) are importable without installation.
ROOT = Path(__file__).resolve().parents[2]
PANDA_ROOT = ROOT / "panda_mujoco_gym_ref"
for candidate in (ROOT, PANDA_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

import panda_mujoco_gym  # noqa: F401
from gymnasium.wrappers import TimeLimit

from act_project.envs.pick_place_table_env import make_env
from act_project.scripts.scripted_policy_table import GraspState, ScriptedPickAndPlaceTablePolicy


def build_env(render_mode: Optional[str]):
    base = make_env(render_mode=render_mode, randomize_target=False)
    return TimeLimit(base, max_episode_steps=220)


def save_video(frames: List[np.ndarray], path: Path, fps: int = 20) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8)


def run_episode(
    env,
    policy: ScriptedPickAndPlaceTablePolicy,
    seed: int,
    max_steps: int,
    capture: bool,
    video_path: Optional[Path],
) -> Dict:
    obs, info = env.reset(seed=seed)
    policy.reset()
    frames: List[np.ndarray] = []

    success = False
    success_seen = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if capture:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if info.get("is_success"):
            success_seen = True
        if policy.state == GraspState.DONE:
            success = success_seen
            break
        if truncated:
            break

    if capture and frames and video_path is not None:
        save_video(frames, video_path)

    return {
        "success": bool(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": step + 1,
        "seed": seed,
    }


def evaluate(
    episodes: int,
    video_episodes: int,
    max_steps: int,
    seed: int,
    output_dir: Path,
) -> Dict:
    capture = video_episodes > 0
    render_mode = "rgb_array" if capture else None
    env = build_env(render_mode=render_mode)

    policy = ScriptedPickAndPlaceTablePolicy()
    rng = np.random.default_rng(seed)
    stats: List[Dict] = []

    video_dir = output_dir / "videos"

    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 1_000_000_000))
        video_path = None
        if capture and ep < video_episodes:
            video_path = video_dir / f"m1_scripted_ep_{ep:03d}.mp4"

        result = run_episode(
            env=env,
            policy=policy,
            seed=ep_seed,
            max_steps=max_steps,
            capture=capture,
            video_path=video_path,
        )
        stats.append(result)

    env.close()

    success_rate = float(np.mean([s["success"] for s in stats])) if stats else 0.0
    mean_steps = float(np.mean([s["steps"] for s in stats])) if stats else 0.0

    report = {
        "episodes": episodes,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "success_count": int(sum(s["success"] for s in stats)),
        "seeds": [s["seed"] for s in stats],
        "max_steps": max_steps,
        "video_episodes": video_episodes,
        "timestamp": time.time(),
    }

    report_path = output_dir / "m1_scripted_policy_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": report, "episodes": stats}, f, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scripted pick-and-place policy for M1 gate.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--video-episodes", type=int, default=5, help="Number of episodes to record as video.")
    parser.add_argument("--max-steps", type=int, default=150, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "act_project" / "reports" / "m1"),
        help="Directory to store reports and videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        episodes=args.episodes,
        video_episodes=args.video_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(report, indent=2))
    if report["success_rate"] < 0.9:
        print("WARNING: success_rate below 0.9 threshold; do not advance to M2.", file=sys.stderr)


if __name__ == "__main__":
    main()
