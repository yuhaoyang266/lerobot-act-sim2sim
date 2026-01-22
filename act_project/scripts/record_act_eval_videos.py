from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env
from act_project.scripts.train_act import ACTModel, Config, InferencePolicy


def build_env(max_steps: int, capture: bool):
    render_mode = "rgb_array" if capture else None
    env = make_env(render_mode=render_mode, randomize_target=False)
    return TimeLimit(env, max_episode_steps=max_steps)


def save_video(frames: List[np.ndarray], path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, quality=8)


def evaluate_with_video(
    ckpt_path: Path,
    episodes: int,
    video_episodes: int,
    max_steps: int,
    seed: int,
    output_dir: Path,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    state_dict = ckpt["model"]
    obs_dim = state_dict["obs_proj.weight"].shape[1]
    pos_dim = state_dict["head_pos.weight"].shape[0]
    act_dim = pos_dim + 1

    cfg_dict = ckpt["cfg"]
    cfg_dict["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    cfg.eval_episodes = episodes
    cfg.max_steps = max_steps

    stats = ckpt.get("stats", None)
    model = ACTModel(obs_dim, act_dim, cfg).to(cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    policy = InferencePolicy(model, cfg, stats)

    capture = video_episodes > 0
    env = build_env(cfg.max_steps, capture)
    rng = np.random.default_rng(seed)
    results: List[Dict] = []

    for ep in range(episodes):
        ep_seed = int(rng.integers(0, 1_000_000_000))
        obs, info = env.reset(seed=ep_seed)
        policy.reset()
        frames: List[np.ndarray] = []
        success = False
        steps = 0

        for step in range(cfg.max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps = step + 1
            if capture and ep < video_episodes:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            if info.get("is_success"):
                success = True
                break
            if truncated:
                break

        if frames and ep < video_episodes:
            video_path = output_dir / f"act_eval_ep_{ep:03d}.mp4"
            save_video(frames, video_path)

        results.append({"success": success, "steps": steps, "seed": ep_seed})

    env.close()

    success_rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    mean_steps = float(np.mean([r["steps"] for r in results])) if results else 0.0
    summary = {
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "episodes": episodes,
        "video_episodes": video_episodes,
        "max_steps": max_steps,
        "seeds": [r["seed"] for r in results],
        "ckpt": str(ckpt_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "act_eval_report.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACT policy and record videos.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to ACT checkpoint.")
    parser.add_argument("--episodes", type=int, default=10, help="Total episodes to run.")
    parser.add_argument("--video-episodes", type=int, default=5, help="Number of episodes to record.")
    parser.add_argument("--max-steps", type=int, default=220, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("lerobot/act/reports/act_eval_videos")),
        help="Directory to store videos and report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_with_video(
        ckpt_path=Path(args.ckpt),
        episodes=args.episodes,
        video_episodes=args.video_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
