from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env  # noqa: E402
from act_project.scripts.scripted_policy_table import ScriptedPickAndPlaceTablePolicy  # noqa: E402


def main() -> None:
    out_dir = ROOT / "reports" / "table" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(render_mode="rgb_array", randomize_target=False)
    policy = ScriptedPickAndPlaceTablePolicy()

    num_eps = 3
    max_steps = 220
    successes = 0

    for ep in range(num_eps):
        obs, info = env.reset()
        policy.reset()
        frames = []
        success = False
        success_seen = False
        for step in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success"):
                success_seen = True
            if policy.state == policy.state.DONE:
                success = success_seen
                successes += int(success_seen)
                break
            if terminated or truncated:
                break
        if frames:
            video_path = out_dir / f"table_ep_{ep:03d}.mp4"
            imageio.mimwrite(video_path, frames, fps=20, quality=8)
            print(f"ep {ep}: frames={len(frames)} success={success} -> {video_path}")
        else:
            print(f"ep {ep}: no frames captured")

    env.close()
    print(f"successes {successes}/{num_eps}")


if __name__ == "__main__":
    main()
