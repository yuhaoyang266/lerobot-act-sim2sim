from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env  # noqa: E402
from act_project.scripts.scripted_policy_table import ScriptedPickAndPlaceTablePolicy  # noqa: E402


def main() -> None:
    env = make_env(render_mode="human", randomize_target=False)
    policy = ScriptedPickAndPlaceTablePolicy()

    successes = 0
    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        policy.reset()
        print(f"\nEpisode {ep}: obj {obs['observation'][7:10]}, goal {obs['desired_goal']}")
        for step in range(160):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success"):
                print(f"  success at step {step}")
                successes += 1
                break
            if terminated or truncated:
                print(f"  terminated at step {step}")
                break
    env.close()
    print(f"\nSuccess rate: {successes}/{episodes} = {successes/episodes:.2f}")


if __name__ == "__main__":
    main()
