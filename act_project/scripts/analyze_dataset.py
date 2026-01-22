"""分析 M2 采集的 HDF5 数据集：轨迹长度分布、动作范围统计。"""
from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path
import json

DATASET_DIR = Path(__file__).resolve().parents[2] / "lerobot" / "act" / "datasets" / "pnp-sim2sim"


def main():
    episodes_dir = DATASET_DIR / "episodes"
    files = sorted(episodes_dir.glob("episode_*.hdf5"))

    if not files:
        print(f"No HDF5 files found in {episodes_dir}")
        return

    lengths = []
    all_actions = []

    for f in files:
        with h5py.File(f, "r") as h5:
            actions = h5["actions"][:]
            lengths.append(len(actions))
            all_actions.append(actions)

    all_actions = np.concatenate(all_actions, axis=0)

    print("=== Trajectory Length Statistics ===")
    print(f"Total trajectories: {len(lengths)}")
    print(f"Length range: [{min(lengths)}, {max(lengths)}]")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Median: {np.median(lengths):.1f}")
    print(f"Std dev: {np.std(lengths):.1f}")

    # Percentiles
    pcts = [10, 25, 50, 75, 90]
    for p in pcts:
        print(f"P{p}: {np.percentile(lengths, p):.0f}")

    print()
    print("=== Action Range Statistics ===")
    print(f"Total actions: {len(all_actions)}")
    for i, name in enumerate(["dx", "dy", "dz", "gripper"]):
        col = all_actions[:, i]
        print(f"{name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}")

    # Action distribution by phase (rough estimate based on gripper value)
    gripper = all_actions[:, 3]
    open_mask = gripper > 0.5
    close_mask = gripper < -0.5
    print()
    print("=== Action Phase Distribution ===")
    print(f"Gripper open (>0.5): {open_mask.sum()} ({100*open_mask.mean():.1f}%)")
    print(f"Gripper close (<-0.5): {close_mask.sum()} ({100*close_mask.mean():.1f}%)")
    print(f"Gripper neutral: {(~open_mask & ~close_mask).sum()} ({100*(~open_mask & ~close_mask).mean():.1f}%)")

    # Summary for ACT training
    print()
    print("=== ACT Training Recommendations ===")
    mean_len = np.mean(lengths)
    if mean_len > 100:
        print(f"WARNING: Mean trajectory length ({mean_len:.0f}) is high.")
        print("  Consider: chunk_size=50-100, or reduce stabilize_steps in scripted policy.")
    else:
        print(f"Mean trajectory length ({mean_len:.0f}) is reasonable for ACT.")

    # Check action saturation
    for i, name in enumerate(["dx", "dy", "dz"]):
        col = all_actions[:, i]
        sat_rate = ((np.abs(col) > 0.95).sum()) / len(col)
        if sat_rate > 0.1:
            print(f"WARNING: {name} saturates at ±1.0 in {100*sat_rate:.1f}% of actions.")


if __name__ == "__main__":
    main()
