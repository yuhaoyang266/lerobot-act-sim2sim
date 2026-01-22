# Change: Sim2Sim ACT pipeline for Franka pick-and-place

## Why
We need a reproducible sim2sim pipeline that (1) validates the panda MuJoCo pick-and-place environment in the `lerobot3` conda runtime, (2) records 500 successful demonstrations across 50 goal positions for ACT, and (3) trains and evaluates an ACT policy with ≥90% success before advancing each stage, with artifacts and visualizations saved for inspection.

## What Changes
- Add a new capability spec (`pnp-act-sim2sim`) that formalizes stage gates, demo coverage (50 goals × 10 demos), LeRobot ACT-compatible dataset structure, and training/evaluation outputs.
- Document environment/runtime expectations for `lerobot3` (CUDA 13.0, RTX 5060 Ti 16GB) and success gating ≥0.90 before data collection or training proceeds.
- Plan data collection, training, and visualization directories under `lerobot/act/`, including manifesting seeds/goals, checkpoints, metrics, and plots.

## Impact
- Affected specs: pnp-act-sim2sim (new capability)
- Affected code/assets: panda_mujoco_gym_ref (env use), new lerobot/act data-collection + training scripts/configs, evaluation/visualization tooling and reports
