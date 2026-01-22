# pnp-act-sim2sim Specification

## Purpose
TBD - created by archiving change orchestrate-act-pnp-sim2sim. Update Purpose after archive.
## Requirements
### Requirement: lerobot3 runtime gate for Franka pick-and-place
The system SHALL verify that the `lerobot3` conda environment can import torch/mujoco/panda_mujoco_gym, run `FrankaPickAndPlaceSparse-v0` with GPU enabled, and achieve baseline success ≥0.90 before any demo collection or training starts.

#### Scenario: Baseline readiness check
- **WHEN** the baseline script is executed in `lerobot3`, running ≥20 evaluation episodes with the planned data-collection policy
- **THEN** torch reports CUDA availability (RTX 5060 Ti 16GB) and MuJoCo renders without error
- **AND** the mean success_rate (count of `info["is_success"]` / episodes) is ≥0.90 and logged to `lerobot/act/reports/baseline.json` with seeds and episode count
- **AND** if success_rate <0.90 the pipeline SHALL block subsequent stages until corrected

### Requirement: Structured demonstration collection for ACT
The system SHALL collect 500 successful Franka pick-and-place trajectories (50 distinct goal targets × 10 demos each) in MuJoCo simulation and store them as a LeRobot ACT-compatible dataset under `lerobot/act/datasets/pnp-sim2sim/`.

#### Scenario: Goal coverage and metadata
- **WHEN** the goal sampler enumerates 50 deterministic target positions within the environment goal ranges using recorded seeds
- **THEN** each target accumulates exactly 10 successful trajectories tagged with goal_id, seed, and success flag
- **AND** `dataset_manifest.json` records environment id `FrankaPickAndPlaceSparse-v0`, episode_count=500, goal coordinates, seeds, success_rate, and file layout

#### Scenario: Success-only retention
- **WHEN** an episode finishes
- **THEN** it is persisted only if `info["is_success"] == 1`
- **AND** the aggregate success rate across attempts is ≥0.90 before the dataset is declared complete; otherwise collection SHALL continue or be flagged as failed

### Requirement: ACT training and evaluation gate
The system SHALL train an ACT policy on the collected dataset and only accept the run when evaluation success is ≥0.90.

#### Scenario: Training artifacts
- **WHEN** launching ACT training from `lerobot3` with a config that references the dataset and random seeds
- **THEN** `config.yaml`, checkpoints, and metrics (loss, eval success per evaluation window) are saved under `lerobot/act/training/<run-id>/`, capturing dataset hash/size and code revision

#### Scenario: Post-training evaluation
- **WHEN** running evaluation for ≥50 episodes in `FrankaPickAndPlaceSparse-v0`
- **THEN** the mean success_rate is ≥0.90 with a reported confidence interval, and results are written to `lerobot/act/reports/train_eval.json`; runs below 0.90 SHALL be marked failed and not promoted

### Requirement: Reporting and visualization for sim2sim pipeline
The system SHALL produce human-readable reports and plots covering baseline, data collection, and training progress.

#### Scenario: Dataset coverage report
- **WHEN** the dataset is finalized
- **THEN** coverage plots (goal scatter/heatmap, per-goal success histogram) and summaries are emitted to `lerobot/act/reports/plots/collection_*` along with a JSON summary linked from `lerobot/act/reports/collection.json`

#### Scenario: Training progress plots
- **WHEN** ACT training completes
- **THEN** plots for loss and evaluation success over time/steps are saved to `lerobot/act/reports/plots/train_*` and cross-referenced in `lerobot/act/reports/train_eval.json`

