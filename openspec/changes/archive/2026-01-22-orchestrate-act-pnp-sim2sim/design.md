## Context
- 目标：在 `lerobot3`（CUDA 13.0，RTX 5060 Ti 16GB）内完成 Franka PickAndPlace 的 sim2sim ACT 流水线：环境基线 ≥90% 成功、录制 500 条成功 demo（50 目标 × 10 条）、ACT 训练与评估 ≥90% 成功，并产出可视化与可复现工件。
- 现状：`panda_mujoco_gym_ref` 提供 `FrankaPickAndPlaceSparse-v0`，success 判定基于 `info["is_success"]`（距离 < distance_threshold）。没有现成 LeRobot 数据或 ACT 训练脚本。
- 目录规划（新增）：`lerobot/act/` 下按 datasets（含 manifest）、training（config/ckpt/metrics）、reports（baseline/collection/train_eval/plots）分层。

## Goals / Non-Goals
- Goals: 阶段门控（每步 ≥0.90 成功率）；可重复的目标采样与 seeds；LeRobot ACT 兼容数据集与训练配置；可视化与报告；全流程在本机 GPU 上可运行。
- Non-Goals: 真实机器人执行、Push/Slide 多任务扩展、分布式训练或云管线。

## Decisions
- 阶段门控与样本量：基线评估 ≥20 episode；采集成功率计算包含重试；训练后评估 ≥50 episode；任一阶段 <0.90 则阻断后续。
- 目标采样：从环境 goal range 生成 50 个确定性目标（记录 seeds/坐标），每目标收集 10 条成功轨迹；失败重试直至满额。
- 数据采集策略：优先实现/复用确定性抓取-放置脚本（或已有高成功率 policy checkpoint）；统一入口负责 env reset、动作发布、成功判定与存储。
- 数据格式：采用 LeRobot ACT 兼容结构（轨迹级观测/动作/奖励/goal/成功标记），伴随 `dataset_manifest.json` 记录 episode_count=500、成功率、seeds、goal 索引、环境版本；存储于 `lerobot/act/datasets/pnp-sim2sim/`。
- 训练与评估：ACT 训练配置记录于 `lerobot/act/training/<run-id>/config.yaml`，产出 checkpoint、metrics（loss、eval success），评估脚本复用 env 并写入 `train_eval.json`。
- 可视化：报告包含目标覆盖散点/热力、训练曲线（loss、eval success），存放 `lerobot/act/reports/`。

## Risks / Trade-offs
- 脚本化策略若成功率不足需先行调试或加载已训练 policy，可能增加前置时间。
- 500 成功 demo 可能耗时较长；需确保重试逻辑和日志可断点续跑。
- ACT 训练可能受显存约束（16GB）；需要合理 batch/sequence 配置以避免 OOM。

## Migration Plan
1) 基线验证：env 导入 + 20+ episode 成功率记录；未达标先修复。
2) 目标采样与 manifest 固化；实现采集入口，循环至 50×10 成功，实时记录成功率。
3) 数据集验证（episode 计数、成功率、示例加载），再启动 ACT 训练。
4) 训练与评估；若 eval <0.90，调参/数据增补后重跑。
5) 生成可视化与最终报告，归档 config/seed/metrics。

## Open Questions
- 如无可用高成功率 policy，是否允许先行短训/微调以解锁 ≥0.90 的采集起点？
