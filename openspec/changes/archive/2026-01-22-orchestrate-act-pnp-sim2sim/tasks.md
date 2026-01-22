## 1. 准备
- [x] 1.1 在 `project.md` 补充 `lerobot3` 运行时信息（CUDA 13.0、RTX 5060 Ti 16GB、MuJoCo/torch/gym 版本基线），并确认 `FrankaPickAndPlaceSparse-v0` 可加载。
- [x] 1.2 设计并记录基线评估入口（激活 `lerobot3`，导入 torch/mujoco/panda_mujoco_gym，20+ episode 成功率计算与日志落地到 `lerobot/act/reports/baseline.json`），未达 0.90 阈值需调整前行。

## 2. Demo 采集
- [x] 2.1 定义 50 个可重复的目标位姿采样方案（使用环境 goal range，持久化 seeds/目标坐标），并规划 10 条成功 demo/目标。
- [x] 2.2 实现采集与存储管线：驱动 `FrankaPickAndPlaceSparse-v0`，只保留 `info["is_success"]==1` 的轨迹，生成 LeRobot ACT 兼容数据集与 `dataset_manifest.json`，目录 `lerobot/act/datasets/pnp-sim2sim/`。
- [x] 2.3 运行采集至成功 demo 数量达 500 且尝试成功率 ≥0.90；在 `lerobot/act/reports/collection.json` 记录成功率、重试次数、失败样本摘要。

## 3. 训练与评估
- [x] 3.1 编写 ACT 训练配置（数据路径、超参、随机种子、评估频率），落地到 `lerobot/act/training/<run-id>/config.yaml`。
- [x] 3.2 运行 ACT 训练并产出 checkpoint、metrics 日志；使用独立评估脚本在 MuJoCo 中至少 50 episode 验证，成功率 ≥0.90 前不可宣告完成，结果写入 `lerobot/act/reports/train_eval.json`。

## 4. 可视化与验证
- [x] 4.1 生成数据覆盖图（目标分布与成功计数）与训练曲线（loss、eval success），保存到 `lerobot/act/reports/`。
- [x] 4.2 执行 `openspec validate orchestrate-act-pnp-sim2sim --strict` 并修复所有校验问题。
