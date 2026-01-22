# lerobot-act-sim2sim

基于 MuJoCo 的 Franka Panda Pick&Place sim-to-sim ACT 管线。此仓库包含：
- MuJoCo 机械臂建模文件：根目录 `scene.xml`，以及参考环境 `panda_mujoco_gym_ref/panda_mujoco_gym/assets/pick_and_place.xml` 等。
- Panda Pick&Place 环境与算法：`panda_mujoco_gym_ref/panda_mujoco_gym/envs/pick_and_place.py`（参考环境）、`act_project/envs/pick_place_table_env.py`（自定义封装）。
- ACT 训练/评估脚本：`act_project/scripts/train_act.py`、`debug_compare_actions.py`、`record_act_eval_videos.py`。
- 最新训练配置与报告：`lerobot/act/training/run-ce-grip-cuda-20260122-081800/config.yaml`，`lerobot/act/reports/{baseline.json,collection.json,train_eval.json,training_curves.png,collection_coverage.png}`。

## 快速开始（不含数据/模型）

> 数据集和模型未随仓库提供，请在 `lerobot/act/datasets/pnp-sim2sim/` 放置 HDF5 demos（500 条成功示例）后再运行。

一键脚本（训练 + 评估，可选录制视频）：
```bash
conda activate lerobot3
python run_pipeline.py --mode all \
  --dataset-dir lerobot/act/datasets/pnp-sim2sim \
  --epochs 100 --device cuda --eval-episodes 50 \
  --video-episodes 0
```
- 仅评估已有模型：`python run_pipeline.py --mode eval --ckpt <path_to_ckpt> --eval-episodes 50 --device cuda`
- 录制评估视频：增加 `--video-episodes 5 --video-output lerobot/act/reports/act_eval_videos`

## 目录速览
- `scene.xml`：MuJoCo 建模文件（Franka + 桌面场景）。
- `panda_mujoco_gym_ref/`：参考 Pick&Place 环境与资源（mesh/xml）。
- `act_project/`：训练/诊断脚本与环境封装。
- `lerobot/act/training/run-ce-grip-cuda-20260122-081800/`：最新 ACT 训练配置与指标（不含 ckpt，已在 .gitignore）。
- `lerobot/act/reports/`：基线、采集与训练评估报告/曲线。
- `openspec/`：需求与变更档案。

## 报告
- M1 基线（scripted）：`lerobot/act/reports/baseline.json`（100 ep，97%）。
- Demo 采集：`lerobot/act/reports/collection.json`（500/535，93.46%）。
- ACT 评估：`lerobot/act/reports/train_eval.json`（ckpt epoch_100，50 ep，98%，mean_steps 67.66）。

## 注意
- `.gitignore` 已过滤数据集、ckpt、视频、旧 run 输出，推送前无需手动清理大文件。
- 如需自定义运行参数，请直接编辑 `run_pipeline.py` 或传入对应 CLI 参数。
