# Change: Add ACT Sim-to-Sim Training Pipeline

## Why
需要构建一个完整的 ACT (Action Chunking with Transformers) 模仿学习流程，从环境验证到数据采集、模型训练、策略评估。每个阶段都需要达到 90% 成功率才能进入下一阶段。

## What Changes
- **新增** Scripted Policy：基于状态的规则策略，用于采集高质量示范数据
- **新增** Demo Collector：轨迹录制脚本，保存为 LeRobot HDF5 格式
- **新增** ACT Trainer：基于 Transformer 的动作分块策略训练器
- **新增** Policy Evaluator：策略评估脚本，生成视频和统计报告
- **新增** 配置系统：统一管理超参数

## Impact
- Affected specs: `act-training` (新建)
- Affected code: 
  - `act_project/scripts/scripted_policy.py`
  - `act_project/scripts/collect_demos.py`
  - `act_project/scripts/train_act.py`
  - `act_project/scripts/evaluate_policy.py`

## Milestones

| ID | 名称 | 验收标准 | 依赖 |
|----|------|----------|------|
| M1 | 环境验证 | Scripted Policy ≥ 90% (100 eps) | - |
| M2 | Demo 采集 | 500 条成功轨迹 | M1 |
| M3 | ACT 训练 | Loss 收敛 | M2 |
| M4 | 策略评估 | ACT ≥ 90% (50 eps) | M3 |

## Quality Gates
- 每阶段必须达标才能进入下一阶段
- 用户需要人工检查视频和报告确认
