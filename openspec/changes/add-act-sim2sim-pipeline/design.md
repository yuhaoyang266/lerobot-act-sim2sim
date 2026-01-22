# Design: ACT Sim-to-Sim Pipeline

## Context

### 背景
用户希望体验完整的 ACT (Action Chunking with Transformers) 模仿学习流程，从仿真环境验证到最终策略部署。目标是在 MuJoCo 中完成 Panda 机械臂的 Pick and Place 任务。

### 约束
- 每阶段必须达到 90% 成功率
- 只保存成功的示范轨迹
- 用户需要人工检查每个阶段的输出

### 硬件
- GPU: RTX 5060 Ti 16GB
- RAM: 32GB

## Goals / Non-Goals

### Goals
- ✅ 实现端到端的 ACT 训练流程
- ✅ 每阶段提供可视化检查点（视频 + 报告）
- ✅ 代码教程级别，详细注释

### Non-Goals
- ❌ 不涉及真实机器人部署（Sim-to-Real）
- ❌ 不优化到 SOTA 性能
- ❌ 不使用视觉输入（仅状态输入）

## Decisions

### D1: Scripted Policy 状态机设计

```
状态机流程:
┌─────────┐    ┌─────────┐    ┌─────────┐
│APPROACH │───▶│  GRASP  │───▶│  LIFT   │
│(接近物体)│    │ (抓取)  │    │ (提起)  │
└─────────┘    └─────────┘    └─────────┘
                                   │
     ┌─────────┐    ┌─────────┐    │
     │ RELEASE │◀───│  PLACE  │◀───┘
     │ (释放)  │    │ (放置)  │
     └─────────┘    └─────────┘
```

**决策**: 使用基于距离阈值的状态切换
- 接近阈值: 0.02m
- 抓取阈值: gripper_width < 0.06
- 提升高度: 0.15m above table
- 放置阈值: 0.02m from goal

### D2: 数据格式选择

**选项**:
1. LeRobot HDF5（推荐）
2. Zarr
3. 原始 numpy

**决策**: 使用 LeRobot HDF5 格式

```python
# HDF5 结构
data/
├── episode_0/
│   ├── observations/          # [T, obs_dim]
│   │   ├── ee_position        # [T, 3]
│   │   ├── ee_velocity        # [T, 3]
│   │   ├── fingers_width      # [T, 1]
│   │   ├── object_position    # [T, 3]
│   │   ├── object_rotation    # [T, 3]
│   │   ├── object_velp        # [T, 3]
│   │   └── object_velr        # [T, 3]
│   ├── actions/               # [T, 4] (dx, dy, dz, gripper)
│   ├── achieved_goal/         # [T, 3]
│   ├── desired_goal/          # [T, 3]
│   └── metadata/
│       ├── success            # bool
│       ├── length             # int
│       └── seed               # int
```

### D3: ACT 网络架构

```
输入: state [B, T, 20]
      ↓
State Encoder: Linear(20 → 256) + ReLU
      ↓
Positional Encoding (sinusoidal)
      ↓
Transformer Encoder (4 layers, 4 heads)
      ↓
Action Decoder: Linear(256 → 4)
      ↓
输出: actions [B, T, 4]
```

**超参数**:
- hidden_dim: 256
- num_layers: 4
- num_heads: 4
- chunk_size: 50 (与 max_episode_steps 一致)
- learning_rate: 1e-4
- batch_size: 32 (16GB 显存足够)

### D4: 训练策略

- Optimizer: AdamW
- Loss: MSE (action 级别)
- Epochs: 100-200
- 早停: patience=10

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Scripted Policy 在边缘情况失败 | Demo 质量下降 | 增加边界检查，只保存成功轨迹 |
| ACT 过拟合 | 泛化性差 | 数据增强，dropout，早停 |
| 训练不收敛 | 无法完成 M3 | 调整 lr，检查数据归一化 |

## Migration Plan

N/A - 新项目，无需迁移

## Open Questions

1. 是否需要添加视觉输入（camera images）？
   - 当前计划：仅状态输入，简化实现
   - 未来可扩展

2. chunk_size 设为多少？
   - 暂定 50（与 max_episode_steps 一致）
   - 可根据任务复杂度调整
