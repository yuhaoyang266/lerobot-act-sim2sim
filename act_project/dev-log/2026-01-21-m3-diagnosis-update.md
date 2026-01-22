# 2026-01-21 M3 诊断更新 - 后半段行为方向错误

## 今日关键发现

### 诊断结果汇总

| 阶段 | 控制方式 | 期望行为 | 模型实际行为 | 结论 |
|------|----------|----------|--------------|------|
| APPROACH | 规则 | - | - | ✅ 规则工作 |
| DESCEND | 规则 | - | - | ✅ 规则工作 |
| CLOSE | 规则 | - | - | ✅ 规则工作 |
| LIFT | 规则 | - | - | ✅ 规则工作 |
| TRANSPORT dz | 模型 | ≈0 (平移) | +0.32~0.47 (上升) | ❌ 方向错误 |
| LOWER dz | 模型 | -0.78 (下降) | +0.31 (上升) | ❌ 方向错误 |
| LOWER gripper | 模型 | -1 (闭合) | +1 (张开) | ❌ 完全相反 |

### 核心洞察

- **问题不是幅度，而是方向完全错误**
- 移除 action 归一化后问题更严重，不是归一化的锅
- TRANSPORT 样本占比 39%，数据量充足，不是样本量问题
- 模型在后半段（TRANSPORT/LOWER）根本没学会正确行为

---

## 已尝试的修复

| 尝试 | 结果 |
|------|------|
| ❌ 移除 action 归一化 | 无效，方向错误反而更严重 |

---

## 下一步计划 (Step 1-3)

### Step 1: 规则延伸 baseline（调试工具）

**目的**：
- 验证环境和数据流没有其他 bug
- 得到一个可对比的 baseline 数字
- 为后续调试提供参照

**修改方案**：`InferencePolicy.act()` 延伸规则覆盖：
- TRANSPORT: 规则控制 XY 平移到 goal 上方，gripper=-1
- LOWER: 规则控制 dz 下降，gripper=-1
- RELEASE: 规则控制 gripper=+1 打开

**心态**：这是调试工具，不是最终方案。

### Step 2: 根因调查（三个假设）

#### 假设 A: LOWER 阶段样本严重不足

**验证方法**：
```python
# 统计 demo 中各阶段样本占比
# 如果 LOWER < 5%，模型几乎没见过这个阶段
```

**修复方向**：
- 采集更多 demo，或对 LOWER 阶段过采样
- 在 DataLoader 中加权采样（weight by phase）

#### 假设 B: Gripper 不该用回归

**验证方法**：
- 看 demo 中 gripper 的分布
- 如果是双峰（-1 和 +1），回归 + sigmoid 不适合

**修复方向**：
- Gripper 单独用分类 head（CrossEntropy Loss）
- 或者用确定性阈值（不是 sigmoid 概率）

#### 假设 C: 序列预测的分布偏移

**验证方法**：
- 对比训练时和推理时的状态分布
- 训练时：状态来自 scripted policy 轨迹
- 推理时：前半段由规则控制，状态可能不同

**修复方向**：
- DAgger：用模型跑轨迹，标注 scripted action，加入训练
- 或者：完全移除规则，让模型从头到尾端到端

### Step 3: 选择修复路径

根据 Step 2 的验证结果选择：

```
验证结果
│
├── LOWER 样本 < 5%
│   → 过采样 LOWER 阶段，重新训练
│
├── Gripper 分布双峰
│   → 改为分类 head，重新训练
│
├── 分布偏移严重
│   → 移除全部规则 + DAgger 迭代
│
└── 多个因素叠加
    → 逐一修复，每次验证
```

---

## 成功标准

| 里程碑 | 目标 | 状态 |
|--------|------|------|
| 诊断完成 | 定位瓶颈 | ✅ 后半段方向错误 |
| 规则 baseline | 验证环境正确 | ⏳ Step 1 |
| 根因确认 | A/B/C 假设验证 | ⏳ Step 2 |
| 模型修复 | 后半段行为正确 | ⏳ Step 3 |
| 端到端成功率 | ≥ 10% | ⏳ |
| M3 Gate | ≥ 90% | ⏳ 最终目标 |

---

## 架构师分析

### 1. 核心问题定位

诊断正确识别了 **distribution shift (分布偏移)** 这一 Behavior Cloning 的经典问题：
- 训练时：所有状态都在 scripted policy 轨迹上
- 推理时：前半段由规则控制，到 TRANSPORT 时状态分布可能已偏移

### 2. 假设 B 最值得优先验证

```
Gripper 回归 + sigmoid 的问题：

Demo 中:  gripper ∈ {-1, +1}  (离散)
模型输出: gripper ∈ ℝ       (连续)

如果训练数据 73% 是 -1，27% 是 +1
BCE loss 会让模型偏向预测 "闭合"
但 sigmoid 阈值 0.5 可能造成边界附近震荡
```

**建议**：直接看 demo 中 gripper 的分布，如果确实是双峰，改用 CrossEntropy 分类 head（2 类：open/close）会更稳健。

### 3. 数据管道复查建议

`EpisodeDataset.__getitem__` 中的 50% 起点采样逻辑可能导致 **LOWER 阶段在训练集中占比偏低**（因为 LOWER 在轨迹末尾，随机采样很难命中）。

---

## 关键文件

- `act_project/scripts/train_act.py:332-461` - InferencePolicy 混合策略
- `act_project/scripts/debug_compare_actions.py` - 诊断对比脚本
- `lerobot/act/datasets/pnp-sim2sim/episodes/` - Demo 数据

---

## 重要认知

- 当前规则延伸只是调试工具，不是"ACT 学会了任务"
- 真正的成功是模型端到端完成任务（≥90% 成功率）
- 问题根源需要 Step 2 验证后才能确定
