# 2026-01-22 M3 突破：0% → 98% 成功率

## 核心成果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 成功率 | 0% | **98%** |
| 平均步数 | 超时 (220) | 67.66 |
| TRANSPORT 方向 | ❌ 相反 | ✅ 正确 |
| LOWER 方向 | ❌ 相反 | ✅ 正确 |
| Gripper 后半段 | ❌ +1 张开 | ✅ -1 闭合 |

**最佳 Checkpoint**: `lerobot/act/training/run-ce-grip-cuda-20260122-081800/epoch_100.pt`

---

## 根因分析与修复

### 问题本质

昨日诊断发现：TRANSPORT/LOWER 阶段模型输出 **方向完全相反**。

| 阶段 | 期望 | 实际 (修复前) |
|------|------|---------------|
| TRANSPORT dz | ≈0 (平移) | +0.32~0.47 (上升) |
| LOWER dz | -0.78 (下降) | +0.31 (上升) |
| LOWER gripper | -1 (闭合) | +1 (张开) |

### 三大假设验证

| 假设 | 描述 | 验证结果 |
|------|------|---------|
| A | LOWER 样本不足 | 未验证 (B 已解决问题) |
| **B** | **Gripper 双峰分布不适合回归** | ✅ **确认为根因** |
| C | 分布偏移 (规则/模型切换) | 未验证 |

### 关键修复

**修复 1: Gripper 二分类**
```
之前: 回归 + sigmoid + 阈值 → 双峰分布下产生系统性偏移
现在: CrossEntropy + argmax → 直接输出 {-1, +1}
```

**修复 2: 移除 Action 归一化**
```
之前: obs 和 action 都做 z-score 标准化
现在: 仅 obs 归一化，action 保持原始 [-1, 1]
```

**修复 3: 模型架构拆分**
```python
# 之前: 单一 action_head
self.action_head = nn.Linear(dim, 4)

# 现在: 双 head 分离
self.pos_head = nn.Linear(dim, 3)      # 连续回归
self.gripper_head = nn.Linear(dim, 2)  # 二分类
```

---

## 验证过程

### Step 1: 最小重训 (1 epoch, CPU)
- 目的: 验证新架构能运行
- 结果: DESCEND/CLOSE 阶段 dz 为负，方向初步修正

### Step 2: 正式训练 (100 epoch, CUDA)
- 命令: `python train_act.py --epochs 100 --device cuda`
- 运行时间: ~15 分钟

### Step 3: 对比验证
```bash
python debug_compare_actions.py --ckpt epoch_100.pt --device cuda --steps 80
```

| 阶段 | 偏差 (mean) |
|------|-------------|
| TRANSPORT | 0.0477 |
| LOWER | 0.2347 |
| 整体 | 0.1039 |

### Step 4: 完整评估
```bash
# 50 episodes, max_steps=220
python -c "from act_project.scripts.train_act import evaluate_policy; ..."
```

结果: **98% 成功率, 平均 67.66 步**

---

## 遗留技术债务

| 债务 | 影响 | 优先级 |
|------|------|--------|
| LOWER 幅度偏浅 (-0.40 vs -0.78) | 阈值收紧会失败 | 低 |
| 混合策略仍存在 | 端到端纯度不够 | 低 |
| Padding 策略 | 影响 episode 初期 | 低 |

---

## 关键文件变更

| 文件 | 修改内容 |
|------|---------|
| `train_act.py` | 双 head 架构, CE loss, obs-only 归一化 |
| `debug_compare_actions.py` | 适配新 head, 移除反标准化 |
| `EpisodeDataset` | 移除 act_mean/act_std |

---

## 经验总结

1. **Gripper 建模**: 离散动作 {open, close} 用分类，不要用回归
2. **归一化一致性**: 训练和推理必须使用相同的归一化策略
3. **诊断先行**: `debug_compare_actions.py` 的分阶段对比是定位问题的关键

---

## M1-M4 最终状态

| 阶段 | 目标 | 实际 | 状态 |
|------|------|------|------|
| M1 环境验证 | ≥90% | 95%+ | ✅ |
| M2 Demo 采集 | 500 条 | 500 条 | ✅ |
| M3 ACT 训练 | Loss 收敛 | 收敛 | ✅ |
| M4 策略评估 | ≥90% | **98%** | ✅ |
