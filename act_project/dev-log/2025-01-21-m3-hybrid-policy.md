# M3 模型调试日志 - 2025-01-21

## 问题背景

- **现象**: 训练 Loss 收敛（0.203 → 0.0016），但评估成功率 0%
- **所有 episode 跑满 220 步**，物体从未到达目标

---

## 今日诊断与修复

### 1. Gripper 双峰分布问题

**发现**: Gripper 是双峰分布（27% 张开 +1，73% 闭合 -1），MSE 学到中间值

**修复**:
- 将 gripper 从 MSE 改为 **BCE Loss + Sigmoid**
- 训练时 gripper 目标转换：`[-1,1] → [0,1]`
- 推理时用 sigmoid 阈值化：`prob > 0.5 → +1, else → -1`

**结果**: Gripper 不再输出中间值，但仍有阶段性错误

### 2. APPROACH 阶段数据采样不足

**发现**: 随机 chunk 采样导致轨迹开始阶段（APPROACH）数据稀疏

**修复**:
- 50% 概率从轨迹起点采样
- 50% 概率随机采样

```python
if np.random.random() < 0.5:
    start = 0  # 从轨迹开始
else:
    start = np.random.randint(0, len(obs) - self.chunk_size)
```

### 3. 混合策略实现

**发现**: 模型在 APPROACH/DESCEND 阶段输出几乎为零的位置动作

**修复**: 实现规则+模型混合策略

```python
# InferencePolicy.act() 中的混合控制
if ee_obj_xy_dist > 0.02:
    # APPROACH: 规则控制移动到物体上方
    action[3] = 1.0  # 张开夹爪
elif ee_pos[2] > obj_pos[2] + 0.01:
    # DESCEND: 规则控制下降
    action[3] = 1.0  # 张开夹爪
else:
    # MODEL: 交给模型控制 CLOSE/LIFT/TRANSPORT
```

### 4. GRASP_HEIGHT 阈值调整

**发现**: 测试显示 EE 在 z=0.033 时切换到模型，但物体在 z=0.020，夹爪抓不到

**修复**:
- 原: `ee_pos[2] > GRASP_HEIGHT + 0.01` (固定 0.035)
- 改: `ee_pos[2] > obj_pos[2] + 0.01` (动态跟踪物体高度)

---

## 训练运行记录

| Run ID | 配置 | Epoch | Final Loss | 成功率 |
|--------|------|-------|------------|--------|
| run-m3-11-bce | BCE for gripper | 50 | 0.903 | 0% |
| run-m3-12-bce-nonorm | BCE + 无归一化 | 100 | 0.020 | 0% |
| run-m3-13-bce-balanced | BCE + pos_weight | 100 | 0.032 | 0% |
| run-m3-14-approach-fix | BCE + 50%起点采样 | 100 | 0.049 | 0% (待测混合策略) |

---

## 修改文件

1. **`act_project/scripts/train_act.py`**
   - `Config`: 添加 `normalize`, `gripper_weight` 参数
   - `EpisodeDataset`: 50% 轨迹起点采样，归一化统计
   - `hybrid_loss()`: MSE(位置) + BCE(gripper) 混合损失
   - `InferencePolicy.act()`: 混合策略（规则 + 模型）

2. **`act_project/scripts/debug_compare_actions.py`**
   - 诊断脚本：对比 scripted policy vs model 动作

---

## 待办事项

- [ ] 测试调整后的混合策略（`obj_pos[2] + 0.01` 阈值）
- [ ] 如果 CLOSE 阶段 gripper 仍错误，强制规则 `gripper=-1`
- [ ] 评估完整 10 episode 成功率

---

## 关键发现

1. **行为克隆对离散动作敏感**: Gripper 的双峰分布不适合 MSE，需要 BCE
2. **数据分布偏斜**: 随机采样导致早期阶段数据不足
3. **模型泛化有限**: 早期阶段（APPROACH/DESCEND）模型表现差，需要规则兜底
4. **阈值要动态**: 固定阈值无法适应不同物体位置
