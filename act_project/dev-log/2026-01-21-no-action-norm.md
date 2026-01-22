# 2026-01-21 Dev Log - No Action Norm Trial

## 今日产出（关键结论）
- 去掉动作归一化后模型在 TRANSPORT/LOWER 仍失败：dz 持续为正（上升），gripper 在 handoff 后翻到 +1，完全不放下；成功率仍 0。
- 新训练跑通：`run-m3-17-no-action-norm`（100 epochs，混合策略保留），loss 收敛到 ~0.0043，但评估因缺失 `act_project/assets/pick_and_place_with_table.xml` 未执行。

## 过程与证据
- 训练：`train_act.py` 移除 action 归一化/反归一化，仅保留 obs 归一化；命令行训练到 `lerobot/act/training/run-m3-17-no-action-norm/ckpt/epoch_100.pt`。loss 末尾约 0.0043。
- 诊断对比（compare, steps=80，seed=42）：
  - TRANSPORT 41-50：model dz ≈ 0.32~0.47（期望接近 0 或轻微负），gripper 在 45 步起变为 +1。
  - LOWER 51-53：script dz=-0.784/-0.383，model dz ≈ 0.314~0.317；gripper = +1；偏差均值 TRANSPORT 1.62，LOWER 2.18。
- 数据覆盖：TRANSPORT 样本占比 ~39%（前 50 条 episode），数据量充足，问题不在覆盖。

## 当前阻碍
- 评估脚本缺资源：`act_project/assets/pick_and_place_with_table.xml` 未找到，自动 eval 报错。
- 模型后段行为错向：放置阶段学不到下压与闭合。

## 后续建议
- 若要快速拿到可用策略：将 InferencePolicy 的规则扩展至 TRANSPORT/LOWER/RELEASE，全程强制 gripper=-1，dz 直接用 scripted 值（或 clamp 至 -0.8），先解成功率。
- 若坚持模型端修复：恢复动作归一化或改 gripper 为分类任务，提升 gripper/LOWER 权重后重训；评估路径补齐缺失的 XML，再做验证。***
