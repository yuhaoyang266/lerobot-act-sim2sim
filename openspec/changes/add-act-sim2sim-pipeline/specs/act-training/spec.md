# ACT Training Pipeline Specification

## ADDED Requirements

### Requirement: Scripted Policy for Pick and Place
系统 SHALL 提供一个基于状态机的 Scripted Policy，用于 Franka Panda 机械臂的 Pick and Place 任务。

#### Scenario: 成功抓取并放置
- **GIVEN** 物体在桌面随机位置，目标在工作空间内
- **WHEN** 执行 Scripted Policy
- **THEN** 机械臂接近物体 → 抓取 → 提起 → 移动到目标 → 放置 → 释放
- **AND** `info['is_success'] == True`

#### Scenario: 环境边界处理
- **GIVEN** 物体或目标位于工作空间边缘
- **WHEN** 执行 Scripted Policy
- **THEN** 动作被裁剪在合法范围内
- **AND** 不触发环境异常

---

### Requirement: Demo Collection Script
系统 SHALL 提供轨迹录制脚本，将成功的示范保存为 LeRobot HDF5 格式。

#### Scenario: 采集 500 条成功轨迹
- **GIVEN** Scripted Policy 成功率 ≥ 90%
- **WHEN** 运行采集脚本
- **THEN** 保存 500 条 `is_success=True` 的轨迹
- **AND** 每条轨迹包含完整的 observations, actions, goals

#### Scenario: 数据格式验证
- **GIVEN** 采集完成的 HDF5 文件
- **WHEN** 检查数据结构
- **THEN** observations shape = [T, 20]
- **AND** actions shape = [T, 4]
- **AND** 所有数值在合理范围内

---

### Requirement: ACT Policy Network
系统 SHALL 实现基于 Transformer 的 ACT 策略网络。

#### Scenario: 前向传播维度正确
- **GIVEN** 输入 states [B, T, 20]
- **WHEN** 执行 forward pass
- **THEN** 输出 actions [B, T, 4]

#### Scenario: 训练收敛
- **GIVEN** 500 条示范数据
- **WHEN** 训练 100 epochs
- **THEN** MSE Loss 单调下降并收敛

---

### Requirement: ACT Trainer
系统 SHALL 提供完整的训练循环，支持检查点保存和 TensorBoard 监控。

#### Scenario: 训练流程
- **GIVEN** 配置好的数据集和模型
- **WHEN** 调用 `trainer.train(num_epochs=100)`
- **THEN** 每 epoch 打印 loss
- **AND** 每 10 epochs 保存检查点
- **AND** 记录到 TensorBoard

#### Scenario: 模型保存与加载
- **GIVEN** 训练完成的模型
- **WHEN** 保存并重新加载
- **THEN** 模型权重完全一致

---

### Requirement: Policy Evaluation
系统 SHALL 提供策略评估脚本，输出成功率统计和可视化视频。

#### Scenario: 评估 50 episodes
- **GIVEN** 训练好的 ACT 模型
- **WHEN** 运行评估脚本
- **THEN** 输出成功率、平均步数、奖励分布

#### Scenario: 生成评估视频
- **GIVEN** 评估过程中的轨迹
- **WHEN** 渲染视频
- **THEN** 保存 MP4 文件，可视化机械臂运动

---

### Requirement: Quality Gate Enforcement
系统 SHALL 在每个阶段强制执行质量门禁检查。

#### Scenario: M1 门禁
- **GIVEN** Scripted Policy 测试完成
- **WHEN** 成功率 < 90%
- **THEN** 阻止进入 M2 阶段
- **AND** 提示用户检查并调整

#### Scenario: M4 门禁
- **GIVEN** ACT Policy 评估完成
- **WHEN** 成功率 ≥ 90%
- **THEN** 标记项目完成
- **AND** 输出最终对比报告
