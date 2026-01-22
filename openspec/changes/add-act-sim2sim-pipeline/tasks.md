# Tasks: ACT Sim-to-Sim Pipeline

## 0. 环境准备
- [ ] 0.1 激活 lerobot3 conda 环境
- [ ] 0.2 安装依赖包 (gymnasium, mujoco, torch, h5py, etc.)
- [ ] 0.3 验证 panda-mujoco-gym-ref 环境可正常运行
- [ ] 0.4 创建项目目录结构 `act_project/`

## 1. M1: Scripted Policy 开发与验证 🎯 ≥90%
- [ ] 1.1 分析 Pick and Place 任务的状态空间和动作空间
- [ ] 1.2 设计 Scripted Policy 状态机（approach → grasp → lift → move → place → release）
- [ ] 1.3 实现 `scripted_policy.py`
- [ ] 1.4 运行 100 episodes 测试
- [ ] 1.5 生成测试视频（5-10 条）
- [ ] 1.6 输出统计报告（成功率、平均步数、奖励分布）
- [ ] 1.7 **⏸️ 用户检查确认**（成功率 ≥ 90% 才继续）

## 2. M2: Demo 数据采集 🎯 500 条成功轨迹
- [ ] 2.1 实现 `collect_demos.py` 轨迹录制脚本
- [ ] 2.2 定义 LeRobot HDF5 数据格式
- [ ] 2.3 采集 500 条成功轨迹（只保存 is_success=True）
- [ ] 2.4 数据质量检查（维度、数值范围、完整性）
- [ ] 2.5 随机抽取 5 条轨迹渲染视频
- [ ] 2.6 **⏸️ 用户检查确认**（数据质量合格才继续）

## 3. M3: ACT 模型训练 🎯 Loss 收敛
- [ ] 3.1 实现 `EpisodeDataset` 数据加载器
- [ ] 3.2 实现 `ACTPolicy` 网络（State Encoder + Transformer + Action Decoder）
- [ ] 3.3 实现 `ACTTrainer` 训练循环
- [ ] 3.4 配置 TensorBoard 日志记录
- [ ] 3.5 训练模型（~100 epochs，监控 loss 曲线）
- [ ] 3.6 保存最佳检查点
- [ ] 3.7 **⏸️ 用户检查确认**（loss 收敛才继续）

## 4. M4: 策略评估 🎯 ≥90%
- [ ] 4.1 实现 `evaluate_policy.py` 评估脚本
- [ ] 4.2 加载训练好的 ACT 模型
- [ ] 4.3 运行 50 episodes 评估
- [ ] 4.4 生成评估视频（随机 10 条）
- [ ] 4.5 输出最终报告（成功率、对比 Scripted Policy）
- [ ] 4.6 **⏸️ 用户最终确认**

## 5. 文档与清理
- [ ] 5.1 更新 README.md
- [ ] 5.2 整理代码注释
- [ ] 5.3 归档 OpenSpec change
