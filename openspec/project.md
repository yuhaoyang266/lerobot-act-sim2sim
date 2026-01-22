# Project: Panda ACT Sim-to-Sim Pipeline

## Purpose
在 MuJoCo 仿真中完成 Franka Panda 机械臂的 ACT 模仿学习全流程：
1. 验证 panda-mujoco-gym-ref 环境稳定性（Pick and Place 任务）
2. 编写 Scripted Policy 采集 500 条高质量示范轨迹
3. 训练 ACT (Action Chunking with Transformers) 策略
4. 在相同仿真环境中验证 ACT 策略性能

## Quality Gates（质量门禁）

| 阶段 | 验收标准 | 检查方式 | 通过才能继续 |
|------|----------|----------|-------------|
| M1: 环境验证 | Scripted Policy ≥ 90% 成功率 | 100 episodes 测试 + 视频 | ✅ |
| M2: Demo 采集 | 500 条轨迹，每条 `is_success=True` | HDF5 数据统计 | ✅ |
| M3: ACT 训练 | Loss 收敛稳定 | TensorBoard + 训练曲线 | ✅ |
| M4: 策略评估 | ACT Policy ≥ 90% 成功率 | 50 episodes 测试 + 视频 | ✅ |

## Tech Stack

| 类别 | 工具 | 版本 |
|------|------|------|
| Python 环境 | conda: `lerobot3` | Python 3.10+ |
| 深度学习 | PyTorch | 2.x + CUDA 12.x |
| 仿真 | MuJoCo | 3.x |
| 机器人环境 | panda-mujoco-gym-ref | gymnasium-robotics |
| 数据格式 | LeRobot HDF5 | - |
| 可视化 | TensorBoard, OpenCV | - |

## Hardware

```
GPU: RTX 5060 Ti 16GB (CUDA 核心 4608, 显存位宽 128-bit)
CPU: AMD Ryzen 7 7500H
RAM: 32GB DDR5
```

## Environment Details

```
环境 ID: FrankaPickAndPlaceSparse-v0 / FrankaPickAndPlaceDense-v0
任务: 抓取桌面物体并放置到随机目标位置
动作空间: Box(4,) → [dx, dy, dz, gripper] (连续)
观测空间: Dict
  - observation: (20,) → [ee_pos(3), ee_vel(3), fingers(1), obj_pos(3), obj_rot(3), obj_velp(3), obj_velr(3)]
  - achieved_goal: (3,) → 物体当前位置
  - desired_goal: (3,) → 目标位置
成功标准: ||achieved_goal - desired_goal|| < 0.05m
最大步数: 50 steps/episode
```

## Project Conventions

### Code Style
- 详细注释：每个函数、关键代码块都要有中文注释
- 教程级别：新手一看就能理解的清晰度
- 命名规范：遵循 LeRobot 格式（snake_case 变量，CamelCase 类）

### Architecture: ACT Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              ACT Sim-to-Sim Training Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │ ScriptedPolicy│──▶│ DemoCollector│──▶│ HDF5 Dataset │     │
│  │  (专家策略)   │   │  (轨迹录制)   │   │  (500 demos) │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
│                                               │              │
│                                               ▼              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │  Evaluation  │◀──│  ACTTrainer  │◀──│EpisodeDataset│     │
│  │  (策略评估)   │   │  (训练循环)   │   │  (数据加载)   │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
│         │                                                    │
│         ▼                                                    │
│  [视频 + 统计报告]                                            │
└─────────────────────────────────────────────────────────────┘
```

### Testing Strategy
- 每个里程碑都需要人工检查确认
- Demo 视频：随机抽取 5-10 条轨迹渲染
- 统计报告：成功率、平均步数、奖励分布
- 交互式检查：可选实时渲染观察

### Git Workflow
- `panda_mujoco_gym_ref/` 保持不动，作为参考
- 新代码放在独立目录（如 `act_project/`）

## Important Constraints

1. **分阶段验证**：每阶段必须达到 90% 成功率才能进入下一阶段
2. **数据质量**：只保存 `is_success=True` 的轨迹
3. **可复现性**：设置随机种子，记录所有超参数

## Directory Structure (Planned)

```
mujoco/
├── panda_mujoco_gym_ref/     # [保留] 原始环境参考
├── act_project/              # [新建] ACT 训练项目
│   ├── configs/              # 配置文件
│   ├── data/                 # 数据集
│   │   └── pnp_demos.h5      # 500 条示范轨迹
│   ├── scripts/              # 脚本
│   │   ├── scripted_policy.py
│   │   ├── collect_demos.py
│   │   ├── train_act.py
│   │   └── evaluate_policy.py
│   ├── checkpoints/          # 模型检查点
│   ├── videos/               # 评估视频
│   └── logs/                 # TensorBoard 日志
└── openspec/                 # OpenSpec 规范
```

## External Dependencies

```bash
# 在 lerobot3 环境中安装
conda activate lerobot3
pip install gymnasium gymnasium-robotics mujoco torch h5py tensorboard opencv-python tqdm
```

