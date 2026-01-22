"""
M3 诊断脚本：对比 scripted policy 和 ACT 模型的动作输出

功能：
- 加载训练好的 checkpoint
- 在同一个环境状态下，分别用 scripted policy 和 model 计算动作
- 打印前 N 步的对比结果，分析偏差来源

使用方法：
    conda run -n lerobot3 python act_project/scripts/debug_compare_actions.py \
      --ckpt lerobot/act/training/run-m3-7/ckpt/epoch_300_alt.pt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

# 路径设置：确保本地包可导入
ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env  # noqa: E402
from act_project.scripts.scripted_policy_table import GraspState, ScriptedPickAndPlaceTablePolicy  # noqa: E402


# === 从 train_act.py 复制的模型定义（避免循环导入）===
import math
import torch.nn as nn


@dataclass
class Config:
    """模型配置（与 train_act.py 保持一致）"""
    dataset_dir: str = ""
    run_dir: str = ""
    chunk_size: int = 20
    observation_horizon: int = 2
    action_horizon: int = 10
    hidden_dim: int = 256
    num_layers: int = 4
    nheads: int = 8
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 300
    eval_every: int = 50
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps: int = 220
    eval_episodes: int = 10
    normalize: bool = True
    gripper_weight: float = 2.0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ACTModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Config):
        super().__init__()
        self.pos_dim = act_dim - 1
        self.obs_proj = nn.Linear(obs_dim, cfg.hidden_dim)
        self.pos_enc = PositionalEncoding(cfg.hidden_dim, max_len=cfg.chunk_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nheads,
            dim_feedforward=cfg.hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.head_pos = nn.Linear(cfg.hidden_dim, self.pos_dim)
        self.head_gripper = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, obs_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.obs_proj(obs_seq)
        x = self.pos_enc(x)
        x = self.encoder(x)
        pos = self.head_pos(x)
        gripper_logits = self.head_gripper(x)
        return pos, gripper_logits


class InferencePolicy:
    """模型推理 Policy（与 train_act.py 保持一致，支持归一化和混合策略）"""
    def __init__(self, model: ACTModel, cfg: Config, stats: dict | None = None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.window: List[np.ndarray] = []

        # 归一化统计量
        self.stats = stats
        self.normalize = (
            getattr(cfg, "normalize", False)
            and stats is not None
            and "obs_mean" in stats
            and "obs_std" in stats
        )
        if self.normalize:
            self.obs_mean = stats["obs_mean"]
            self.obs_std = stats["obs_std"]

        # === 混合策略状态变量 ===
        self.close_step_counter = 0
        self.lift_step_counter = 0
        self.initial_obj_z: float | None = None
        self.grasp_confirmed = False

    def reset(self) -> None:
        self.window = []
        self.close_step_counter = 0
        self.lift_step_counter = 0
        self.initial_obj_z = None
        self.grasp_confirmed = False

    def act(self, obs: dict) -> np.ndarray:
        o = np.concatenate([obs["observation"], obs["desired_goal"]])
        aug = self._augment_single(o)

        # 应用归一化
        if self.normalize:
            aug = (aug - self.obs_mean) / self.obs_std

        self.window.append(aug)
        if len(self.window) > self.cfg.chunk_size:
            self.window = self.window[-self.cfg.chunk_size:]
        obs_seq = np.stack(self.window, axis=0)
        pad_len = self.cfg.chunk_size - len(obs_seq)
        if pad_len > 0:
            obs_seq = np.concatenate([np.tile(obs_seq[0:1], (pad_len, 1)), obs_seq], axis=0)
        obs_tensor = torch.from_numpy(obs_seq).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pos_seq, gripper_logits_seq = self.model(obs_tensor)
        pos_action = pos_seq[0, -1].cpu().numpy()
        gripper_logits = gripper_logits_seq[0, -1].cpu().numpy()
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(pos_action, -1.0, 1.0)
        gripper_class = int(np.argmax(gripper_logits))
        action[3] = 1.0 if gripper_class == 1 else -1.0

        # 获取 EE 和物体位置
        ee_pos = obs["observation"][0:3]
        obj_pos = obs["observation"][7:10]
        ee_obj_xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])

        # 记录初始物体高度
        if self.initial_obj_z is None:
            self.initial_obj_z = obj_pos[2]

        # === 混合策略：APPROACH/DESCEND/CLOSE/LIFT 阶段使用规则控制 ===
        APPROACH_XY_THRESHOLD = 0.02
        GRASP_HEIGHT = 0.025
        SAFE_HEIGHT = 0.22
        POS_GAIN = 10.0
        CLOSE_STEPS = 25
        CLOSE_PUSH = -0.03
        MIN_LIFT_DELTA = 0.05
        LIFT_STEPS_MAX = 30

        if ee_obj_xy_dist > APPROACH_XY_THRESHOLD:
            # APPROACH
            target = np.array([obj_pos[0], obj_pos[1], SAFE_HEIGHT])
            delta = target - ee_pos
            action[:3] = np.clip(delta * POS_GAIN, -1.0, 1.0)
            action[3] = 1.0
            self.close_step_counter = 0
            self.lift_step_counter = 0
        elif ee_pos[2] > obj_pos[2] + 0.01:
            # DESCEND
            target = np.array([obj_pos[0], obj_pos[1], GRASP_HEIGHT])
            delta = target - ee_pos
            action[:3] = np.clip(delta * POS_GAIN, -1.0, 1.0)
            action[3] = 1.0
            self.close_step_counter = 0
            self.lift_step_counter = 0
        elif self.close_step_counter < CLOSE_STEPS:
            # CLOSE
            action[:3] = np.array([0.0, 0.0, CLOSE_PUSH])
            action[3] = -1.0
            self.close_step_counter += 1
            self.lift_step_counter = 0
        elif not self.grasp_confirmed and self.lift_step_counter < LIFT_STEPS_MAX:
            # LIFT (规则控制)
            target = np.array([ee_pos[0], ee_pos[1], SAFE_HEIGHT])
            delta = target - ee_pos
            action[:3] = np.clip(delta * POS_GAIN, -1.0, 1.0)
            action[3] = -1.0
            self.lift_step_counter += 1
            if self.initial_obj_z is not None and (obj_pos[2] - self.initial_obj_z) > MIN_LIFT_DELTA:
                self.grasp_confirmed = True
        # else: 模型控制 TRANSPORT/LOWER/RELEASE

        action[:3] = np.clip(action[:3], -1.0, 1.0)
        return action

    @staticmethod
    def _augment_single(obs_row: np.ndarray) -> np.ndarray:
        ee = obs_row[0:3]
        obj = obs_row[7:10]
        goal = obs_row[-3:]
        rel_goal_obj = goal - obj
        rel_obj_ee = obj - ee
        return np.concatenate([obs_row, rel_goal_obj, rel_obj_ee])


def load_model(ckpt_path: Path, device: str = "cuda") -> tuple[ACTModel, Config, dict | None]:
    """加载 checkpoint 并返回模型、配置和归一化统计量"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 从 checkpoint 恢复配置
    cfg_dict = ckpt["cfg"]
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    cfg.device = device

    # 推断 obs_dim 和 act_dim（从 state_dict 反推）
    state_dict = ckpt["model"]
    obs_dim = state_dict["obs_proj.weight"].shape[1]
    if "head_pos.weight" not in state_dict or "head_gripper.weight" not in state_dict:
        raise ValueError("Checkpoint 缺少 head_pos/head_gripper（需要使用分类 gripper 的新模型）")
    pos_dim = state_dict["head_pos.weight"].shape[0]
    act_dim = pos_dim + 1

    # 加载归一化统计量（如果存在）
    stats = ckpt.get("stats", None)
    normalize_str = "Yes" if stats is not None else "No"

    print(f"模型配置: obs_dim={obs_dim}, act_dim={act_dim}, chunk_size={cfg.chunk_size}")
    print(f"          hidden_dim={cfg.hidden_dim}, num_layers={cfg.num_layers}, nheads={cfg.nheads}")
    print(f"          normalize={normalize_str}")
    if stats is not None and "obs_mean" in stats and "obs_std" in stats:
        print(f"          obs_mean[:3]={stats['obs_mean'][:3]}")
        print(f"          obs_std[:3]={stats['obs_std'][:3]}")

    # 创建并加载模型
    model = ACTModel(obs_dim, act_dim, cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, cfg, stats


def compare_actions(
    ckpt_path: Path,
    num_steps: int = 20,
    seed: int = 42,
    device: str = "cuda",
    use_scripted_trajectory: bool = True,
) -> None:
    """
    对比 scripted policy 和 ACT 模型的动作输出

    参数：
        ckpt_path: checkpoint 路径
        num_steps: 对比的步数
        seed: 随机种子
        device: 计算设备
        use_scripted_trajectory: 如果为 True，用 scripted action 执行来保持在正确轨迹上
    """
    print("=" * 60)
    print("M3 诊断：Scripted Policy vs ACT Model 动作对比")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1] 加载 checkpoint: {ckpt_path}")
    model, cfg, stats = load_model(ckpt_path, device)
    inference_policy = InferencePolicy(model, cfg, stats)

    # 2. 创建环境和 scripted policy
    print("\n[2] 创建环境和 scripted policy...")
    env = make_env(render_mode=None, randomize_target=False)
    scripted_policy = ScriptedPickAndPlaceTablePolicy()

    # 3. Reset
    print(f"\n[3] Reset 环境 (seed={seed})...")
    obs, info = env.reset(seed=seed)
    scripted_policy.reset()
    inference_policy.reset()

    # 打印初始状态
    ee_pos = obs["observation"][0:3]
    obj_pos = obs["observation"][7:10]
    goal = obs["desired_goal"]
    print(f"    初始 EE 位置:  {ee_pos}")
    print(f"    初始物体位置: {obj_pos}")
    print(f"    目标位置:     {goal}")

    # 4. 动作对比
    print(f"\n[4] 动作对比 (前 {num_steps} 步):")
    print("-" * 100)
    print(f"{'Step':>4} | {'State':<12} | {'Scripted Action':<32} | {'Model Action':<32} | {'Diff':>8}")
    print("-" * 100)

    diffs = []
    per_dim_diffs = [[] for _ in range(4)]  # dx, dy, dz, gripper
    state_diffs = {}  # 按状态统计偏差

    for step in range(num_steps):
        # 计算两种动作
        scripted_action = scripted_policy.act(obs)
        model_action = inference_policy.act(obs)

        # 计算偏差
        diff = np.linalg.norm(scripted_action - model_action)
        diffs.append(diff)

        for i in range(4):
            per_dim_diffs[i].append(abs(scripted_action[i] - model_action[i]))

        # 按状态统计
        state_name = scripted_policy.state.name
        if state_name not in state_diffs:
            state_diffs[state_name] = []
        state_diffs[state_name].append(diff)

        # 格式化输出
        scripted_str = f"[{scripted_action[0]:>6.3f}, {scripted_action[1]:>6.3f}, {scripted_action[2]:>6.3f}, {scripted_action[3]:>6.3f}]"
        model_str = f"[{model_action[0]:>6.3f}, {model_action[1]:>6.3f}, {model_action[2]:>6.3f}, {model_action[3]:>6.3f}]"

        print(f"{step:>4} | {state_name:<12} | {scripted_str} | {model_str} | {diff:>8.4f}")

        # 执行动作
        if use_scripted_trajectory:
            # 用 scripted action 执行，保持在正确轨迹上
            obs, _, terminated, truncated, info = env.step(scripted_action)
        else:
            # 用 model action 执行，观察真实表现
            obs, _, terminated, truncated, info = env.step(model_action)

        if terminated or truncated:
            print(f"  [!] Episode 结束 at step {step}")
            break

        if scripted_policy.state == GraspState.DONE:
            print(f"  [!] Scripted policy 完成 at step {step}")
            break

    env.close()

    # 5. 统计分析
    print("\n" + "=" * 60)
    print("[5] 统计分析")
    print("=" * 60)

    print(f"\n总体偏差:")
    print(f"  Mean diff:  {np.mean(diffs):.4f}")
    print(f"  Std diff:   {np.std(diffs):.4f}")
    print(f"  Max diff:   {np.max(diffs):.4f}")
    print(f"  Min diff:   {np.min(diffs):.4f}")

    print(f"\n各维度偏差 (dx, dy, dz, gripper):")
    dim_names = ["dx", "dy", "dz", "gripper"]
    for i, name in enumerate(dim_names):
        mean_d = np.mean(per_dim_diffs[i])
        max_d = np.max(per_dim_diffs[i])
        print(f"  {name:>7}: mean={mean_d:.4f}, max={max_d:.4f}")

    print(f"\n各阶段偏差:")
    for state_name, state_diff_list in state_diffs.items():
        if state_diff_list:
            print(f"  {state_name:<12}: mean={np.mean(state_diff_list):.4f}, count={len(state_diff_list)}")

    # 6. 诊断建议
    print("\n" + "=" * 60)
    print("[6] 诊断建议")
    print("=" * 60)

    mean_diff = np.mean(diffs)
    gripper_mean = np.mean(per_dim_diffs[3])

    if mean_diff < 0.1:
        print("  [+] 总体偏差较小 (< 0.1)，模型学习效果较好")
    elif mean_diff < 0.3:
        print("  [!] 中等偏差 (0.1-0.3)，可能存在累积误差问题")
    else:
        print("  [X] 偏差较大 (> 0.3)，需要检查输入预处理或模型结构")

    if gripper_mean > np.mean(per_dim_diffs[:3]):
        print("  [!] Gripper 维度偏差较大，考虑将 gripper 作为离散动作单独处理")

    # 检查早期偏差
    early_diffs = diffs[:5] if len(diffs) >= 5 else diffs
    if np.mean(early_diffs) > mean_diff * 1.5:
        print("  [!] 早期 (step 0-5) 偏差较大，检查 obs 归一化和初始状态处理")

    # 检查特定阶段
    for state_name, state_diff_list in state_diffs.items():
        if state_diff_list and np.mean(state_diff_list) > mean_diff * 1.5:
            print(f"  [!] {state_name} 阶段偏差较大，检查该阶段的数据分布")


def run_model_only_episode(
    ckpt_path: Path,
    num_steps: int = 220,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """仅用模型执行一个完整 episode，观察真实行为"""
    print("=" * 60)
    print("M3 诊断：Model-Only Episode 执行")
    print("=" * 60)

    model, cfg, stats = load_model(ckpt_path, device)
    inference_policy = InferencePolicy(model, cfg, stats)

    env = make_env(render_mode=None, randomize_target=False)
    obs, info = env.reset(seed=seed)
    inference_policy.reset()

    ee_pos = obs["observation"][0:3]
    obj_pos = obs["observation"][7:10]
    goal = obs["desired_goal"]
    print(f"初始 EE:  {ee_pos}")
    print(f"初始 Obj: {obj_pos}")
    print(f"目标:     {goal}")

    print(f"\n执行中 (max {num_steps} steps)...")
    for step in range(num_steps):
        action = inference_policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            ee = obs["observation"][0:3]
            obj = obs["observation"][7:10]
            dist = np.linalg.norm(obj[:2] - goal[:2])
            print(f"  Step {step:>3}: EE={ee}, Obj={obj}, dist_to_goal={dist:.3f}")

        if info.get("is_success"):
            print(f"\n[SUCCESS] at step {step + 1}")
            break
        if terminated or truncated:
            print(f"\n[TRUNCATED] at step {step + 1}")
            break

    env.close()


def main():
    parser = argparse.ArgumentParser(description="M3 诊断：对比 scripted 和 model 动作")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint 路径")
    parser.add_argument("--steps", type=int, default=50, help="对比步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["compare", "model-only", "both"], default="both",
                        help="运行模式: compare=对比分析, model-only=仅模型执行, both=两者都运行")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    if args.mode in ("compare", "both"):
        compare_actions(
            ckpt_path=ckpt_path,
            num_steps=args.steps,
            seed=args.seed,
            device=args.device,
            use_scripted_trajectory=True,
        )

    if args.mode in ("model-only", "both"):
        print("\n\n")
        run_model_only_episode(
            ckpt_path=ckpt_path,
            num_steps=220,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
