from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT.parent):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

from act_project.envs.pick_place_table_env import make_env  # noqa: E402


@dataclass
class Config:
    dataset_dir: str
    run_dir: str
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
    # 归一化和 gripper 权重
    normalize: bool = True
    gripper_weight: float = 2.0  # gripper 维度 loss 权重


class EpisodeDataset(Dataset):
    SAMPLING_MULTIPLIER = 4

    def __init__(self, root: Path, chunk_size: int, normalize: bool = True):
        self.files = sorted((root / "episodes").glob("episode_*.hdf5"))
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.data = []

        # 第一遍：加载所有数据
        for f in self.files:
            with h5py.File(f, "r") as h5:
                obs = h5["observations"][:]
                act = h5["actions"][:]
                self.data.append((obs, act))

        # 计算归一化统计量（仅用于 obs）
        all_obs = []
        for obs, _ in self.data:
            aug_obs = self._augment(obs)
            all_obs.append(aug_obs)
        all_obs = np.concatenate(all_obs, axis=0)

        self.obs_mean = all_obs.mean(axis=0).astype(np.float32)
        self.obs_std = all_obs.std(axis=0).astype(np.float32)
        self.obs_std = np.clip(self.obs_std, 1e-6, None)  # 防止除零

        # action 范围，用于人工验证
        all_act = np.concatenate([a for _, a in self.data], axis=0)
        self.act_min = all_act.min(axis=0).astype(np.float32)
        self.act_max = all_act.max(axis=0).astype(np.float32)

        print(f"[Dataset] obs_mean: {self.obs_mean[:6]}... (dim={len(self.obs_mean)})")
        print(f"[Dataset] obs_std:  {self.obs_std[:6]}...")
        print(f"[Dataset] act_min:  {self.act_min}")
        print(f"[Dataset] act_max:  {self.act_max}")

    def get_stats(self) -> Dict[str, np.ndarray]:
        """返回归一化统计量，用于保存和推理"""
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
        }

    def __len__(self) -> int:
        return len(self.data) * self.SAMPLING_MULTIPLIER

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ep_idx = idx % len(self.data)
        obs, act = self.data[ep_idx]
        if len(obs) <= self.chunk_size:
            start = 0
        else:
            # 50% 概率从轨迹开始采样（确保包含 APPROACH 阶段）
            # 50% 概率随机采样
            if np.random.random() < 0.5:
                start = 0  # 从轨迹开始
            else:
                start = np.random.randint(0, len(obs) - self.chunk_size)
        end = start + self.chunk_size
        obs_w = obs[start:end]
        obs_w = self._augment(obs_w)
        act_w = act[start:end]

        # 应用归一化（仅 obs）
        if self.normalize:
            obs_w = (obs_w - self.obs_mean) / self.obs_std

        return torch.from_numpy(obs_w).float(), torch.from_numpy(act_w).float()

    @staticmethod
    def _augment(obs_seq: np.ndarray) -> np.ndarray:
        """
        Augment observation with relative features:
        - goal - object position
        - object - ee position
        """
        ee_pos = obs_seq[:, 0:3]
        obj_pos = obs_seq[:, 7:10]
        goal_pos = obs_seq[:, -3:]
        rel_goal_obj = goal_pos - obj_pos
        rel_obj_ee = obj_pos - ee_pos
        return np.concatenate([obs_seq, rel_goal_obj, rel_obj_ee], axis=1)


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

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.obs_proj(obs_seq)
        x = self.pos_enc(x)
        x = self.encoder(x)
        pos = self.head_pos(x)
        gripper_logits = self.head_gripper(x)
        return pos, gripper_logits


def hybrid_loss(
    pos_pred: torch.Tensor,
    gripper_logits: torch.Tensor,
    target: torch.Tensor,
    gripper_weight: float = 2.0,
    pos_weight: float = 2.68,
) -> torch.Tensor:
    """
    混合 loss：位置维度用 MSE，gripper 用二分类 CrossEntropy
    pos_pred: (B, T, 3)
    gripper_logits: (B, T, 2)
    target: (B, T, 4) 其中 target[..., 3] in {-1, +1}
    """
    pos_target = target[..., :3]
    pos_loss = F.mse_loss(pos_pred, pos_target)

    gripper_target = ((target[..., 3] + 1) / 2).long().clamp(0, 1)
    class_weights = torch.tensor([1.0, pos_weight], device=gripper_logits.device)
    gripper_loss = F.cross_entropy(
        gripper_logits.view(-1, 2), gripper_target.view(-1), weight=class_weights
    )

    return pos_loss + gripper_weight * gripper_loss


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset_path = Path(cfg.dataset_dir)
    ds = EpisodeDataset(dataset_path, cfg.chunk_size, normalize=cfg.normalize)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # infer dims
    sample_obs, sample_act = ds[0]
    obs_dim = sample_obs.shape[-1]
    act_dim = sample_act.shape[-1]

    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "ckpt"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 保存归一化统计量
    stats = ds.get_stats()
    stats_path = run_dir / "norm_stats.npz"
    np.savez(stats_path, **stats)
    print(f"[Train] 归一化统计量已保存到 {stats_path}")

    device = torch.device(cfg.device)
    model = ACTModel(obs_dim, act_dim, cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"[Train] Using hybrid loss: MSE for position, CE for gripper (weight={cfg.gripper_weight})")

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            pos_pred, gripper_logits = model(obs_batch)
            loss = hybrid_loss(pos_pred, gripper_logits, act_batch, cfg.gripper_weight)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)

        rec = {"epoch": epoch, "loss": epoch_loss, "time": time.time()}
        metrics_file.write(json.dumps(rec) + "\n")
        metrics_file.flush()

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            # 将 stats 一起保存到 checkpoint 中
            ckpt_data = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "stats": stats,  # 归一化统计量
            }
            save_error: Exception | None = None
            for _ in range(2):
                try:
                    torch.save(ckpt_data, ckpt_path)
                    save_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    save_error = exc
                    time.sleep(0.5)
            if save_error:
                fallback_path = ckpt_dir / f"epoch_{epoch:03d}_alt.pt"
                try:
                    torch.save(ckpt_data, fallback_path, _use_new_zipfile_serialization=False)
                    save_error = None
                except Exception as exc:  # noqa: BLE001
                    save_error = exc
            if save_error:
                raise save_error
            eval_report = evaluate_policy(model, cfg, eval_dir, stats)
            metrics_file.write(json.dumps({"epoch": epoch, "eval": eval_report, "time": time.time()}) + "\n")
            metrics_file.flush()

    metrics_file.close()


def evaluate_policy(model: ACTModel, cfg: Config, eval_dir: Path, stats: Dict[str, np.ndarray] | None = None) -> Dict:
    env = None
    last_err: Exception | None = None
    for _ in range(2):
        try:
            env = gym.wrappers.TimeLimit(
                make_env(render_mode=None, randomize_target=False),
                max_episode_steps=cfg.max_steps,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if env is None:
        return {"success_rate": 0.0, "mean_steps": 0.0, "error": str(last_err) if last_err else "unknown"}
    policy = InferencePolicy(model, cfg, stats)
    successes = 0
    steps_list = []
    for ep in range(cfg.eval_episodes):
        obs, info = env.reset()
        policy.reset()
        for step in range(cfg.max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success"):
                successes += 1
                steps_list.append(step + 1)
                break
            if truncated:
                steps_list.append(step + 1)
                break
    env.close()
    rate = successes / cfg.eval_episodes
    mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
    return {"success_rate": rate, "mean_steps": mean_steps}


class InferencePolicy:
    """推理策略，支持归一化"""
    def __init__(self, model: ACTModel, cfg: Config, stats: Dict[str, np.ndarray] | None = None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.window: List[np.ndarray] = []

        # 归一化统计量
        self.stats = stats
        self.normalize = (
            cfg.normalize
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
            self.window = self.window[-self.cfg.chunk_size :]
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

        # 获取 EE 和物体位置（用于启发式覆盖）
        ee_pos = obs["observation"][0:3]
        obj_pos = obs["observation"][7:10]
        ee_obj_dist = np.linalg.norm(ee_pos - obj_pos)
        ee_obj_xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])

        # 记录初始物体高度（用于判断是否抓住物体）
        if self.initial_obj_z is None:
            self.initial_obj_z = obj_pos[2]

        # === 混合策略：APPROACH/DESCEND/CLOSE/LIFT 阶段使用规则控制 ===
        # 诊断显示模型在 APPROACH/DESCEND 完全失效（diff > 2.0）
        # 仅在 TRANSPORT 阶段（diff=0.21）让模型接管
        
        # 参数（与 scripted_policy_table.py 保持一致）
        APPROACH_XY_THRESHOLD = 0.02  # XY 距离阈值
        GRASP_HEIGHT = 0.025  # 抓取高度
        SAFE_HEIGHT = 0.22  # 安全高度
        POS_GAIN = 10.0
        CLOSE_STEPS = 25  # CLOSE 阶段步数
        CLOSE_PUSH = -0.03  # CLOSE 阶段微下压（与 scripted 一致）
        MIN_LIFT_DELTA = 0.05  # 判断物体被抬起的最小高度差
        LIFT_STEPS_MAX = 30  # LIFT 阶段最大步数

        if ee_obj_xy_dist > APPROACH_XY_THRESHOLD:
            # APPROACH 阶段：移动到物体上方，张开夹爪
            target = np.array([obj_pos[0], obj_pos[1], SAFE_HEIGHT])
            delta = target - ee_pos
            action[:3] = np.clip(delta * POS_GAIN, -1.0, 1.0)
            action[3] = 1.0  # 强制张开
            self.close_step_counter = 0
            self.lift_step_counter = 0
            
        elif ee_pos[2] > obj_pos[2] + 0.01:
            # DESCEND 阶段：下降到物体高度，张开夹爪
            target = np.array([obj_pos[0], obj_pos[1], GRASP_HEIGHT])
            delta = target - ee_pos
            action[:3] = np.clip(delta * POS_GAIN, -1.0, 1.0)
            action[3] = 1.0  # 张开
            self.close_step_counter = 0
            self.lift_step_counter = 0
            
        elif self.close_step_counter < CLOSE_STEPS:
            # CLOSE 阶段：关闭夹爪 + 微下压
            action[:3] = np.array([0.0, 0.0, CLOSE_PUSH])
            action[3] = -1.0  # 强制关闭
            self.close_step_counter += 1
            self.lift_step_counter = 0
            
        elif not self.grasp_confirmed and self.lift_step_counter < LIFT_STEPS_MAX:
            # LIFT 阶段（规则控制）：抬升到安全高度
            target = np.array([ee_pos[0], ee_pos[1], SAFE_HEIGHT])
            delta = target - ee_pos
            action[:3] = np.clip(delta * POS_GAIN, -1.0, 1.0)
            action[3] = -1.0  # 保持夹紧
            self.lift_step_counter += 1
            
            # 检查物体是否被抬起
            if self.initial_obj_z is not None and (obj_pos[2] - self.initial_obj_z) > MIN_LIFT_DELTA:
                self.grasp_confirmed = True
                
        # else: 模型控制 TRANSPORT/LOWER/RELEASE（diff=0.21 可接受）

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(ROOT.parent / "lerobot" / "act" / "datasets" / "pnp-sim2sim"))
    parser.add_argument("--run-dir", type=str, default=str(ROOT.parent / "lerobot" / "act" / "training" / f"run-{int(time.time())}"))
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--observation-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 归一化和 gripper 权重
    parser.add_argument("--normalize", type=lambda x: x.lower() == "true", default=True, help="是否启用 obs 归一化")
    parser.add_argument("--gripper-weight", type=float, default=2.0, help="gripper 维度 loss 权重")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        dataset_dir=args.dataset_dir,
        run_dir=args.run_dir,
        chunk_size=args.chunk_size,
        observation_horizon=args.observation_horizon,
        action_horizon=args.action_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nheads=args.nheads,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_every=args.eval_every,
        seed=args.seed,
        device=args.device,
        eval_episodes=args.eval_episodes,
        normalize=args.normalize,
        gripper_weight=args.gripper_weight,
    )
    train(cfg)


if __name__ == "__main__":
    main()
