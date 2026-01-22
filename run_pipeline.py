from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def train(args: argparse.Namespace) -> Path:
    dataset = Path(args.dataset_dir).resolve()
    if not dataset.exists():
        raise SystemExit(f"dataset_dir not found: {dataset}")
    run_id = args.run_dir or f"lerobot/act/training/run-ce-grip-{int(time.time())}"
    run_dir = Path(run_id)
    cmd = [
        sys.executable,
        "act_project/scripts/train_act.py",
        "--dataset-dir",
        str(dataset),
        "--run-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--eval-every",
        str(args.eval_every),
        "--device",
        args.device,
        "--normalize",
        str(args.normalize).lower(),
        "--gripper-weight",
        str(args.gripper_weight),
    ]
    _run(cmd)
    ckpt = run_dir / "ckpt" / f"epoch_{args.epochs:03d}.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    return ckpt


def evaluate(args: argparse.Namespace, ckpt_path: Path) -> dict:
    from act_project.scripts.train_act import ACTModel, Config, evaluate_policy  # noqa: WPS433
    import torch  # noqa: WPS433

    device = args.device if args.device in ("cuda", "cpu") else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    obs_dim = state_dict["obs_proj.weight"].shape[1]
    pos_dim = state_dict["head_pos.weight"].shape[0]
    act_dim = pos_dim + 1
    cfg_dict = ckpt["cfg"]
    cfg_dict["device"] = device
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    cfg.eval_episodes = args.eval_episodes
    cfg.max_steps = args.max_steps
    stats = ckpt.get("stats", None)
    model = ACTModel(obs_dim, act_dim, cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    eval_dir = ckpt_path.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_policy(model, cfg, eval_dir, stats)
    out = {
        "ckpt": str(ckpt_path),
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "device": device,
        "report": report,
        "timestamp": time.time(),
    }
    out_path = eval_dir / f"quick_eval_{args.eval_episodes}ep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def record_videos(args: argparse.Namespace, ckpt_path: Path) -> None:
    if args.video_episodes <= 0:
        return
    out_dir = Path(args.video_output).resolve()
    cmd = [
        sys.executable,
        "act_project/scripts/record_act_eval_videos.py",
        "--ckpt",
        str(ckpt_path),
        "--episodes",
        str(args.eval_episodes),
        "--video-episodes",
        str(args.video_episodes),
        "--max-steps",
        str(args.max_steps),
        "--output-dir",
        str(out_dir),
    ]
    _run(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click runner for LeRobot ACT sim2sim pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--dataset-dir", type=str, default="lerobot/act/datasets/pnp-sim2sim")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--gripper-weight", type=float, default=2.0)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--video-output", type=str, default="lerobot/act/reports/act_eval_videos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if args.mode == "train":
        ckpt_path = train(args)
    elif args.mode == "eval" and not ckpt_path:
        raise SystemExit("--ckpt is required for eval mode")
    elif args.mode == "all":
        ckpt_path = train(args)
    if ckpt_path is None:
        raise SystemExit("No checkpoint available")
    evaluate(args, ckpt_path)
    record_videos(args, ckpt_path)


if __name__ == "__main__":
    main()
