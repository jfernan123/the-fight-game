"""Train the red fighter with PPO against fight_env's scripted blue opponent. Run from the
project root:

    python fighter/train_fighter.py --timesteps 2000000
    python fighter/train_fighter.py --resume runs/fighter/checkpoints/fighter_500000_steps.zip

Progress: tensorboard --logdir runs/fighter/tb_logs
"""

import argparse
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from fight_gym import FightGymEnv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # this file lives in fighter/, root is one level up


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fighter with PPO against the scripted opponent.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-dir", default=str(_PROJECT_ROOT / "runs" / "fighter" / "checkpoints"))
    parser.add_argument("--tensorboard-log", default=str(_PROJECT_ROOT / "runs" / "fighter" / "tb_logs"))
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .zip to resume from.")
    parser.add_argument(
        "--device", default="cpu", choices=["cuda", "cpu", "auto"],
        help="cpu by default -- measured ~3-4x faster than cuda here (single env, small policy, "
             "so PPO is CPU-bound on environment stepping, not network compute). Pass --device "
             "cuda if you scale up to many parallel environments later, where GPU starts winning.",
    )
    args = parser.parse_args()

    env = Monitor(FightGymEnv())
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=args.tensorboard_log, device=args.device)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log, device=args.device,
            n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.99,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path=args.checkpoint_dir, name_prefix="fighter",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=args.resume is None,
    )

    final_path = os.path.join(args.checkpoint_dir, "fighter_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
