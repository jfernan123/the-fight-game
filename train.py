"""Train the walker with PPO to run forward.

    python train.py --timesteps 2000000
    python train.py --resume runs/single/walker_500000_steps.zip
    python train.py --ankle dual

Progress: tensorboard --logdir tb_logs

Checkpoints/logs default to runs/<mode>/<run-id>/ and tb_logs/<mode>/<run-id>/ (mode = "baseline"
if --ankle is omitted, run-id = a timestamp) -- every launch gets its own folder, so two
independent runs can never collide or overwrite each other's checkpoints (this actually happened
once: a second run into the same runs/<mode>/ silently clobbered a completed run's walker_final.zip
via CheckpointCallback). This is the same layout train_all.py's parallel runs use, so watch.py/
watch_all.py can always find the newest checkpoint (they search recursively) regardless of whether
a run was launched directly (this script) or via train_all.py. Pass --checkpoint-dir/
--tensorboard-log explicitly to override.

To train all four ankle variants at once, use train_all.py instead -- it launches this script
once per variant, in parallel.
"""

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from walk_env import WalkEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the walker with PPO.")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-dir", default=None, help="default: runs/<mode>/<run-id>/ (run-id = timestamp)")
    parser.add_argument("--tensorboard-log", default=None, help="default: tb_logs/<mode>/<run-id>/ (run-id = timestamp)")
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .zip to resume from.")
    parser.add_argument(
        "--ankle", choices=["single", "dual", "detailed"], default=None,
        help="add an ankle joint (see walk_env.py/fight_env.py). Omit for the original rigid "
             "(no-ankle) foot.",
    )
    args = parser.parse_args()
    mode = args.ankle or "baseline"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = args.checkpoint_dir or os.path.join("runs", mode, run_id)
    tensorboard_log = args.tensorboard_log or os.path.join("tb_logs", mode, run_id)

    env = Monitor(WalkEnv(ankle_mode=args.ankle))
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=tensorboard_log, device="cpu")
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, device="cpu",
            n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.99,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path=checkpoint_dir, name_prefix="walker",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=args.resume is None,
    )

    final_path = os.path.join(checkpoint_dir, "walker_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
