"""Watch a trained walker checkpoint run.

    python watch.py                       # newest checkpoint in runs/
    python watch.py runs/walker_final.zip  # a specific one
"""

import argparse
import glob
import os

from stable_baselines3 import PPO

from walk_env import WalkEnv


def _latest_checkpoint(checkpoint_dir: str) -> str:
    # Recursive since train.py now nests checkpoints under a per-run timestamp folder
    # (runs/<mode>/<run-id>/) -- "**" also matches zero intermediate dirs, so this still finds
    # checkpoints saved directly in checkpoint_dir by an older, un-timestamped run.
    candidates = glob.glob(os.path.join(checkpoint_dir, "**", "*.zip"), recursive=True)
    if not candidates:
        raise SystemExit(f"No checkpoint .zip files found under '{checkpoint_dir}'.")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained walker checkpoint run.")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to a saved PPO checkpoint .zip. Omit to auto-pick the most recently "
             "modified .zip in --checkpoint-dir -- safe to run mid-training to see progress.",
    )
    parser.add_argument("--checkpoint-dir", default="runs")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    print(f"Loading {checkpoint}")

    env = WalkEnv(render_mode="human")
    model = PPO.load(checkpoint)

    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: total_reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
