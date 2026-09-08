"""Watch a trained fighter checkpoint fight the scripted opponent in the viewer. Run from the
project root:

    python fighter/watch_fighter.py                                          # newest checkpoint
    python fighter/watch_fighter.py runs/fighter/checkpoints/fighter_final.zip # a specific one
"""

import argparse
import glob
import os
from pathlib import Path

from stable_baselines3 import PPO

from fight_gym import FightGymEnv

_DEFAULT_CHECKPOINT_DIR = str(Path(__file__).resolve().parent.parent / "runs" / "fighter" / "checkpoints")


def _latest_checkpoint(checkpoint_dir: str) -> str:
    candidates = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not candidates:
        raise SystemExit(f"No checkpoint .zip files found in '{checkpoint_dir}'.")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained fighter checkpoint fight.")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to a saved PPO checkpoint .zip. Omit to auto-pick the most recently "
             "modified .zip in --checkpoint-dir -- safe to run mid-training to see progress.",
    )
    parser.add_argument("--checkpoint-dir", default=_DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    print(f"Loading {checkpoint}")

    env = FightGymEnv(render_mode="human")
    model = PPO.load(checkpoint)

    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        if info["opponent_hp"] <= 0:
            winner = "red (agent)"
        elif info["agent_hp"] <= 0:
            winner = "blue (scripted)"
        else:
            winner = "draw (timeout)"
        print(
            f"Episode {episode + 1}: winner={winner}  "
            f"agent_hp={info['agent_hp']:.1f}  opponent_hp={info['opponent_hp']:.1f}"
        )

    env.close()


if __name__ == "__main__":
    main()
