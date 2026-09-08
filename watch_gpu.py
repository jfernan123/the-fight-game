"""Watch a GPU-trained (train_gpu.py) checkpoint run. Rendering itself uses the plain CPU
WalkEnv (walk_env.py) -- a single visible episode doesn't need GPU batching, only the policy
network's forward pass comes from the GPU-trained checkpoint (and that's cheap enough to run on
CPU for one env at a time, same lesson as watch_walker_mjx.py in archive/walker_v1_mjx/).

    python watch_gpu.py                                   # newest checkpoint in runs_gpu/
    python watch_gpu.py runs_gpu/walker_final.pt --ankle single
    python watch_gpu.py runs_gpu/single/walker_final.pt --ankle single   # train_gpu_all.py's layout

--ankle MUST match whatever the checkpoint was actually trained with -- the network's input/output
sizes are baked into its saved weights, so a mismatch fails to load rather than silently misbehaving.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from train_gpu import ActorCritic
from walk_env import WalkEnv


def _latest_checkpoint(checkpoint_dir: str) -> str:
    # Recursive since train_gpu.py now nests checkpoints under a per-run timestamp folder
    # (runs_gpu/<mode>/<run-id>/) -- "**" also matches zero intermediate dirs, so this still finds
    # checkpoints saved directly in checkpoint_dir by an older, un-timestamped run.
    candidates = glob.glob(os.path.join(checkpoint_dir, "**", "*.pt"), recursive=True)
    if not candidates:
        raise SystemExit(f"No checkpoint .pt files found under '{checkpoint_dir}'.")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a GPU-trained (train_gpu.py) checkpoint run.")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to a saved .pt checkpoint. Omit to auto-pick the most recently modified one "
             "in --checkpoint-dir -- safe to run mid-training to see progress.",
    )
    parser.add_argument("--checkpoint-dir", default="runs_gpu")
    parser.add_argument(
        "--ankle", choices=["single", "dual", "detailed"], default=None,
        help="must match what this checkpoint was trained with (omit for no-ankle).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    print(f"Loading {checkpoint}")

    env = WalkEnv(render_mode="human", ankle_mode=args.ankle)
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    agent.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent.eval()

    for episode in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.actor_mean(obs_t).squeeze(0).numpy()  # deterministic: the mean, not a sampled action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: total_reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
