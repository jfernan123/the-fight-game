"""Train the walker with PPO across thousands of parallel GPU-batched environments (walk_gym_mjx.py).

Normally launched via `python train_walker.py --backend gpu ...` from Windows, which dispatches
into WSL2 and forwards args here. Can also be run directly if you're already inside WSL2, using
the `fight-game-mjx` conda env there -- JAX has no GPU support on native Windows, so this can't
run from the same Windows Python everything else in this project uses.

    conda run -n fight-game-mjx python walker/train_walker_mjx.py --timesteps 20000000
    conda run -n fight-game-mjx python walker/train_walker_mjx.py --num-envs 2048

Progress: tensorboard --logdir runs/walker/tb_logs_walk_mjx   (run from Windows -- same disk, /mnt/c mount)

Benchmarked on this machine's RTX 3070 Ti (8GB): 256 envs ~4.7k env-steps/sec, 1024 envs ~12.7k,
2048 envs ~16.5k (vs. ~1.7-2k env-steps/sec for the single-env CPU fallback, `train_walker.py
--backend cpu`). 1024 is the default -- 2048 works but pushes this card's 8GB close to its limit
(CUDA OOM retry warnings observed, not fatal, but real memory pressure).
"""

import os
from pathlib import Path

# Must be set before jax initializes the GPU backend. This card only has 8GB and other WSL/GUI
# processes already hold some of it -- JAX's default (near-100%) preallocation triggers CUDA
# OOM retries at num_envs=1024+ without this (observed directly while benchmarking).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import argparse
import time

from tensorboardX import SummaryWriter

from walk_gym_mjx import WalkGymMjxEnv

# Anchored to the project root (this file lives in walker/) rather than left as a plain relative
# string, so the default lands in the same place regardless of the caller's CWD -- whether
# launched via train_walker.py's WSL dispatch (which cd's to the root) or run directly from
# inside walker/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHECKPOINT_DIR = str(_PROJECT_ROOT / "runs" / "walker" / "walk_checkpoints_mjx")
_DEFAULT_TENSORBOARD_LOG = str(_PROJECT_ROOT / "runs" / "walker" / "tb_logs_walk_mjx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the walker with PPO on thousands of parallel GPU envs.")
    # 20M steps only took ~14-15 min wall-clock at 1024 envs (training/walltime ended at 858s in
    # the last run) -- plenty of headroom, so bumped 3x. num_evals scaled up to match, keeping
    # checkpoint/log density at roughly one every 500k steps instead of getting coarser.
    parser.add_argument("--timesteps", type=int, default=60_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--episode-length", type=int, default=2000)  # matches walk_gym.py's MAX_EPISODE_STEPS
    parser.add_argument("--checkpoint-dir", default=_DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tensorboard-log", default=_DEFAULT_TENSORBOARD_LOG)
    parser.add_argument("--num-evals", type=int, default=120, help="how many times to log/checkpoint during training")
    parser.add_argument(
        "--entropy-cost", type=float, default=5e-2,
        help="weight on the entropy bonus (higher = more exploration, less premature convergence). "
             "1e-2 (brax's own default is 1e-4) collapsed hard: policy_dist_min_std dropped to ~0.05 "
             "by step 2.6M and reward plateaued at -125 to -157. 3e-2 was better (min_std ~0.08, "
             "reward plateaued higher at -81 to -95) but watch_walker_mjx.py showed the trained "
             "policy still only takes a few real steps then freezes into a static lean -- it never "
             "learned to sustain a repeating gait. 5e-2 plus the widened command-hold window in "
             "walk_gym_mjx.py (COMMAND_RESAMPLE_MAX_STEPS) are meant to push it further; still not "
             "a fully tuned value.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from brax.training.agents.ppo import train as ppo_train  # deferred: needs the env var set first

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)  # orbax requires an absolute path
    writer = SummaryWriter(args.tensorboard_log)
    start_time = time.time()

    def progress_fn(num_steps: int, metrics: dict) -> None:
        for key, value in metrics.items():
            writer.add_scalar(key, float(value), num_steps)
        writer.flush()
        elapsed = time.time() - start_time
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(f"step {num_steps:>12,d}  eval/episode_reward={reward: .3f}  elapsed={elapsed:.0f}s")

    env = WalkGymMjxEnv()
    ppo_train.train(
        environment=env,
        num_timesteps=args.timesteps,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
        action_repeat=1,
        learning_rate=3e-4,
        entropy_cost=args.entropy_cost,
        discounting=0.97,
        unroll_length=20,
        batch_size=args.num_envs // 4,
        num_minibatches=8,
        num_updates_per_batch=4,
        normalize_observations=True,
        num_evals=args.num_evals,
        seed=args.seed,
        progress_fn=progress_fn,
        save_checkpoint_path=checkpoint_dir,
    )

    writer.close()
    print(f"Training complete. Checkpoints saved under {checkpoint_dir}/")


if __name__ == "__main__":
    main()
