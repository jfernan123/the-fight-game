"""Single entry point for training the walker, on either backend. Run from the project root:

    python walker/train_walker.py --backend gpu                        # thousands of parallel envs via MJX (default)
    python walker/train_walker.py --backend gpu --timesteps 150000000 --entropy-cost 0.05
    python walker/train_walker.py --backend cpu --timesteps 2000000    # single-env SB3, runs directly, no WSL

GPU is the backend actually used for real training -- it dispatches into WSL2 and runs
train_walker_mjx.py there (JAX has no native Windows GPU support), forwarding every extra
argument through unchanged, so train_walker_mjx.py's own --help is the source of truth for GPU
options (--entropy-cost, --num-envs, --num-evals, etc). Checkpoints/logs land in
runs/walker/walk_checkpoints_mjx and runs/walker/tb_logs_walk_mjx.

CPU is a single-environment stable-baselines3 PPO trainer against the same walk_gym.py env,
kept as a fallback in case WSL/JAX ever breaks -- it's ~1000x fewer parallel environments than
the GPU path, so treat it as "does the reward/model still work at all", not real training.
Runs directly in the current (Windows) Python process -- needs the `fight-game` conda env, not
`fight-game-mjx`. Checkpoints/logs land in runs/walker/walk_checkpoints and
runs/walker/tb_logs_walk.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # this file lives in walker/, root is one level up
WSL_DISTRO = "Ubuntu"
WSL_CONDA_ENV = "fight-game-mjx"


def _wsl_project_root() -> str:
    drive = PROJECT_ROOT.drive.rstrip(":").lower()
    rest = PROJECT_ROOT.as_posix()[len(PROJECT_ROOT.drive):]
    return f"/mnt/{drive}{rest}"


def run_gpu(extra_args: list[str]) -> int:
    # cd's into the project root (not walker/) so train_walker_mjx.py's own relative default
    # checkpoint/log paths (runs/walker/...) resolve the same way no matter which entry point
    # launched it.
    forwarded = " ".join(shlex.quote(a) for a in extra_args)
    remote_cmd = (
        f"export PATH=$HOME/miniconda3/bin:$PATH; cd '{_wsl_project_root()}' && "
        f"conda run --no-capture-output -n {WSL_CONDA_ENV} python walker/train_walker_mjx.py {forwarded}"
    )
    return subprocess.call(["wsl", "-d", WSL_DISTRO, "-e", "bash", "-c", remote_cmd])


def run_cpu(extra_args: list[str]) -> int:
    import os

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from walk_gym import WalkGymEnv

    parser = argparse.ArgumentParser(description="Train a walking policy with PPO (single-env CPU fallback).")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "runs" / "walker" / "walk_checkpoints"))
    parser.add_argument("--tensorboard-log", default=str(PROJECT_ROOT / "runs" / "walker" / "tb_logs_walk"))
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .zip to resume from.")
    args = parser.parse_args(extra_args)

    env = Monitor(WalkGymEnv())
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=args.tensorboard_log, device="cpu")
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log, device="cpu",
            n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.99,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq, save_path=args.checkpoint_dir, name_prefix="walker",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=args.resume is None,
    )

    final_path = os.path.join(args.checkpoint_dir, "walker_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")
    return 0


def main() -> int:
    # Deliberately not argparse for this top layer: argparse auto-adds -h/--help, which would
    # intercept and print *this* parser's help instead of forwarding --help down to run_cpu's own
    # parser (or through to train_walker_mjx.py's) -- confirmed this bites in practice: `--backend
    # cpu --help` printed only the 3-line dispatcher usage, not the actual CPU training options.
    argv = list(sys.argv[1:])
    backend = "gpu"
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            backend = argv.pop(i + 1)
            argv.pop(i)
            break
        if arg.startswith("--backend="):
            backend = argv.pop(i).split("=", 1)[1]
            break

    if backend not in ("cpu", "gpu"):
        print(f"--backend must be 'cpu' or 'gpu', got {backend!r}", file=sys.stderr)
        return 2

    return run_gpu(argv) if backend == "gpu" else run_cpu(argv)


if __name__ == "__main__":
    sys.exit(main())
