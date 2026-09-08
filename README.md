# The Fight Game

A single humanoid learning to walk with PPO, on either CPU (for quick iteration/testing) or GPU
(for real training) -- and four variants of the body's ankle, trained and compared side by side.

- `fight_env.py` -- the humanoid body model: torso, arms, legs, waist, real human range-of-motion
  joint limits, empirically gear-tuned torque actuators. Also has its own scripted two-fighter
  demo and an interactive `--explore` viewer -- this file predates the walker and is shared
  infrastructure, not walker-specific. `ankle_mode` (`None`/`"single"`/`"dual"`/`"detailed"`) adds
  a real ankle joint between shin and foot, which the base body never had -- see its
  `_humanoid_body` docstring for what each mode actually changes.
- `walk_env.py` -- the CPU walking task: a `WalkEnv` Gymnasium env built on that body. Only the
  legs + hip_twist (+ ankle, if `ankle_mode` is set) are RL-controlled. Reward: forward-speed
  tracking toward a target pace, minus effort/wobble/jerkiness penalties, plus a feet air-time
  shaping term, minus a fall penalty -- see the module for the exact terms (adapted from "Learning
  to Walk in Minutes", arXiv:2109.11978, Appendix A.3). The torso *can* fall by default (real
  pitch/roll, no mechanical autobalance) -- pass `WalkEnv(free_torso=False)` to lock it upright
  instead, only meant for manually poking at joints (see `--explore` below), not training.
- `walk_env_gpu.py` -- the same task/reward/body, GPU-batched via **MuJoCo Warp** + PyTorch
  (`WalkEnvGPU`) -- thousands of parallel copies stepped together in one process, natively on
  Windows (no WSL2, no JAX). Measured on this machine's RTX 3070 Ti: ~40-66k env-steps/sec
  depending on ankle_mode, vs ~1,800 for the single-env CPU version.
- `train.py` / `watch.py` -- CPU: train one variant with `stable-baselines3` PPO, watch a
  checkpoint run. Pass `--ankle {single,dual,detailed}` to either (omit for no ankle).
- `train_all.py` / `watch_all.py` -- CPU: train all four ankle variants at once (separate
  processes, genuinely parallel across CPU cores) into `runs/<mode>/` + `tb_logs/<mode>/`; watch
  them together in one shared scene, each in its own lane, with 1/2/3/4 to pause/resume one at a
  time. Filter with `--baseline`/`--single`/`--dual`/`--detailed`.
- `train_gpu.py` -- GPU: train one variant with a from-scratch PyTorch PPO loop (CleanRL-style --
  `stable-baselines3` doesn't speak batched-tensor GPU envs) against `walk_env_gpu.py`. Same
  `--ankle` flag as `train.py`.
- `train_gpu_all.py` -- GPU: train all four variants, but **sequentially**, not in parallel like
  the CPU version -- they'd otherwise all fight over the same one GPU. Into `runs_gpu/<mode>/` +
  `tb_logs_gpu/<mode>/`.

Older, more complex versions of this project (a two-fighter combat task, an earlier GPU-parallel
attempt built on JAX/MJX/Brax + WSL2, a command-tracking curriculum) are kept in `archive/` for
reference, not maintained -- the JAX/WSL2 path was abandoned specifically in favor of MuJoCo Warp,
which runs natively on Windows with far less friction.

## Setup

```powershell
conda create -n fight-game python=3.11
conda activate fight-game
python -m pip install -r requirements.txt
python -m pip install warp-lang mujoco-warp   # GPU path only -- needs CUDA 12.4+, no WSL2 needed
```

## Run

```powershell
python fight_env.py                          # scripted two-fighter demo, no RL
python fight_env.py --explore                 # interactive viewer, drag joint sliders by hand

python walk_env.py --explore                  # walker's own joints, autobalance ON (torso locked upright)
python walk_env.py --explore --ankle dual --no-autobalance  # ...with an ankle, and real falling

# CPU: one variant, or all four
python train.py --timesteps 2000000 --ankle dual
python watch.py
python train_all.py
python watch_all.py

# GPU: one variant, or all four (sequential)
python train_gpu.py --timesteps 20000000 --ankle dual
python train_gpu_all.py
```

CPU checkpoints/logs: `runs/`, `tb_logs/` (single-variant) or `runs/<mode>/`, `tb_logs/<mode>/`
(via `train_all.py`). GPU: `runs_gpu/`, `tb_logs_gpu/` the same way. `tensorboard --logdir
tb_logs` (or `tb_logs_gpu`) picks up everything underneath for side-by-side comparison.
