"""Watch a checkpoint trained by train_walker_mjx.py, steering it live with the keyboard.

Run this INSIDE WSL2 (fight-game-mjx conda env) -- it needs jax to load the checkpoint. Physics
playback itself runs on plain CPU MuJoCo (walk_gym.WalkGymEnv, the same env train_walker.py's
--backend cpu path uses directly) via WSLg's GUI forwarding -- only the trained policy's forward
pass is JAX, and it's forced onto CPU below so watching a checkpoint mid-training doesn't compete
with train_walker_mjx.py for the GPU (a single tiny-network inference per frame is faster on CPU
anyway -- dispatch overhead dominates at this scale, same lesson as the fighter's earlier
GPU-vs-CPU benchmark).

    conda run -n fight-game-mjx python walker/watch_walker_mjx.py                    # newest checkpoint
    conda run -n fight-game-mjx python walker/watch_walker_mjx.py runs/walker/walk_checkpoints_mjx/12345678

Controls: W/S/A/D set a commanded direction, X stops.
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # must be set before `import jax`

import argparse
import glob
import time
from pathlib import Path

_DEFAULT_CHECKPOINT_DIR = str(Path(__file__).resolve().parent.parent / "runs" / "walker" / "walk_checkpoints_mjx")

import jax
import jax.numpy as jnp
import mujoco.viewer
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from walk_gym import MAX_SPEED, WalkGymEnv

# brax 0.14.2's checkpoint.load_policy()/load_config() is broken for this env on two separate
# counts (confirmed by reading its source, not guessed): (1) it crashes with KeyError(None) on
# any kernel-init kwarg left at its library default (saved as null, which load_config treats as
# a literal dict key instead of "no override"), and (2) even past that, the saved
# observation_size round-trips as a stringified ShapeDtypeStruct instead of a plain int, which
# then fails inside make_ppo_networks. Rather than patch two independent upstream bugs, this
# loads the raw trained params directly (checkpoint.load, which never touches the broken config
# file) and reconstructs the network from values we already know are correct because we chose
# them in train_walker_mjx.py -- observation_size/action_size come straight from WalkGymEnv, and
# every other kwarg here is make_ppo_networks' own default, matching what an unconfigured
# ppo_train.train() call already uses.


def _latest_checkpoint(checkpoint_dir: str) -> str:
    candidates = [p for p in glob.glob(os.path.join(checkpoint_dir, "*")) if os.path.isdir(p)]
    if not candidates:
        raise SystemExit(f"No checkpoint directories found in '{checkpoint_dir}'.")
    return max(candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an MJX-trained walking checkpoint, steered by keyboard.")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to a saved checkpoint step directory (e.g. runs/walker/walk_checkpoints_mjx/12345678). "
             "Omit to auto-pick the most recently modified one in --checkpoint-dir.",
    )
    parser.add_argument("--checkpoint-dir", default=_DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint or _latest_checkpoint(args.checkpoint_dir))
    print(f"Loading {checkpoint}")
    print("Controls: W=forward  S=back  A=left  D=right  X=stop")

    env = WalkGymEnv(render_mode=None)
    obs, info = env.reset()

    params = ppo_checkpoint.load(checkpoint)
    net = ppo_networks.make_ppo_networks(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        preprocess_observations_fn=running_statistics.normalize,  # train_walker_mjx.py uses normalize_observations=True
    )
    inference_fn = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    rng = jax.random.PRNGKey(0)  # unused: deterministic=True ignores the sampling key

    command = {"vx": 0.0, "vy": 0.0}

    def key_callback(keycode: int) -> None:
        if keycode == ord("W"):
            command["vx"], command["vy"] = MAX_SPEED, 0.0
        elif keycode == ord("S"):
            command["vx"], command["vy"] = -MAX_SPEED, 0.0
        elif keycode == ord("A"):
            command["vx"], command["vy"] = 0.0, MAX_SPEED
        elif keycode == ord("D"):
            command["vx"], command["vy"] = 0.0, -MAX_SPEED
        elif keycode == ord("X"):
            command["vx"], command["vy"] = 0.0, 0.0

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18
        while viewer.is_running():
            env.set_command(command["vx"], command["vy"])
            action, _ = inference_fn(jnp.array(obs), rng)
            obs, reward, terminated, truncated, info = env.step(jax.device_get(action))
            if terminated or truncated:
                obs, info = env.reset()
            viewer.sync()
            time.sleep(env.dt)


if __name__ == "__main__":
    main()
