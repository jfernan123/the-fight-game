"""Watch multiple ankle-mode checkpoints at once, all in ONE shared scene/window -- each walking
forward in its own parallel lane (offset sideways) so they don't collide, all stepped together in
lockstep and each driven by its own trained checkpoint. Works with checkpoints from EITHER
backend (train_all.py's *.zip via stable-baselines3, or train_gpu_all.py's *.pt raw PyTorch) --
auto-detected per agent by file extension, so mixing (e.g. "single" trained on GPU, "dual" trained
on CPU) just works without needing to know or care which produced which.

    python watch_all.py                       # all four -- searches runs/<mode>/ and runs_gpu/<mode>/,
                                               # newest checkpoint (by mtime) wins per mode
    python watch_all.py --single --dual        # just these two
    python watch_all.py --runs-dir runs_gpu    # only look in one specific root, not both

Press 1/2/3/4 (in the order the agents were launched, printed at startup) to toggle an agent --
paused agents go limp (zero control, held up only by their passive joint springs) rather than
disappearing, so you can silence the ones you don't want and focus on one at a time.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from fight_env import STAND_HEIGHT, _actuators, _humanoid_body, _set_rest_pose_qpos
from walk_env import BASE_WALK_JOINTS, FALL_ANGLE_DEG, FALL_HEIGHT, _ankle_joint_names

MODES = {"baseline": None, "single": "single", "dual": "dual", "detailed": "detailed"}
# Sideways lane offset (meters) so agents walking forward don't collide with each other.
LANE_Y = {"baseline": -3.0, "single": -1.0, "dual": 1.0, "detailed": 3.0}
LANE_COLOR = {
    "baseline": ("0.85 0.16 0.12 1", "0.98 0.45 0.18 1", "1 0.82 0.2 1"),
    "single": ("0.10 0.35 0.88 1", "0.25 0.75 1 1", "1 0.82 0.2 1"),
    "dual": ("0.15 0.75 0.30 1", "0.55 0.95 0.55 1", "1 0.82 0.2 1"),
    "detailed": ("0.75 0.55 0.10 1", "0.98 0.85 0.35 1", "1 0.82 0.2 1"),
}


def _find_checkpoint(mode: str, search_dirs: list[str]) -> str | None:
    """Looks for a checkpoint for `mode` under each of search_dirs/<mode>/ -- both *.zip (CPU,
    stable-baselines3) and *.pt (GPU, train_gpu.py's raw PyTorch) -- and returns the newest one
    by mtime across all of them, or None if nothing was found anywhere. Searches recursively since
    train.py/train_gpu.py now nest checkpoints under a per-run timestamp folder (<mode>/<run-id>/)
    -- "**" also matches zero intermediate dirs, so this still finds checkpoints saved directly in
    <mode>/ by an older, un-timestamped run."""
    candidates = []
    for root in search_dirs:
        checkpoint_dir = os.path.join(root, mode)
        candidates += glob.glob(os.path.join(checkpoint_dir, "**", "*.zip"), recursive=True)
        candidates += glob.glob(os.path.join(checkpoint_dir, "**", "*.pt"), recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


class _Policy:
    """Normalizes a stable-baselines3 (.zip, CPU) or raw-PyTorch (.pt, train_gpu.py's GPU
    ActorCritic) checkpoint into one common .predict(obs) -> action interface, so the rest of
    this script doesn't need to know or care which backend produced a given checkpoint."""

    def __init__(self, checkpoint_path: str, obs_dim: int, action_dim: int) -> None:
        if checkpoint_path.endswith(".pt"):
            from train_gpu import ActorCritic

            self._net = ActorCritic(obs_dim, action_dim)
            self._net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            self._net.eval()
            self._backend = "gpu"
        else:
            from stable_baselines3 import PPO

            self._net = PPO.load(checkpoint_path)
            self._backend = "cpu"

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self._backend == "gpu":
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                return self._net.actor_mean(obs_t).squeeze(0).numpy()  # deterministic: the mean, not a sample
        action, _ = self._net.predict(obs, deterministic=True)
        return action


def _build_combined_model_xml(names: list[str]) -> str:
    bodies = "\n".join(
        _humanoid_body(name, *LANE_COLOR[name], free_torso=True, ankle_mode=MODES[name], y_offset=LANE_Y[name])
        for name in names
    )
    actuators = "\n".join(_actuators(name, ankle_mode=MODES[name]) for name in names)
    return f"""
<mujoco model="ankle_comparison">
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast" />
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" />
  </visual>
  <asset>
    <texture name="floor_texture" type="2d" builtin="checker" width="300" height="300" rgb1="0.12 0.14 0.18" rgb2="0.18 0.20 0.24" />
    <material name="floor_material" texture="floor_texture" texrepeat="8 8" />
  </asset>
  <worldbody>
    <light pos="0 0 7" dir="0 0 -1" directional="true" />
    <geom name="floor" type="plane" size="40 8 0.1" material="floor_material" />
    {bodies}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


class _Agent:
    """Per-agent bookkeeping against the ONE shared model/data -- same observation/fall logic as
    WalkEnv, just re-based to read/write this agent's own slice of the shared arrays instead of
    owning a whole model to itself. All comparison agents are watched under free_torso=True
    (real falling), matching what train.py always trains against."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, name: str, ankle_mode: str | None) -> None:
        self.name = name
        self.model = model
        self.data = data
        self.walk_joints = BASE_WALK_JOINTS + _ankle_joint_names(ankle_mode)

        self.agent_id = model.body(name).id
        self._x_dof = int(model.jnt_dofadr[model.body(name).jntadr[0]])
        self._actuator_ids = [model.actuator(f"{name}_{j}_m").id for j in self.walk_joints]
        self._joint_qpos_adr = [model.jnt_qposadr[model.joint(f"{name}_{j}").id] for j in self.walk_joints]
        self._joint_dof_adr = [model.jnt_dofadr[model.joint(f"{name}_{j}").id] for j in self.walk_joints]
        self._pitch_qpos_adr = int(model.jnt_qposadr[model.joint(f"{name}_pitch").id])
        self._pitch_dof_adr = int(model.jnt_dofadr[model.joint(f"{name}_pitch").id])
        self._roll_qpos_adr = int(model.jnt_qposadr[model.joint(f"{name}_roll").id])
        self._roll_dof_adr = int(model.jnt_dofadr[model.joint(f"{name}_roll").id])

    def observation(self) -> np.ndarray:
        joint_state = []
        for qpos_adr, dof_adr in zip(self._joint_qpos_adr, self._joint_dof_adr):
            joint_state.append(self.data.qpos[qpos_adr])
            joint_state.append(self.data.qvel[dof_adr])
        forward_vel = self.data.qvel[self._x_dof]
        torso_z = self.data.xpos[self.agent_id, 2]
        obs = [
            np.array(joint_state, dtype=np.float64),
            [forward_vel, torso_z],
            [self.data.qpos[self._pitch_qpos_adr], self.data.qvel[self._pitch_dof_adr]],
            [self.data.qpos[self._roll_qpos_adr], self.data.qvel[self._roll_dof_adr]],
        ]
        return np.concatenate(obs).astype(np.float32)

    def apply_action(self, action: np.ndarray) -> None:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for actuator_id, value in zip(self._actuator_ids, action):
            self.data.ctrl[actuator_id] = value

    def go_limp(self) -> None:
        for actuator_id in self._actuator_ids:
            self.data.ctrl[actuator_id] = 0.0

    def has_fallen(self) -> bool:
        pitch_deg = np.degrees(self.data.qpos[self._pitch_qpos_adr])
        roll_deg = np.degrees(self.data.qpos[self._roll_qpos_adr])
        torso_z = self.data.xpos[self.agent_id, 2]
        return abs(pitch_deg) > FALL_ANGLE_DEG or abs(roll_deg) > FALL_ANGLE_DEG or torso_z < FALL_HEIGHT

    def reset(self) -> None:
        """Resets only THIS agent's own DOFs -- mj_resetData would reset every agent sharing the
        model, so this hand-zeroes just this one's slice instead."""
        for joint_suffix in ("_x", "_y", "_z", "_hip_twist", "_pitch", "_roll"):
            qpos_adr = self.model.jnt_qposadr[self.model.joint(f"{self.name}{joint_suffix}").id]
            dof_adr = self.model.jnt_dofadr[self.model.joint(f"{self.name}{joint_suffix}").id]
            self.data.qpos[qpos_adr] = 0.0
            self.data.qvel[dof_adr] = 0.0
        for qpos_adr, dof_adr in zip(self._joint_qpos_adr, self._joint_dof_adr):
            self.data.qvel[dof_adr] = 0.0
        _set_rest_pose_qpos(self.model, self.data, self.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch multiple ankle-mode checkpoints together in one scene.")
    parser.add_argument("--baseline", action="store_true", help="no ankle (original rigid foot)")
    parser.add_argument("--single", action="store_true", help="single-hinge ankle")
    parser.add_argument("--dual", action="store_true", help="dual-hinge (pitch+roll) ankle")
    parser.add_argument("--detailed", action="store_true", help="dual-hinge ankle + split heel/toe foot")
    parser.add_argument(
        "--runs-dir", action="append",
        help="root(s) to search for <root>/<mode>/*.zip or *.pt -- defaults to both runs/ "
             "(train_all.py, CPU) and runs_gpu/ (train_gpu_all.py, GPU); pass this (repeatably) "
             "to search only specific root(s) instead.",
    )
    args = parser.parse_args()
    search_dirs = args.runs_dir or ["runs", "runs_gpu"]

    selected = [name for name in ("baseline", "single", "dual", "detailed") if getattr(args, name)]
    if not selected:
        selected = list(MODES.keys())  # nothing specified -> all four

    checkpoints = {}
    names = []
    for name in selected:
        checkpoint = _find_checkpoint(name, search_dirs)
        if checkpoint is None:
            print(f"[{name}] skipping -- no checkpoint found under {[os.path.join(d, name) for d in search_dirs]}")
            continue
        checkpoints[name] = checkpoint
        names.append(name)
    if not names:
        raise SystemExit("No checkpoints found for any selected mode.")

    model = mujoco.MjModel.from_xml_string(_build_combined_model_xml(names))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    agents = {name: _Agent(model, data, name, MODES[name]) for name in names}
    policies = {}
    for name in names:
        print(f"[{name}] loading {checkpoints[name]}")
        policies[name] = _Policy(checkpoints[name], agents[name].observation().shape[0], len(agents[name].walk_joints))
        agents[name].reset()
    mujoco.mj_forward(model, data)

    active = {name: True for name in names}

    def key_callback(keycode: int) -> None:
        idx = keycode - ord("1")
        if 0 <= idx < len(names):
            name = names[idx]
            active[name] = not active[name]
            if not active[name]:
                agents[name].go_limp()
            print(f"[{name}] {'resumed' if active[name] else 'paused (went limp)'}")

    print("\nPress 1/2/3/4 to pause/resume: " + ", ".join(f"{i + 1}={n}" for i, n in enumerate(names)))

    obs = {name: agents[name].observation() for name in names}
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 10
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        while viewer.is_running():
            for name in names:
                if active[name]:
                    action = policies[name].predict(obs[name])
                    agents[name].apply_action(action)

            mujoco.mj_step(model, data)

            for name in names:
                if active[name]:
                    if agents[name].has_fallen():
                        agents[name].reset()
                        mujoco.mj_forward(model, data)
                    obs[name] = agents[name].observation()

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
