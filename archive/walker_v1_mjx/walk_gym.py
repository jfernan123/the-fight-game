"""Gymnasium environment: train a single fighter (red) to walk in a commanded direction using
real leg torque, before any combat is involved.

This is a standalone locomotion skill, trained first and separately from fight_gym.py's combat
task. fight_gym.py currently "moves" the agent via a magic xfrc_applied force at every distance
(including inside punching range, which is what caused the drag-the-opponent exploit) -- this
env is the first step toward replacing that with an actual leg-driven gait. Merging the two is
future work, not done here.

The torso has real pitch/roll on top of the always-present x/y/z slides + hip_twist yaw (unlike
fight_env.py's combat model, which stays mechanically upright) -- it can actually fall, and doing
so ends the episode immediately with a fixed penalty (see FALL_ANGLE_DEG/FALL_HEIGHT/FALL_PENALTY
below) rather than just quietly degrading tracking_error for the rest of a long episode. So the
task is genuinely two things now: swing the legs against the ground to track a commanded
horizontal velocity (including turning via hip_twist, since hip/knee only flex in the sagittal
plane), AND stay upright doing it -- balance is no longer free.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for fight_env

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from fight_env import _build_model_xml, _actuators, _set_rest_pose_qpos

AGENT = "red"

WALK_JOINTS = ["hip_r", "hip_l", "knee_r", "knee_l", "hip_twist"]

MAX_SPEED = 0.6  # m/s -- matches the scripted position-actuator walk cycle's measured speed
# Widened from 150-300 (1.5-3s) to 150-1000 (1.5-10s) -- kept in sync with walk_gym_mjx.py, see
# that file's comment: a trained policy took a few real steps then froze after ~3s even when
# held on one command far longer than training ever asked of it, because it was almost never
# rewarded for sustaining a gait past ~3s during training.
COMMAND_RESAMPLE_MIN_STEPS = 150
COMMAND_RESAMPLE_MAX_STEPS = 1000
MAX_EPISODE_STEPS = 2000  # 20s at dt=0.01

ENERGY_COST_WEIGHT = 0.0005
TIME_PENALTY = 0.001

# The torso can now actually tip over (fight_env.py's free_torso=True adds pitch/roll hinges to
# the walker's model, unlike the fighter's combat model which stays mechanically upright) -- a
# fall ends the episode immediately with a fixed penalty instead of just quietly degrading
# tracking_error for the rest of a 20s episode, which was diluting the training signal for bad
# posture into something barely distinguishable from a slightly-off gait.
FALL_ANGLE_DEG = 50.0
FALL_HEIGHT = 0.5  # below this torso z the character is considered collapsed even if upright
FALL_PENALTY = 20.0


class WalkGymEnv(gym.Env):
    """Single-fighter Gymnasium env: red learns to track a commanded 2D horizontal velocity
    using its hip/knee/hip_twist torque motors. No opponent, no combat."""

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._viewer = None

        self.model = mujoco.MjModel.from_xml_string(
            _build_model_xml(_actuators, fighters=(AGENT,), free_torso=True)
        )
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.agent_id = self.model.body(AGENT).id
        self._agent_qvel = int(self.model.jnt_dofadr[self.model.body(AGENT).jntadr[0]])

        self._actuator_ids = [self.model.actuator(f"{AGENT}_{j}_m").id for j in WALK_JOINTS]
        self._joint_qpos_adr = [self.model.jnt_qposadr[self.model.joint(f"{AGENT}_{j}").id] for j in WALK_JOINTS]
        self._joint_dof_adr = [self.model.jnt_dofadr[self.model.joint(f"{AGENT}_{j}").id] for j in WALK_JOINTS]

        # Passive (unactuated) DOFs -- not part of the action space, but the policy needs to see
        # them to have any chance of learning to balance.
        self._pitch_qpos_adr = int(self.model.jnt_qposadr[self.model.joint(f"{AGENT}_pitch").id])
        self._pitch_dof_adr = int(self.model.jnt_dofadr[self.model.joint(f"{AGENT}_pitch").id])
        self._roll_qpos_adr = int(self.model.jnt_qposadr[self.model.joint(f"{AGENT}_roll").id])
        self._roll_dof_adr = int(self.model.jnt_dofadr[self.model.joint(f"{AGENT}_roll").id])

        self._command = np.zeros(2, dtype=np.float64)
        self._steps_until_resample = 0
        self._step_count = 0

        n_obs = len(self._build_observation())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(WALK_JOINTS),), dtype=np.float32)

    # ---- Gymnasium API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        _set_rest_pose_qpos(self.model, self.data, AGENT)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._resample_command()

        if self.render_mode == "human":
            self._render_frame()
        return self._build_observation(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for actuator_id, value in zip(self._actuator_ids, action):
            self.data.ctrl[actuator_id] = value

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        self._steps_until_resample -= 1
        if self._steps_until_resample <= 0:
            self._resample_command()

        fallen = self._has_fallen()
        reward = self._compute_reward(action, fallen)
        terminated = fallen
        truncated = self._step_count >= MAX_EPISODE_STEPS

        if self.render_mode == "human":
            self._render_frame()
        return self._build_observation(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ---- command handling (exposed for watch_walker_mjx.py's interactive keyboard control) ----

    def set_command(self, vx: float, vy: float) -> None:
        """Override the commanded velocity directly, bypassing random resampling -- used by
        watch_walker_mjx.py's key_callback to let a person steer the trained policy live."""
        self._command = np.array([vx, vy], dtype=np.float64)
        self._steps_until_resample = COMMAND_RESAMPLE_MAX_STEPS

    def _resample_command(self) -> None:
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        speed = self.np_random.uniform(0.0, MAX_SPEED)
        self._command = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64) * speed
        self._steps_until_resample = self.np_random.integers(
            COMMAND_RESAMPLE_MIN_STEPS, COMMAND_RESAMPLE_MAX_STEPS + 1
        )

    # ---- internals ----

    def _has_fallen(self) -> bool:
        pitch_deg = np.degrees(self.data.qpos[self._pitch_qpos_adr])
        roll_deg = np.degrees(self.data.qpos[self._roll_qpos_adr])
        torso_z = self.data.xpos[self.agent_id, 2]
        return abs(pitch_deg) > FALL_ANGLE_DEG or abs(roll_deg) > FALL_ANGLE_DEG or torso_z < FALL_HEIGHT

    def _compute_reward(self, action: np.ndarray, fallen: bool) -> float:
        actual_vel = self.data.qvel[self._agent_qvel:self._agent_qvel + 2]
        # Squared tracking error against the commanded velocity vector -- penalizes wrong
        # direction, wrong speed, and off-axis drift all in one term (maximized at 0 when
        # actual velocity exactly matches the command, including a near-zero command to stop).
        tracking_error = float(np.sum((actual_vel - self._command) ** 2))
        energy_cost = ENERGY_COST_WEIGHT * float(np.sum(action ** 2))
        reward = -tracking_error - energy_cost - TIME_PENALTY
        if fallen:
            reward -= FALL_PENALTY
        return reward

    def _build_observation(self) -> np.ndarray:
        joint_state = []
        for qpos_adr, dof_adr in zip(self._joint_qpos_adr, self._joint_dof_adr):
            joint_state.append(self.data.qpos[qpos_adr])
            joint_state.append(self.data.qvel[dof_adr])

        actual_vel = self.data.qvel[self._agent_qvel:self._agent_qvel + 2]
        torso_z = self.data.xpos[self.agent_id, 2]

        return np.concatenate([
            np.array(joint_state, dtype=np.float64),
            actual_vel,
            self._command,
            [torso_z],
            [self.data.qpos[self._pitch_qpos_adr], self.data.qvel[self._pitch_dof_adr]],
            [self.data.qpos[self._roll_qpos_adr], self.data.qvel[self._roll_dof_adr]],
        ]).astype(np.float32)

    def _info(self) -> dict:
        actual_vel = self.data.qvel[self._agent_qvel:self._agent_qvel + 2]
        return {
            "command": self._command.copy(),
            "velocity": np.array(actual_vel),
            "step": self._step_count,
        }

    def _render_frame(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.distance = 6
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = -18
        self._viewer.sync()
        time.sleep(self.dt)
