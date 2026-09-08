"""GPU-batched version of walk_gym.py, built on MuJoCo MJX + Brax's PipelineEnv.

Same task as walk_gym.py (track a commanded 2D horizontal velocity using hip/knee/hip_twist
torque), same reward, same model -- the only thing that changes is the physics backend: MJX
runs entirely on GPU and vmaps over thousands of environments at once, which is the actual lever
behind "learning to walk in minutes" (Isaac Gym / Rudin et al.) rather than a smarter algorithm.

This only runs where JAX has GPU support, which on this machine means inside WSL2, not native
Windows Python -- run it with the `fight-game-mjx` conda env there, not the Windows `fight-game`
env used by everything else in this project.

Reset/step are written in JAX's functional style (no Python-level `if`/`for` branching on traced
values) because Brax jits and vmaps the whole step function across the batch of environments --
that's what `jax.lax`/`jnp.where` are doing below in place of walk_gym.py's plain Python control
flow.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for fight_env

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from fight_env import STAND_HEIGHT, _actuators, _build_model_xml, _set_rest_pose_qpos

AGENT = "red"
WALK_JOINTS = ["hip_r", "hip_l", "knee_r", "knee_l", "hip_twist"]

MAX_SPEED = 0.6  # m/s -- same cap as walk_gym.py
# Widened from 150-300 (1.5-3s) to 150-1000 (1.5-10s): diagnosed via watch_walker_mjx.py that a
# trained policy took a few real steps then froze into a static lean after ~3s even when held on
# a single command far longer than it ever saw in training -- it had learned one burst of motion,
# never repeated cycling, because it was almost never rewarded for SUSTAINING a gait past ~3s
# during training. Widening the hold so it sometimes has to walk for many seconds straight is
# meant to force that repeated-cycle behavior to actually get reinforced.
COMMAND_RESAMPLE_MIN_STEPS = 150
COMMAND_RESAMPLE_MAX_STEPS = 1000

ENERGY_COST_WEIGHT = 0.0005
TIME_PENALTY = 0.001

# Mirrors walk_gym.py's FALL_ANGLE_DEG/FALL_HEIGHT/FALL_PENALTY -- kept as a separate duplicated
# constant (like MAX_SPEED/COMMAND_RESAMPLE_* above) rather than importing from walk_gym.py, since
# that file pulls in gymnasium and this one has to stay runnable standalone under the WSL
# fight-game-mjx env. Keep the values in sync by hand.
FALL_ANGLE_DEG = 50.0
FALL_ANGLE_RAD = math.radians(FALL_ANGLE_DEG)
FALL_HEIGHT = 0.5
FALL_PENALTY = 20.0

# Built once at import time with plain (CPU) mujoco, purely to look up joint/actuator addresses
# by name and to compute the constant rest-pose qpos -- never stepped, just introspected.
_MJ_MODEL = mujoco.MjModel.from_xml_string(
    _build_model_xml(_actuators, fighters=(AGENT,), free_torso=True)
)
_MJ_DATA = mujoco.MjData(_MJ_MODEL)
mujoco.mj_resetData(_MJ_MODEL, _MJ_DATA)
_set_rest_pose_qpos(_MJ_MODEL, _MJ_DATA, AGENT)
mujoco.mj_forward(_MJ_MODEL, _MJ_DATA)

INIT_QPOS = jnp.array(_MJ_DATA.qpos.copy())

_ACT_IDX = jnp.array([_MJ_MODEL.actuator(f"{AGENT}_{j}_m").id for j in WALK_JOINTS])
_JOINT_QPOS_ADR = [int(_MJ_MODEL.jnt_qposadr[_MJ_MODEL.joint(f"{AGENT}_{j}").id]) for j in WALK_JOINTS]
_JOINT_DOF_ADR = [int(_MJ_MODEL.jnt_dofadr[_MJ_MODEL.joint(f"{AGENT}_{j}").id]) for j in WALK_JOINTS]
_VEL_X_ADR = int(_MJ_MODEL.jnt_dofadr[_MJ_MODEL.joint(f"{AGENT}_x").id])  # vy is the next dof
_Z_QPOS_ADR = int(_MJ_MODEL.jnt_qposadr[_MJ_MODEL.joint(f"{AGENT}_z").id])
_PITCH_QPOS_ADR = int(_MJ_MODEL.jnt_qposadr[_MJ_MODEL.joint(f"{AGENT}_pitch").id])
_PITCH_DOF_ADR = int(_MJ_MODEL.jnt_dofadr[_MJ_MODEL.joint(f"{AGENT}_pitch").id])
_ROLL_QPOS_ADR = int(_MJ_MODEL.jnt_qposadr[_MJ_MODEL.joint(f"{AGENT}_roll").id])
_ROLL_DOF_ADR = int(_MJ_MODEL.jnt_dofadr[_MJ_MODEL.joint(f"{AGENT}_roll").id])


def _sample_command(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    key_angle, key_speed = jax.random.split(rng)
    angle = jax.random.uniform(key_angle, (), minval=0.0, maxval=2.0 * jnp.pi)
    speed = jax.random.uniform(key_speed, (), minval=0.0, maxval=MAX_SPEED)
    return jnp.array([jnp.cos(angle), jnp.sin(angle)]) * speed


def _check_fallen(pitch: jax.Array, roll: jax.Array, torso_z: jax.Array) -> jax.Array:
    """Pure function, pulled out of step() so the fall-termination thresholds can be unit-tested
    directly with plain scalars instead of only by driving real MJX physics into a specific
    state. Forcing a specific configuration via state.replace(pipeline_state=pipeline_state.replace(
    q=...)) before calling step() does NOT actually perturb the physics (confirmed empirically) --
    pipeline_state's .q/.qd are read-oriented aliases over mjx's internal _impl structure (same
    thing the "Accessing contact directly from Data is deprecated" warning is about elsewhere),
    not plain writable dataclass fields wired back into the physics stepper."""
    return (jnp.abs(pitch) > FALL_ANGLE_RAD) | (jnp.abs(roll) > FALL_ANGLE_RAD) | (torso_z < FALL_HEIGHT)


def _compute_reward(actual_vel: jax.Array, command: jax.Array, action: jax.Array, fallen: jax.Array) -> jax.Array:
    """Pure reward function, identical formula to walk_gym.WalkGymEnv._compute_reward -- pulled
    out of step() so it can be unit-tested directly instead of only through a full physics step."""
    tracking_error = jnp.sum((actual_vel - command) ** 2)
    energy_cost = ENERGY_COST_WEIGHT * jnp.sum(action ** 2)
    reward = -tracking_error - energy_cost - TIME_PENALTY
    return reward - jnp.where(fallen, FALL_PENALTY, 0.0)


class WalkGymMjxEnv(PipelineEnv):
    """Brax PipelineEnv wrapping the same humanoid model as walk_gym.WalkGymEnv, batched on GPU."""

    def __init__(self, **kwargs):
        sys = mjcf.load_model(_MJ_MODEL)
        kwargs["backend"] = "mjx"
        kwargs.setdefault("n_frames", 1)  # matches walk_gym.py: one RL step == one 0.01s physics step
        super().__init__(sys, **kwargs)

    @property
    def action_size(self) -> int:
        return len(WALK_JOINTS)

    def reset(self, rng: jax.Array) -> State:
        rng, key_cmd, key_dur = jax.random.split(rng, 3)
        pipeline_state = self.pipeline_init(INIT_QPOS, jnp.zeros(self.sys.qd_size()))
        command = _sample_command(key_cmd)
        steps_until_resample = jax.random.randint(
            key_dur, (), COMMAND_RESAMPLE_MIN_STEPS, COMMAND_RESAMPLE_MAX_STEPS + 1
        )
        obs = self._get_obs(pipeline_state, command)
        info = {"command": command, "steps_until_resample": steps_until_resample, "rng": rng}
        return State(pipeline_state, obs, jnp.float32(0.0), jnp.float32(0.0), {}, info)

    def step(self, state: State, action: jax.Array) -> State:
        full_ctrl = jnp.zeros(self.sys.act_size()).at[_ACT_IDX].set(jnp.clip(action, -1.0, 1.0))
        pipeline_state = self.pipeline_step(state.pipeline_state, full_ctrl)

        rng, key_cmd, key_dur = jax.random.split(state.info["rng"], 3)
        steps_left = state.info["steps_until_resample"] - 1
        do_resample = steps_left <= 0
        resampled_command = _sample_command(key_cmd)
        resampled_duration = jax.random.randint(
            key_dur, (), COMMAND_RESAMPLE_MIN_STEPS, COMMAND_RESAMPLE_MAX_STEPS + 1
        )
        command = jnp.where(do_resample, resampled_command, state.info["command"])
        steps_until_resample = jnp.where(do_resample, resampled_duration, steps_left)

        actual_vel = jax.lax.dynamic_slice(pipeline_state.qd, (_VEL_X_ADR,), (2,))
        pitch = pipeline_state.q[_PITCH_QPOS_ADR]
        roll = pipeline_state.q[_ROLL_QPOS_ADR]
        torso_z = STAND_HEIGHT + pipeline_state.q[_Z_QPOS_ADR]
        fallen = _check_fallen(pitch, roll, torso_z)
        reward = _compute_reward(actual_vel, command, action, fallen)

        obs = self._get_obs(pipeline_state, command)
        info = dict(state.info, command=command, steps_until_resample=steps_until_resample, rng=rng)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=fallen.astype(jnp.float32), info=info
        )

    def _get_obs(self, pipeline_state, command: jax.Array) -> jax.Array:
        joint_state = []
        for qpos_adr, dof_adr in zip(_JOINT_QPOS_ADR, _JOINT_DOF_ADR):
            joint_state.append(pipeline_state.q[qpos_adr])
            joint_state.append(pipeline_state.qd[dof_adr])
        actual_vel = jax.lax.dynamic_slice(pipeline_state.qd, (_VEL_X_ADR,), (2,))
        torso_z = STAND_HEIGHT + pipeline_state.q[_Z_QPOS_ADR]
        pitch_state = jnp.array([pipeline_state.q[_PITCH_QPOS_ADR], pipeline_state.qd[_PITCH_DOF_ADR]])
        roll_state = jnp.array([pipeline_state.q[_ROLL_QPOS_ADR], pipeline_state.qd[_ROLL_DOF_ADR]])
        return jnp.concatenate([
            jnp.array(joint_state), actual_vel, command, jnp.array([torso_z]), pitch_state, roll_state
        ])
