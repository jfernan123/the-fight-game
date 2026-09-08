"""Walker: reuses the full humanoid body from fight_env.py (real human range of motion, every
joint's gear empirically tuned -- arms, waist, hip_twist, hip, knee) and actuates EVERY joint that
has a motor -- legs, hip_twist, waist, and arms (including shoulder abduction/adduction, added
specifically so the policy can use arm motion for balance the way a human would, not just the
forward-back shoulder swing that existed before). Nothing usable is left passively spring-held:
the only unactuated DOFs are the torso's own pitch/roll (the actual thing being tested -- giving
the policy direct torque on its own fall angle would defeat the point) and the free x/y/z root
translation (not a joint, nothing to motor).

The physics/body model is deliberately untouched (see fight_env.py). What's simplified here is
the task/training setup: no command-tracking curriculum, no GPU/MJX -- just a straightforward
forward-speed reward on top of the already-tuned body, single CPU env, single train/watch
script.

The torso can fall (real pitch/roll, nothing passively balances it) by default -- this is
"autobalance OFF", `WalkEnv(free_torso=True)`, and is what train.py/watch.py always use. Falling
ends the episode immediately with a fixed penalty instead of quietly dragging the reward down.
"Fallen" means either the torso's own pitch/roll/height crosses a threshold, OR any non-foot body
part (thigh, shin, torso, arms) is touching the ground -- a real trained checkpoint found a stable
crouching pose (both knees bent ~100 degrees, thigh resting on the floor) that stayed inside the
torso-only thresholds the whole time, so torso orientation alone isn't a sufficient definition of
"still standing." Rather than ending the episode the instant that's true, an exponential moving
average of the condition has to cross FALL_EMA_THRESHOLD -- a momentary stumble has time to
recover, but a body part that's genuinely down MOST of the time (even if contact keeps flickering
on and off, as it naturally does while dragging) still gets caught. Jumping (leaving the ground
entirely) isn't penalized -- if it turns out to be a useful strategy, it's allowed to be one.
Pass `free_torso=False` ("autobalance ON") to mechanically lock the torso upright (translation +
yaw only, can't tip) -- not for training, just convenient when you want to manually inspect how
individual joints move without the whole body toppling over mid-test (see --explore below).

    python walk_env.py --explore                   # interactive viewer, autobalance ON by default
    python walk_env.py --explore --no-autobalance   # ...with real falling, like training uses
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from fight_env import _actuators, _build_model_xml, _set_rest_pose_qpos
import reward_terms
from reward_terms import (
    COMMAND_MAX_SPEED, COMMAND_MIN_SPEED, COMMAND_RESAMPLE_INTERVAL, GAIT_CYCLE_TIME,
    NumpyState,
)

AGENT = "red"
# Every joint with a motor in fight_env.py's _actuators() except the ankle (a separate, optional
# variant -- see _ankle_joint_names below). shoulder_abduct_{side} is the arm's second axis (raising
# the arm out to the side, not just forward/back) -- see fight_env.py's SHOULDER_ABDUCT_MAX comment.
BASE_WALK_JOINTS = [
    "hip_r", "hip_l", "knee_r", "knee_l", "hip_twist",
    "shoulder_r", "shoulder_l", "shoulder_abduct_r", "shoulder_abduct_l", "elbow_r", "elbow_l",
    "waist_twist", "waist_bend",
]


def _ankle_joint_names(ankle_mode: str | None) -> list[str]:
    """See fight_env.py's _humanoid_body docstring for what each mode actually adds physically."""
    if ankle_mode is None:
        return []
    if ankle_mode == "single":
        return ["ankle_r", "ankle_l"]
    if ankle_mode in ("dual", "detailed"):
        # detailed reuses the same 2-DOF ankle as dual -- only the foot geometry differs (see
        # fight_env.py), not the joint/actuation set.
        return ["ankle_pitch_r", "ankle_pitch_l", "ankle_roll_r", "ankle_roll_l"]
    raise ValueError(f"unknown ankle_mode: {ankle_mode!r} (expected None, 'single', 'dual', or 'detailed')")

MAX_EPISODE_STEPS = 1000  # 10s at dt=0.01

# What counts as "fallen". These live here rather than in reward_terms.py because they decide when
# an EPISODE ENDS, not what the reward is -- reward_terms.FallPenalty just prices the event.
#
# The definition grew from real failures. It started as "torso pitch/roll past an angle", which a
# policy defeated by finding a stable crouch (both knees bent ~100 degrees, thigh resting on the
# floor) that stayed inside every threshold. Adding "any non-foot body part touching the ground"
# fixed that but was too trigger-happy on momentary stumbles, so it was debounced by requiring N
# consecutive steps -- which then failed the opposite way: a genuinely dragging knee only touches
# ~50-65% of the time (contact breaks and remakes every few frames while the leg is under torque),
# so it almost never produced an unbroken run and could hide indefinitely.
#
# So the two halves are now treated differently, because they genuinely are different:
#   - non-foot GROUND CONTACT ends the episode instantly. A knee or hand on the floor is not a
#     recoverable state, and any debounce is a loophole.
#   - bad ORIENTATION/height is debounced with an exponential moving average, since a brief lean
#     past the threshold really is recoverable. The EMA rewards a sustained tendency to be down
#     without requiring it to be unbroken.
FALL_ANGLE_DEG = 50.0
FALL_HEIGHT = 0.5   # below this torso z the character has collapsed even if it's upright
FALL_EMA_ALPHA = 0.02      # effective memory ~1/alpha = 50 steps (~0.5s) -- first-pass, not tuned
FALL_EMA_THRESHOLD = 0.5   # trigger once recent bad-orientation occupancy crosses this fraction

# Small random perturbation applied to qpos/qvel on every reset. Without it every episode -- and on
# the GPU side every one of the thousands of parallel envs -- starts from the identical rest pose
# with zero velocity, so the only thing making rollouts diverge is the policy's own action noise,
# which shrinks as training converges. That's variance reduction, not exploration: genuinely
# different starting conditions are the actual point of running many envs in parallel.
RESET_QPOS_NOISE_STD = 0.05  # radians, ~3 degrees -- small enough to stay a "similar" start pose
RESET_QVEL_NOISE_STD = 0.05  # rad/s (joints) or m/s (torso) -- a small push, not a shove


class WalkEnv(gym.Env):
    """Single-agent Gymnasium env: learn to run forward using hip/knee/hip_twist torque on the
    full tuned humanoid body. No command -- reward is just forward speed minus a small effort
    penalty, plus a fall penalty when free_torso=True (the default; see module docstring)."""

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(
        self, render_mode: str | None = None, free_torso: bool = True, ankle_mode: str | None = None,
    ) -> None:
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.free_torso = free_torso
        self.ankle_mode = ankle_mode
        self.walk_joints = BASE_WALK_JOINTS + _ankle_joint_names(ankle_mode)
        self._viewer = None

        self.model = mujoco.MjModel.from_xml_string(
            _build_model_xml(_actuators, fighters=(AGENT,), free_torso=free_torso, ankle_mode=ankle_mode)
        )
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.agent_id = self.model.body(AGENT).id
        self._chest_body_id = self.model.body(f"{AGENT}_chest").id
        self._x_dof = int(self.model.jnt_dofadr[self.model.body(AGENT).jntadr[0]])
        self._y_dof = int(self.model.jnt_dofadr[self.model.joint(f"{AGENT}_y").id])
        self._z_dof = int(self.model.jnt_dofadr[self.model.joint(f"{AGENT}_z").id])

        self._actuator_ids = [self.model.actuator(f"{AGENT}_{j}_m").id for j in self.walk_joints]
        self._joint_qpos_adr = [self.model.jnt_qposadr[self.model.joint(f"{AGENT}_{j}").id] for j in self.walk_joints]
        self._joint_dof_adr = [self.model.jnt_dofadr[self.model.joint(f"{AGENT}_{j}").id] for j in self.walk_joints]
        # Real ROM limits for each walk joint, so reset-state noise (see RESET_QPOS_NOISE_STD)
        # can't randomize a joint past a range it could never actually reach.
        self._joint_qpos_range = []
        for j in self.walk_joints:
            jid = self.model.joint(f"{AGENT}_{j}").id
            if self.model.jnt_limited[jid]:
                self._joint_qpos_range.append((float(self.model.jnt_range[jid][0]), float(self.model.jnt_range[jid][1])))
            else:
                self._joint_qpos_range.append(None)

        # For the feet air-time reward -- rewards a foot on touchdown for how long it was
        # airborne, paid once per landing (not every step), see _feet_air_time_bonus. Each side
        # maps to a LIST of geom ids since "detailed" splits one foot into heel+toe geoms --
        # "touching the floor" means either one is in contact, treating the pair as one foot for
        # reward purposes (no separate heel-strike/toe-off signal yet).
        if self.ankle_mode == "detailed":
            self._foot_geom_ids = {
                side: [self.model.geom(f"{AGENT}_heel_{side}").id, self.model.geom(f"{AGENT}_toe_{side}").id]
                for side in ("r", "l")
            }
        else:
            self._foot_geom_ids = {side: [self.model.geom(f"{AGENT}_foot_{side}").id] for side in ("r", "l")}
        self._floor_geom_id = self.model.geom("floor").id
        # Owning body of each foot (for the slip penalty's cvel lookup) -- whatever body the first
        # geom on that side belongs to, since a "detailed" foot's heel+toe geoms are on one body.
        self._foot_body_id = {
            side: int(self.model.geom_bodyid[geom_ids[0]]) for side, geom_ids in self._foot_geom_ids.items()
        }

        # Every other geom on the body -- thighs, shins, torso, arms, etc. -- for the "did some
        # non-foot part collapse onto the ground" fall check. The whole model is one agent, so
        # this is just "everything except the floor and the feet."
        _foot_ids_flat = {gid for ids in self._foot_geom_ids.values() for gid in ids}
        self._non_foot_geom_ids = [
            gi for gi in range(self.model.ngeom) if gi != self._floor_geom_id and gi not in _foot_ids_flat
        ]

        # Passive (unactuated) pitch/roll DOFs only exist in the model when free_torso=True --
        # not part of the action space, but the policy needs to see them to have any chance of
        # learning to balance.
        if self.free_torso:
            self._pitch_qpos_adr = int(self.model.jnt_qposadr[self.model.joint(f"{AGENT}_pitch").id])
            self._pitch_dof_adr = int(self.model.jnt_dofadr[self.model.joint(f"{AGENT}_pitch").id])
            self._roll_qpos_adr = int(self.model.jnt_qposadr[self.model.joint(f"{AGENT}_roll").id])
            self._roll_dof_adr = int(self.model.jnt_dofadr[self.model.joint(f"{AGENT}_roll").id])

        self._step_count = 0
        self._prev_action = np.zeros(len(self.walk_joints), dtype=np.float64)
        self._air_time = {"r": 0.0, "l": 0.0}
        self._down_ema = 0.0
        self._phase = 0.0
        self._target_speed = COMMAND_MIN_SPEED
        self._resample_countdown = 0

        n_obs = len(self._build_observation())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.walk_joints),), dtype=np.float32)

    # ---- Gymnasium API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        _set_rest_pose_qpos(self.model, self.data, AGENT)
        self._randomize_reset_state()
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(len(self.walk_joints), dtype=np.float64)
        self._air_time = {"r": 0.0, "l": 0.0}
        self._down_ema = 0.0
        # Randomized (not always 0) so the policy doesn't just learn one fixed phase-to-pose
        # alignment -- same reset-diversity rationale as RESET_QPOS_NOISE_STD.
        self._phase = float(self.np_random.uniform(0.0, 1.0))
        self._target_speed = float(self.np_random.uniform(COMMAND_MIN_SPEED, COMMAND_MAX_SPEED))
        self._resample_countdown = int(COMMAND_RESAMPLE_INTERVAL / self.dt)
        if self.render_mode == "human":
            self._render_frame()
        return self._build_observation(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        for actuator_id, value in zip(self._actuator_ids, action):
            self.data.ctrl[actuator_id] = value

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self._phase = (self._phase + self.dt / GAIT_CYCLE_TIME) % 1.0
        self._resample_countdown -= 1
        if self._resample_countdown <= 0:
            self._target_speed = float(self.np_random.uniform(COMMAND_MIN_SPEED, COMMAND_MAX_SPEED))
            self._resample_countdown = int(COMMAND_RESAMPLE_INTERVAL / self.dt)

        foot_touching = {side: self._foot_touching_floor(ids) for side, ids in self._foot_geom_ids.items()}
        air_time_before = self._advance_air_time(foot_touching)
        fallen = self._has_fallen()
        state = self._build_reward_state(action, fallen, foot_touching, air_time_before)
        self.reward_breakdown = {k: float(v) for k, v in reward_terms.breakdown(state).items()}
        reward = float(sum(self.reward_breakdown.values()))
        self._prev_action = action.copy()
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

    # ---- internals ----

    def _randomize_reset_state(self) -> None:
        """Perturb the just-set rest pose with small random qpos/qvel noise -- see
        RESET_QPOS_NOISE_STD's comment for why. Uses self.np_random (seeded via reset(seed=...))
        rather than the global numpy RNG, matching Gymnasium's own convention for reproducibility."""
        for qpos_adr, qpos_range in zip(self._joint_qpos_adr, self._joint_qpos_range):
            new_val = self.data.qpos[qpos_adr] + self.np_random.normal(0.0, RESET_QPOS_NOISE_STD)
            if qpos_range is not None:
                new_val = np.clip(new_val, qpos_range[0], qpos_range[1])
            self.data.qpos[qpos_adr] = new_val
        for dof_adr in self._joint_dof_adr:
            self.data.qvel[dof_adr] += self.np_random.normal(0.0, RESET_QVEL_NOISE_STD)

        self.data.qvel[self._x_dof] += self.np_random.normal(0.0, RESET_QVEL_NOISE_STD)
        self.data.qvel[self._y_dof] += self.np_random.normal(0.0, RESET_QVEL_NOISE_STD)
        self.data.qvel[self._z_dof] += self.np_random.normal(0.0, RESET_QVEL_NOISE_STD)

        if self.free_torso:
            self.data.qpos[self._pitch_qpos_adr] += self.np_random.normal(0.0, RESET_QPOS_NOISE_STD)
            self.data.qpos[self._roll_qpos_adr] += self.np_random.normal(0.0, RESET_QPOS_NOISE_STD)
            self.data.qvel[self._pitch_dof_adr] += self.np_random.normal(0.0, RESET_QVEL_NOISE_STD)
            self.data.qvel[self._roll_dof_adr] += self.np_random.normal(0.0, RESET_QVEL_NOISE_STD)

    def _chest_world_pitch_roll(self) -> tuple[float, float]:
        """The chest's actual world-space pitch/roll (radians), from its rotation matrix -- not
        just the root torso's own pitch/roll DOF. See the module-level comment above FALL_EMA_ALPHA
        for why: waist_bend/waist_twist sit between the root and the chest and are now actuated, so
        the root DOF alone no longer tells you how far the upper body is actually leaning."""
        xmat = self.data.xmat[self._chest_body_id].reshape(3, 3)
        pitch = np.arctan2(-xmat[2, 0], np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2))
        roll = np.arctan2(xmat[2, 1], xmat[2, 2])
        return float(pitch), float(roll)

    def _is_badly_oriented(self) -> bool:
        """Instantaneous orientation/height fall condition (this single step only) -- debounced by
        _has_fallen's EMA, since a momentary lean past the threshold really is recoverable."""
        if not self.free_torso:
            return False  # torso is mechanically locked upright -- can't fall by construction
        pitch_deg, roll_deg = (np.degrees(x) for x in self._chest_world_pitch_roll())
        torso_z = self.data.xpos[self.agent_id, 2]
        return abs(pitch_deg) > FALL_ANGLE_DEG or abs(roll_deg) > FALL_ANGLE_DEG or torso_z < FALL_HEIGHT

    def _has_fallen(self) -> bool:
        """Fallen = any non-foot body part touching the ground (INSTANT, no debounce -- a knee or
        hand on the floor is never a recoverable state), OR a sustained bad orientation/height,
        debounced via an exponential moving average so a momentary lean gets a recovery window.
        See FALL_EMA_ALPHA's comment for why these two halves are treated differently."""
        if self.free_torso and self._non_foot_touching_floor():
            return True
        bad_orientation = self._is_badly_oriented()
        self._down_ema = FALL_EMA_ALPHA * float(bad_orientation) + (1.0 - FALL_EMA_ALPHA) * self._down_ema
        return self._down_ema >= FALL_EMA_THRESHOLD

    def _foot_touching_floor(self, foot_geom_ids: list[int]) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self._floor_geom_id not in (contact.geom1, contact.geom2):
                continue
            if any(gid in (contact.geom1, contact.geom2) for gid in foot_geom_ids):
                return True
        return False

    def _non_foot_touching_floor(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self._floor_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self._floor_geom_id else contact.geom1
            if other in self._non_foot_geom_ids:
                return True
        return False

    def _advance_air_time(self, foot_touching: dict[str, bool]) -> dict[str, float]:
        """Update each foot's airborne timer and return what it was BEFORE this step's landing.

        Split out from the reward itself so the air-time TERM (reward_terms.FeetAirTime) stays
        pure arithmetic: it needs both the pre-landing value (to price the touchdown) and the
        post-update value (to price hanging too long), so the env hands it both.
        """
        before = dict(self._air_time)
        for side, touching in foot_touching.items():
            self._air_time[side] = 0.0 if touching else self._air_time[side] + self.dt
        return before

    def _balance_cost(self) -> float:
        """Squared horizontal distance from the center of mass to the midpoint between the feet --
        see reward_terms.BalancePenalty. Uses both feet regardless of contact state: mid-stride one
        foot is legitimately in the air, and gating on contact would make the target jump
        discontinuously every time a foot lifts or lands."""
        com_xy = self.data.subtree_com[self.agent_id][:2]
        feet_xy = np.array([self.data.xpos[bid][:2] for bid in self._foot_body_id.values()])
        return float(np.sum((com_xy - feet_xy.mean(axis=0)) ** 2))

    def _foot_linear_velocity(self, body_id: int) -> np.ndarray:
        """The foot's ACTUAL linear velocity in world coordinates.

        data.cvel is a spatial (com-based) velocity: its linear half is the velocity at the
        subtree's center of mass, NOT at the foot, so it picks up a large omega-cross-lever-arm
        term from whole-body rotation. Reading it directly reported 1.6-1.75 m/s of "slip" for a
        foot genuinely moving 0.13-0.20 m/s -- an 8-10x overestimate, 60-100x once squared, which
        made slip the single largest ongoing cost even standing still. Shifting the reference point
        to the body's inertial center recovers the true velocity -- verified against
        mj_objectVelocity (0.0 error) and finite-differenced position drift.
        """
        omega = self.data.cvel[body_id, 0:3]
        v_com = self.data.cvel[body_id, 3:6]
        com_ref = self.data.subtree_com[self.model.body_rootid[body_id]]
        return v_com + np.cross(omega, self.data.xipos[body_id] - com_ref)

    def _build_reward_state(self, action, fallen, foot_touching, air_time_before) -> NumpyState:
        """Gather everything reward_terms.py needs out of MuJoCo. All the backend-specific reading
        happens here; the terms themselves are shared with the GPU env."""
        chest_pitch, chest_roll = self._chest_world_pitch_roll()
        return NumpyState(
            target_speed=self._target_speed,
            forward_vel=self.data.qvel[self._x_dof],
            lateral_vel=self.data.qvel[self._y_dof],
            vertical_vel=self.data.qvel[self._z_dof],
            action=action,
            prev_action=self._prev_action,
            joint_accel=np.array([self.data.qacc[d] for d in self._joint_dof_adr]),
            joint_vel=np.array([self.data.qvel[d] for d in self._joint_dof_adr]),
            chest_pitch=chest_pitch,
            chest_roll=chest_roll,
            chest_ang_vel_sq=float(np.sum(self.data.cvel[self._chest_body_id, 0:3] ** 2)),
            com_support_offset_sq=self._balance_cost(),
            phase=self._phase,
            foot_touching=foot_touching,
            air_time=dict(self._air_time),
            air_time_before=air_time_before,
            foot_slip_sq={
                side: float(np.sum(self._foot_linear_velocity(bid)[:2] ** 2))
                for side, bid in self._foot_body_id.items()
            },
            fallen=fallen,
        )

    def _build_observation(self) -> np.ndarray:
        joint_state = []
        for qpos_adr, dof_adr in zip(self._joint_qpos_adr, self._joint_dof_adr):
            joint_state.append(self.data.qpos[qpos_adr])
            joint_state.append(self.data.qvel[dof_adr])

        forward_vel = self.data.qvel[self._x_dof]
        torso_z = self.data.xpos[self.agent_id, 2]

        # sin/cos encoding of the gait-phase clock (see GAIT_CYCLE_TIME) -- the policy needs to see
        # this to know which foot is currently supposed to be planted, not just react to state.
        # sin/cos rather than the raw [0,1) phase to avoid the discontinuous jump at wraparound.
        phase_angle = 2 * np.pi * self._phase
        obs = [
            np.array(joint_state, dtype=np.float64),
            # target_speed has to be observed, not just used in the reward -- otherwise a
            # randomized command is unsolvable, the policy would have no way to know what speed
            # it's currently being asked for. See COMMAND_MIN_SPEED's comment for why it's
            # randomized at all instead of one fixed target.
            [forward_vel, torso_z, self._target_speed, np.sin(phase_angle), np.cos(phase_angle)],
        ]
        if self.free_torso:
            obs.append([self.data.qpos[self._pitch_qpos_adr], self.data.qvel[self._pitch_dof_adr]])
            obs.append([self.data.qpos[self._roll_qpos_adr], self.data.qvel[self._roll_dof_adr]])
        return np.concatenate(obs).astype(np.float32)

    def _info(self) -> dict:
        return {
            "step": self._step_count,
            "forward_vel": float(self.data.qvel[self._x_dof]),
            "target_speed": self._target_speed,
        }

    def _render_frame(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.distance = 6
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = -18
        self._viewer.sync()
        time.sleep(self.dt)


def explore(free_torso: bool, ankle_mode: str | None = None) -> None:
    """Interactive viewer with torque sliders in the Control panel for the walker's action-space
    joints -- no trained policy, just hand-drag them to see how the legs (and torso, if
    free_torso, and the ankle, if ankle_mode) respond."""
    env = WalkEnv(free_torso=free_torso, ankle_mode=ankle_mode)
    env.reset()
    print(f"autobalance {'OFF (torso can fall)' if free_torso else 'ON (torso locked upright)'}")
    print(f"ankle_mode={ankle_mode!r}")
    print("Drag sliders in the viewer's Control panel:", ", ".join(f"{AGENT}_{j}_m" for j in env.walk_joints))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(env.dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal MuJoCo walker.")
    parser.add_argument("--explore", action="store_true", help="interactive viewer with torque sliders")
    parser.add_argument(
        "--no-autobalance", action="store_true",
        help="in --explore mode, let the torso actually fall (free_torso=True) instead of the "
             "default locked-upright autobalance -- matches what train.py/watch.py always use.",
    )
    parser.add_argument(
        "--ankle", choices=["single", "dual", "detailed"], default=None,
        help="in --explore mode, add an ankle joint: 'single' (one sagittal hinge, dorsi/"
             "plantarflexion), 'dual' (adds inversion/eversion too), or 'detailed' (same as dual "
             "but with a split heel/toe foot instead of one rigid capsule) -- see fight_env.py's "
             "_humanoid_body docstring. Omit for the original rigid (no-ankle) foot.",
    )
    args = parser.parse_args()
    if args.explore:
        explore(free_torso=args.no_autobalance, ankle_mode=args.ankle)
    else:
        print("Nothing to run without --explore -- use train.py / watch.py for RL.")
