"""GPU-batched version of walk_env.py, built on MuJoCo Warp + PyTorch -- runs thousands of
parallel copies of the walker on GPU in one process, natively on Windows. No JAX, no WSL2 (unlike
the earlier archive/walker_v1_mjx/ attempt, which needed both) -- NVIDIA Warp interops zero-copy
with PyTorch tensors directly, confirmed empirically on this machine.

Same body (fight_env.py, untouched), same task/reward as walk_env.py, including the fall
definition (torso pitch/roll/height OR any non-foot body part touching the ground, debounced via
an exponential moving average of that condition rather than requiring it instantly or for N
consecutive steps -- see walk_env.py's docstring for why) and jumping being an unpenalized,
allowed strategy, and a gait-phase clock driving an alternating stance/swing reward plus a
foot-slip penalty (see GAIT_CYCLE_TIME) -- only the physics backend changes, and "one env per
Python process" becomes "a batch of N worlds in one process". Measured
(not assumed) on this machine's RTX 3070 Ti: ~66,000 env-steps/sec at 1024 worlds, vs ~1,800 for
the single-env CPU walk_env.py and ~12,700 for the earlier JAX/MJX attempt at the same batch size.

No Gymnasium API here -- single-env conventions (per-step tuples, one reward float) don't fit a
batched-tensor workflow. reset()/step() take and return whole-batch torch tensors directly, which
is what a from-scratch PyTorch PPO loop (train_gpu.py) actually wants.

Auto-resets internally: step() detects fallen/truncated envs and resets just those before
returning, so the caller never needs to track episode boundaries itself -- same idea as vectorized
training environments in Isaac Gym/legged_gym-style setups.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from fight_env import _actuators, _build_model_xml, _set_rest_pose_qpos
import reward_terms
from reward_terms import (
    COMMAND_MAX_SPEED, COMMAND_MIN_SPEED, COMMAND_RESAMPLE_INTERVAL, COMMAND_STAND_PROB,
    COMMAND_STAND_SPEED, GAIT_CYCLE_TIME, PUSH_INTERVAL, PUSH_MAX_VEL,
    STANDING_COMMAND_THRESHOLD,
    LEG_BALANCE_EMA_ALPHA, TorchState,
)

AGENT = "red"
# Every joint with a motor in fight_env.py's _actuators() except the ankle (a separate, optional
# variant -- see _ankle_joint_names below). shoulder_abduct_{side} is the arm's second axis (raising
# the arm out to the side, not just forward/back) -- see fight_env.py's SHOULDER_ABDUCT_MAX comment
# and walk_env.py's module docstring for why every actuated DOF is included, not just the legs.
BASE_WALK_JOINTS = [
    "hip_r", "hip_l", "hip_abduct_r", "hip_abduct_l", "knee_r", "knee_l", "hip_twist",
    "shoulder_r", "shoulder_l", "shoulder_abduct_r", "shoulder_abduct_l", "elbow_r", "elbow_l",
    "waist_twist", "waist_bend",
]

MAX_EPISODE_STEPS = 1000  # 10s at dt=0.01

# What counts as "fallen" -- see walk_env.py for the full history behind this definition. Non-foot
# ground contact ends the episode instantly (a knee on the floor is not recoverable, and any
# debounce there is a loophole); bad orientation/height is debounced with an EMA, because a brief
# lean past the threshold genuinely is recoverable.
FALL_ANGLE_DEG = 50.0
FALL_HEIGHT = 0.5
FALL_EMA_ALPHA = 0.02
FALL_EMA_THRESHOLD = 0.5

# Reset-state randomization -- see walk_env.py. Without it all N parallel envs start identically,
# so the only divergence between them is the policy's own shrinking action noise.
RESET_QPOS_NOISE_STD = 0.05
RESET_QVEL_NOISE_STD = 0.05


def _ankle_joint_names(ankle_mode: str | None) -> list[str]:
    if ankle_mode is None:
        return []
    if ankle_mode == "single":
        return ["ankle_r", "ankle_l"]
    if ankle_mode in ("dual", "detailed"):
        return ["ankle_pitch_r", "ankle_pitch_l", "ankle_roll_r", "ankle_roll_l"]
    raise ValueError(f"unknown ankle_mode: {ankle_mode!r} (expected None, 'single', 'dual', or 'detailed')")


class WalkEnvGPU:
    def __init__(self, ankle_mode: str | None = None, num_envs: int = 1024, device: str = "cuda:0") -> None:
        self.ankle_mode = ankle_mode
        self.num_envs = num_envs
        self.device = device
        self.walk_joints = BASE_WALK_JOINTS + _ankle_joint_names(ankle_mode)
        self.action_dim = len(self.walk_joints)

        xml = _build_model_xml(_actuators, fighters=(AGENT,), free_torso=True, ankle_mode=ankle_mode)
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, mj_data)
        _set_rest_pose_qpos(self.mj_model, mj_data, AGENT)
        mujoco.mj_forward(self.mj_model, mj_data)
        self.dt = self.mj_model.opt.timestep

        # put_model/put_data have no device argument -- they use whatever Warp's current default
        # device is, which is cuda:0 whenever a GPU is present regardless of what's passed here.
        # ScopedDevice is what actually controls where the created arrays live. njmax is a
        # fixed-size, per-world buffer for active constraints (GPU kernels need static shapes,
        # unlike classic MuJoCo's dynamic allocation) -- auto-sized from a single deterministic
        # mj_data, which undercounts once reset randomization/orientation changes started
        # producing more varied, sometimes-awkward poses (more simultaneous contacts/joint limits)
        # than that one reference pose ever hit. Confirmed empirically: the auto default was
        # njmax=64, and a real run overflowed needing 66 -- doubling it gives real headroom instead
        # of bumping by the exact overflow amount and hitting this again next time.
        with wp.ScopedDevice(device):
            self.m = mjwarp.put_model(self.mj_model)
            self.d = mjwarp.put_data(self.mj_model, mj_data, nworld=num_envs, njmax=128)
        self._init_qpos = torch.as_tensor(mj_data.qpos.copy(), dtype=torch.float32, device=device)

        m = self.mj_model
        self.agent_id = m.body(AGENT).id
        self._chest_body_id = m.body(f"{AGENT}_chest").id
        self._x_dof = int(m.jnt_dofadr[m.body(AGENT).jntadr[0]])
        self._y_dof = int(m.jnt_dofadr[m.joint(f"{AGENT}_y").id])
        self._z_dof = int(m.jnt_dofadr[m.joint(f"{AGENT}_z").id])
        self._pitch_qpos_adr = int(m.jnt_qposadr[m.joint(f"{AGENT}_pitch").id])
        self._pitch_dof_adr = int(m.jnt_dofadr[m.joint(f"{AGENT}_pitch").id])
        self._roll_qpos_adr = int(m.jnt_qposadr[m.joint(f"{AGENT}_roll").id])
        self._roll_dof_adr = int(m.jnt_dofadr[m.joint(f"{AGENT}_roll").id])

        self._actuator_ids = torch.as_tensor(
            [m.actuator(f"{AGENT}_{j}_m").id for j in self.walk_joints], dtype=torch.long, device=device
        )
        self._joint_qpos_adr = [int(m.jnt_qposadr[m.joint(f"{AGENT}_{j}").id]) for j in self.walk_joints]
        self._joint_dof_adr = [int(m.jnt_dofadr[m.joint(f"{AGENT}_{j}").id]) for j in self.walk_joints]
        # Real ROM limits for each walk joint, so reset-state noise (see RESET_QPOS_NOISE_STD)
        # can't randomize a joint past a range it could never actually reach.
        self._joint_qpos_range: list[tuple[float, float] | None] = []
        for j in self.walk_joints:
            jid = m.joint(f"{AGENT}_{j}").id
            if m.jnt_limited[jid]:
                self._joint_qpos_range.append((float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])))
            else:
                self._joint_qpos_range.append(None)

        self._foot_geom_ids: dict[str, list[int]] = {}
        for side in ("r", "l"):
            if ankle_mode == "detailed":
                self._foot_geom_ids[side] = [m.geom(f"{AGENT}_heel_{side}").id, m.geom(f"{AGENT}_toe_{side}").id]
            else:
                self._foot_geom_ids[side] = [m.geom(f"{AGENT}_foot_{side}").id]
        self._floor_geom_id = m.geom("floor").id
        # Owning body of each foot (for the slip penalty's cvel lookup) -- whatever body the first
        # geom on that side belongs to, since a "detailed" foot's heel+toe geoms are on one body.
        self._foot_body_id = {side: int(m.geom_bodyid[ids[0]]) for side, ids in self._foot_geom_ids.items()}
        # Planted resting height per foot -- FootClearance measures lift above this. Taken from
        # the same rest-pose mj_data used to seed the sim, so it matches walk_env.py exactly.
        # Resting SOLE height per foot (lowest point of its geoms) -- FootClearance measures lift
        # above this. See walk_env.py's _foot_sole_z for why the sole and not the body origin.
        def _rest_sole(side):
            lows = []
            for g in self._foot_geom_ids[side]:
                pos = mj_data.geom_xpos[g]
                mat = mj_data.geom_xmat[g].reshape(3, 3)
                size = m.geom_size[g]
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    ax = mat[:, 2] * size[1]
                    lows.append(min((pos + ax)[2], (pos - ax)[2]) - size[0])
                elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    lows.append(pos[2] - float(abs(mat[2, :3]) @ size[:3]))
                else:
                    lows.append(pos[2] - size[0])
            return float(min(lows))
        self._foot_rest_sole_z = {side: _rest_sole(side) for side in self._foot_geom_ids}
        self._knee_body_id = {side: m.body(f'{AGENT}_shin_{side}').id for side in ('r', 'l')}
        self._knee_rest_z = {side: float(mj_data.xpos[bid][2]) for side, bid in self._knee_body_id.items()}
        # Authored joint limits for JointLimitPenalty (unlimited joints get +/-inf).
        _lo, _hi = [], []
        for j in self.walk_joints:
            jid = m.joint(f'{AGENT}_{j}').id
            if m.jnt_limited[jid]:
                _lo.append(float(m.jnt_range[jid][0])); _hi.append(float(m.jnt_range[jid][1]))
            else:
                _lo.append(float('-inf')); _hi.append(float('inf'))
        self._joint_lower = torch.tensor(_lo, dtype=torch.float32, device=device)
        self._joint_upper = torch.tensor(_hi, dtype=torch.float32, device=device)
        # Per-geom capsule params, cached for the batched sole computation each step.
        self._foot_geom_params = {
            side: [(g, [float(v) for v in m.geom_size[g][:3]], int(m.geom_type[g]))
                   for g in self._foot_geom_ids[side]]
            for side in self._foot_geom_ids
        }
        self._GEOM_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        self._GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
        # Root body of each foot's kinematic tree -- the reference point cvel's linear half is
        # expressed at (see _foot_slip_cost).
        self._foot_rootid = {side: int(m.body_rootid[bid]) for side, bid in self._foot_body_id.items()}

        _foot_ids_flat = {gid for ids in self._foot_geom_ids.values() for gid in ids}
        _non_foot = [gi for gi in range(m.ngeom) if gi != self._floor_geom_id and gi not in _foot_ids_flat]
        self._non_foot_geom_ids = torch.as_tensor(_non_foot, dtype=torch.long, device=device)
        # Shins are debounced rather than instant-kill -- see walk_env.py for the measurements
        # (a shin hangs ~3.8cm off the floor by construction, so brushing it is normal
        # articulation, while the thigh sits 34cm up and touching it is a real collapse).
        _shin_bodies = {m.body(f"{AGENT}_shin_{s}").id for s in ("r", "l")}
        _shins = [gi for gi in _non_foot if m.geom_bodyid[gi] in _shin_bodies]
        self._shin_geom_ids = torch.as_tensor(_shins, dtype=torch.long, device=device)
        self._collapse_geom_ids = torch.as_tensor(
            [gi for gi in _non_foot if gi not in _shins], dtype=torch.long, device=device)

        self._step_count = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._prev_action = torch.zeros(num_envs, self.action_dim, dtype=torch.float32, device=device)
        self._air_time = {side: torch.zeros(num_envs, dtype=torch.float32, device=device) for side in ("r", "l")}
        self._down_ema = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._leg_lift_ema = {side: torch.zeros(num_envs, dtype=torch.float32, device=device)
                              for side in ('r', 'l')}
        self._phase = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._target_speed = torch.full((num_envs,), COMMAND_MIN_SPEED, dtype=torch.float32, device=device)
        self._resample_countdown = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._push_countdown = torch.randint(
            1, int(PUSH_INTERVAL / self.dt) + 1, (num_envs,), device=device)

        with wp.ScopedDevice(device):
            self.obs_dim = self._compute_observation().shape[1]

    # ---- API ----

    def reset(self) -> torch.Tensor:
        with wp.ScopedDevice(self.device):
            self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
            return self._compute_observation()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # mjwarp allocates some internal buffers lazily on first use rather than all up front in
        # put_data/put_model -- confirmed empirically (a device="cpu" env still tried to allocate
        # on cuda:0 the first time step() ran, even though construction was correctly scoped).
        # Scoping every call that touches mjwarp, not just construction, is what actually pins it.
        with wp.ScopedDevice(self.device):
            # Pre-clamp command kept for ActionMagnitudePenalty -- see walk_env.py.
            raw_action = action
            action = torch.clamp(action, -1.0, 1.0)
            ctrl = wp.to_torch(self.d.ctrl)
            ctrl.zero_()
            ctrl[:, self._actuator_ids] = action

            # Push before stepping, so the disturbance is part of the dynamics this action meets.
            self._maybe_push()
            mjwarp.step(self.m, self.d)
            self._step_count += 1
            # The gait clock only runs while there is a gait. Advancing it during a "stand still"
            # command fed the policy a ticking metronome it had learned means "step now", directly
            # contradicting the command in the same observation vector -- measured, standing collapsed
            # from 85% double-support at 2M steps (an inert policy) to 5% once walking was learned, and
            # then had to slowly re-learn to SUPPRESS the clock rather than simply not receiving it.
            # Frozen, sin/cos(phase) hold constant and "stand" becomes an unambiguous input.
            walking = (self._target_speed >= STANDING_COMMAND_THRESHOLD).float()
            self._phase = (self._phase + walking * (self.dt / GAIT_CYCLE_TIME)) % 1.0
            self._resample_command()

            foot_touching = self._foot_contact_state()
            air_time_before = self._advance_air_time(foot_touching)
            fallen = self._has_fallen()
            # Per-leg running average lift, for GaitSymmetry (see walk_env.py).
            _kh = self._knee_heights()
            for _s in self._leg_lift_ema:
                self._leg_lift_ema[_s] = (LEG_BALANCE_EMA_ALPHA * torch.clamp(_kh[_s], min=0.0)
                                          + (1.0 - LEG_BALANCE_EMA_ALPHA) * self._leg_lift_ema[_s])
            state = self._build_reward_state(action, fallen, foot_touching, air_time_before, raw_action)
            # Per-term values kept for TensorBoard (train_gpu.py logs them) -- two separate bugs
            # this project hit were invisible in the total and obvious per-term.
            self.reward_breakdown = reward_terms.breakdown(state)
            reward = sum(self.reward_breakdown.values())
            self._prev_action = action

            truncated = self._step_count >= MAX_EPISODE_STEPS
            done = fallen | truncated
            if done.any():
                self._reset_envs(done)

            obs = self._compute_observation()
        return obs, reward, fallen, truncated

    # ---- internals ----

    def _reset_envs(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        with wp.ScopedDevice(self.device):
            qpos = wp.to_torch(self.d.qpos)
            qvel = wp.to_torch(self.d.qvel)
            qpos[idx] = self._init_qpos
            qvel[idx] = 0.0
            self._randomize_reset_state(qpos, qvel, idx)
            self._step_count[idx] = 0
            self._prev_action[idx] = 0.0
            for side in self._air_time:
                self._air_time[side][idx] = 0.0
            self._down_ema[idx] = 0.0
            for _s in self._leg_lift_ema:
                self._leg_lift_ema[_s][idx] = 0.0
            # Randomized (not always 0) so envs don't all share one fixed phase-to-pose alignment
            # -- same reset-diversity rationale as RESET_QPOS_NOISE_STD.
            self._phase[idx] = torch.rand(idx.numel(), device=self.device)
            self._target_speed[idx] = self._sample_command(idx.numel())
            self._resample_countdown[idx] = int(COMMAND_RESAMPLE_INTERVAL / self.dt)
            # Staggered so the whole batch doesn't get shoved on the same step.
            self._push_countdown[idx] = torch.randint(
                1, int(PUSH_INTERVAL / self.dt) + 1, (idx.numel(),), device=self.device)
            mjwarp.forward(self.m, self.d)

    def _sample_command(self, n: int) -> torch.Tensor:
        """n commanded speeds, a COMMAND_STAND_PROB fraction of them "stand still" (zero)."""
        speeds = torch.empty(n, device=self.device).uniform_(COMMAND_MIN_SPEED, COMMAND_MAX_SPEED)
        stand = torch.rand(n, device=self.device) < COMMAND_STAND_PROB
        return torch.where(stand, torch.full_like(speeds, COMMAND_STAND_SPEED), speeds)

    def _maybe_push(self) -> None:
        """Shove the torso of whichever envs are due, so they have to catch themselves."""
        self._push_countdown -= 1
        due = self._push_countdown <= 0
        if not due.any():
            return
        idx = due.nonzero(as_tuple=True)[0]
        qvel = wp.to_torch(self.d.qvel)
        for dof in (self._x_dof, self._y_dof):
            qvel[idx, dof] += (torch.rand(idx.numel(), device=self.device) * 2.0 - 1.0) * PUSH_MAX_VEL
        self._push_countdown[idx] = int(PUSH_INTERVAL / self.dt)

    def _resample_command(self) -> None:
        """Ticks the per-env countdown and resamples target_speed for whichever envs hit zero --
        see COMMAND_RESAMPLE_INTERVAL's comment for why the command changes mid-episode too, not
        just at reset."""
        self._resample_countdown -= 1
        due = self._resample_countdown <= 0
        if due.any():
            idx = due.nonzero(as_tuple=True)[0]
            self._target_speed[idx] = self._sample_command(idx.numel())
            self._resample_countdown[idx] = int(COMMAND_RESAMPLE_INTERVAL / self.dt)

    def _randomize_reset_state(self, qpos: torch.Tensor, qvel: torch.Tensor, idx: torch.Tensor) -> None:
        """Perturb the just-reset envs (idx) with small random qpos/qvel noise, in place -- see
        RESET_QPOS_NOISE_STD's comment for why."""
        n = idx.numel()
        for qpos_adr, qpos_range in zip(self._joint_qpos_adr, self._joint_qpos_range):
            new_val = qpos[idx, qpos_adr] + torch.randn(n, device=self.device) * RESET_QPOS_NOISE_STD
            if qpos_range is not None:
                new_val = new_val.clamp(qpos_range[0], qpos_range[1])
            qpos[idx, qpos_adr] = new_val
        for dof_adr in self._joint_dof_adr:
            qvel[idx, dof_adr] += torch.randn(n, device=self.device) * RESET_QVEL_NOISE_STD

        qpos[idx, self._pitch_qpos_adr] += torch.randn(n, device=self.device) * RESET_QPOS_NOISE_STD
        qpos[idx, self._roll_qpos_adr] += torch.randn(n, device=self.device) * RESET_QPOS_NOISE_STD
        qvel[idx, self._x_dof] += torch.randn(n, device=self.device) * RESET_QVEL_NOISE_STD
        qvel[idx, self._y_dof] += torch.randn(n, device=self.device) * RESET_QVEL_NOISE_STD
        qvel[idx, self._z_dof] += torch.randn(n, device=self.device) * RESET_QVEL_NOISE_STD
        qvel[idx, self._pitch_dof_adr] += torch.randn(n, device=self.device) * RESET_QVEL_NOISE_STD
        qvel[idx, self._roll_dof_adr] += torch.randn(n, device=self.device) * RESET_QVEL_NOISE_STD

    def _chest_world_pitch_roll(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The chest's actual world-space pitch/roll (radians), from its rotation matrix -- not
        just the root torso's own pitch/roll DOF. See FALL_EMA_ALPHA's comment above for why."""
        xmat = wp.to_torch(self.d.xmat)[:, self._chest_body_id]  # (num_envs, 3, 3)
        pitch = torch.atan2(-xmat[:, 2, 0], torch.sqrt(xmat[:, 2, 1] ** 2 + xmat[:, 2, 2] ** 2))
        roll = torch.atan2(xmat[:, 2, 1], xmat[:, 2, 2])
        return pitch, roll

    def _is_badly_oriented(self) -> torch.Tensor:
        """Instantaneous orientation/height fall condition -- debounced by _has_fallen's EMA."""
        xpos = wp.to_torch(self.d.xpos)
        chest_pitch, chest_roll = self._chest_world_pitch_roll()
        pitch_deg = torch.rad2deg(chest_pitch)
        roll_deg = torch.rad2deg(chest_roll)
        torso_z = xpos[:, self.agent_id, 2]
        return (pitch_deg.abs() > FALL_ANGLE_DEG) | (roll_deg.abs() > FALL_ANGLE_DEG) | (torso_z < FALL_HEIGHT)

    def _has_fallen(self) -> torch.Tensor:
        """Three tiers -- INSTANT for thigh/torso/arms/head on the ground (34cm+ of clearance, so
        that's an unambiguous collapse), DEBOUNCED for shins (~3.8cm of clearance by construction,
        so a graze is normal articulation but sustained contact is kneeling) and for bad
        orientation/height. See walk_env.py's _has_fallen for the full reasoning."""
        collapsed = self._geoms_touching_floor(self._collapse_geom_ids)
        down = self._is_badly_oriented() | self._geoms_touching_floor(self._shin_geom_ids)
        self._down_ema = FALL_EMA_ALPHA * down.float() + (1.0 - FALL_EMA_ALPHA) * self._down_ema
        return collapsed | (self._down_ema >= FALL_EMA_THRESHOLD)

    def _active_contact_mask(self) -> torch.Tensor:
        """Which entries of the flat contact buffer are REAL this step.

        mjwarp's contact arrays are fixed-size (naconmax) and are NOT cleared between steps -- only
        the first d.nacon entries are valid, exactly like classic MuJoCo's d.ncon. Filtering on
        `dist <= 0` (what this used to do) is not enough: measured on this model, all 96 buffer
        slots report dist <= 0 while only 3 are real, so 93 stale/garbage geom pairs were being
        treated as live contacts every single step. That made _non_foot_touching_floor fire almost
        constantly -- which, once non-foot contact became an instant episode-ender, terminated
        episodes after ~9 steps regardless of what the policy did (confirmed against the CPU env,
        which reported zero contacts on the same physics state where the GPU claimed a foot was
        still planted).
        """
        nacon = int(wp.to_torch(self.d.nacon)[0])
        n_slots = wp.to_torch(self.d.contact.dist).shape[0]
        return torch.arange(n_slots, device=self.device) < nacon

    def _geoms_touching_floor(self, geom_ids: torch.Tensor) -> torch.Tensor:
        """Per-env: is any geom in `geom_ids` in contact with the floor this step?"""
        geom = wp.to_torch(self.d.contact.geom)
        worldid = wp.to_torch(self.d.contact.worldid).long()
        active = self._active_contact_mask() & (wp.to_torch(self.d.contact.dist) <= 0.0)

        involves_floor = (geom[:, 0] == self._floor_geom_id) | (geom[:, 1] == self._floor_geom_id)
        other = torch.where(geom[:, 0] == self._floor_geom_id, geom[:, 1], geom[:, 0])
        mask = active & involves_floor & torch.isin(other, geom_ids)

        touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        touching[worldid[mask]] = True
        return touching

    def _non_foot_touching_floor(self) -> torch.Tensor:
        """Any non-foot part touching the ground, shins included -- diagnostics only. The fall
        decision uses the tiered check in _has_fallen."""
        return self._geoms_touching_floor(self._non_foot_geom_ids)

    def _foot_contact_state(self) -> dict[str, torch.Tensor]:
        """Per-env, per-foot touching-the-floor boolean -- computed once per step and shared by
        the air-time bonus, gait-phase reward, and slip penalty instead of each re-walking
        d.contact separately."""
        geom = wp.to_torch(self.d.contact.geom)
        worldid = wp.to_torch(self.d.contact.worldid).long()
        active = self._active_contact_mask() & (wp.to_torch(self.d.contact.dist) <= 0.0)

        result = {}
        for side, geom_ids in self._foot_geom_ids.items():
            touching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for gid in geom_ids:
                is_pair = ((geom[:, 0] == gid) & (geom[:, 1] == self._floor_geom_id)) | (
                    (geom[:, 1] == gid) & (geom[:, 0] == self._floor_geom_id)
                )
                touching[worldid[is_pair & active]] = True
            result[side] = touching
        return result

    def _advance_air_time(self, foot_touching: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update each foot's airborne timer, returning the pre-landing values -- see walk_env.py."""
        before = {side: t.clone() for side, t in self._air_time.items()}
        for side, touching in foot_touching.items():
            self._air_time[side] = torch.where(
                touching, torch.zeros_like(self._air_time[side]), self._air_time[side] + self.dt)
        return before

    def _balance_cost(self) -> torch.Tensor:
        """Squared horizontal distance from CoM to the midpoint between the feet."""
        com_xy = wp.to_torch(self.d.subtree_com)[:, self.agent_id, :2]
        xpos = wp.to_torch(self.d.xpos)
        feet_xy = torch.stack([xpos[:, bid, :2] for bid in self._foot_body_id.values()], dim=1)
        return ((com_xy - feet_xy.mean(dim=1)) ** 2).sum(dim=-1)

    def _foot_heights(self) -> dict[str, torch.Tensor]:
        """Per-env SOLE height above its planted resting height (see walk_env.py's _foot_sole_z --
        measuring the body origin instead made the term blind to foot orientation and let the
        policy park permanently heel-down for free)."""
        gxpos = wp.to_torch(self.d.geom_xpos)
        gxmat = wp.to_torch(self.d.geom_xmat)
        out = {}
        for side, params in self._foot_geom_params.items():
            low = None
            for gid, size, gtype in params:
                pos_z = gxpos[:, gid, 2]
                if gtype == self._GEOM_CAPSULE:
                    ax_z = gxmat[:, gid, 2, 2] * size[1]  # world z-component of the capsule axis
                    g_low = torch.minimum(pos_z + ax_z, pos_z - ax_z) - size[0]
                elif gtype == self._GEOM_BOX:
                    # lowest corner: each half-extent projected onto world z
                    drop = sum(gxmat[:, gid, 2, i].abs() * size[i] for i in range(3))
                    g_low = pos_z - drop
                else:
                    g_low = pos_z - size[0]
                low = g_low if low is None else torch.minimum(low, g_low)
            out[side] = low - self._foot_rest_sole_z[side]
        return out

    def _knee_heights(self) -> dict[str, torch.Tensor]:
        """Per-env knee height above its resting height (see KneeLift)."""
        xpos = wp.to_torch(self.d.xpos)
        return {side: xpos[:, bid, 2] - self._knee_rest_z[side]
                for side, bid in self._knee_body_id.items()}

    def _foot_slip_sq(self) -> dict[str, torch.Tensor]:
        """Squared horizontal speed of each foot.

        cvel's linear half is the velocity at the SUBTREE COM, not at the foot, so reading it
        directly overstated foot speed 8-10x via an omega-cross-lever-arm term. Shifting the
        reference to the body's inertial center (xipos) recovers the true velocity -- verified
        against mj_objectVelocity. See walk_env.py's _foot_linear_velocity.
        """
        cvel = wp.to_torch(self.d.cvel)
        xipos = wp.to_torch(self.d.xipos)
        subtree_com = wp.to_torch(self.d.subtree_com)
        out = {}
        for side, bid in self._foot_body_id.items():
            offset = xipos[:, bid] - subtree_com[:, self._foot_rootid[side]]
            v = cvel[:, bid, 3:6] + torch.cross(cvel[:, bid, 0:3], offset, dim=-1)
            out[side] = v[:, 0] ** 2 + v[:, 1] ** 2
        return out

    def _build_reward_state(self, action, fallen, foot_touching, air_time_before, raw_action=None) -> TorchState:
        """Gather everything reward_terms.py needs out of MuJoCo Warp. All backend-specific reading
        happens here; the terms themselves are shared with the CPU env."""
        qvel = wp.to_torch(self.d.qvel)
        qacc = wp.to_torch(self.d.qacc)
        chest_pitch, chest_roll = self._chest_world_pitch_roll()
        chest_ang_vel = wp.to_torch(self.d.cvel)[:, self._chest_body_id, 0:3]
        return TorchState(
            target_speed=self._target_speed,
            forward_vel=qvel[:, self._x_dof],
            lateral_vel=qvel[:, self._y_dof],
            vertical_vel=qvel[:, self._z_dof],
            action=action,
            prev_action=self._prev_action,
            raw_action=raw_action,
            joint_accel=qacc[:, self._joint_dof_adr],
            joint_vel=qvel[:, self._joint_dof_adr],
            chest_pitch=chest_pitch,
            chest_roll=chest_roll,
            chest_ang_vel_sq=(chest_ang_vel ** 2).sum(dim=-1),
            com_support_offset_sq=self._balance_cost(),
            phase=self._phase,
            foot_touching=foot_touching,
            air_time=dict(self._air_time),
            air_time_before=air_time_before,
            leg_lift_ema=dict(self._leg_lift_ema),
            knee_height=self._knee_heights(),
            joint_pos=wp.to_torch(self.d.qpos)[:, self._joint_qpos_adr],
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
            foot_height=self._foot_heights(),
            foot_slip_sq=self._foot_slip_sq(),
            fallen=fallen,
        )

    def _compute_observation(self) -> torch.Tensor:
        qpos = wp.to_torch(self.d.qpos)
        qvel = wp.to_torch(self.d.qvel)
        xpos = wp.to_torch(self.d.xpos)

        joint_qpos = qpos[:, self._joint_qpos_adr]
        joint_qvel = qvel[:, self._joint_dof_adr]
        joint_state = torch.stack([joint_qpos, joint_qvel], dim=-1).reshape(self.num_envs, -1)

        forward_vel = qvel[:, self._x_dof : self._x_dof + 1]
        torso_z = xpos[:, self.agent_id, 2:3]
        pitch = qpos[:, self._pitch_qpos_adr : self._pitch_qpos_adr + 1]
        pitch_vel = qvel[:, self._pitch_dof_adr : self._pitch_dof_adr + 1]
        roll = qpos[:, self._roll_qpos_adr : self._roll_qpos_adr + 1]
        roll_vel = qvel[:, self._roll_dof_adr : self._roll_dof_adr + 1]

        # sin/cos encoding of the gait-phase clock -- see GAIT_CYCLE_TIME's comment. sin/cos
        # rather than raw phase to avoid the discontinuous jump at wraparound.
        phase_angle = 2 * torch.pi * self._phase
        phase_sin = torch.sin(phase_angle).unsqueeze(-1)
        phase_cos = torch.cos(phase_angle).unsqueeze(-1)
        # target_speed has to be observed, not just used in the reward -- see walk_env.py's
        # comment on the same line for why (a randomized command is unsolvable if the policy can't
        # see what it's currently being asked for).
        target_speed = self._target_speed.unsqueeze(-1)

        return torch.cat(
            [joint_state, forward_vel, torso_z, target_speed, phase_sin, phase_cos, pitch, pitch_vel, roll, roll_vel],
            dim=-1,
        )
