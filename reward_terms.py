"""The walker's reward, defined ONCE for both backends.

Every term the walker is optimizing lives here as a small named class with its constant, its
formula, and a comment saying why it exists. walk_env.py (single CPU env, numpy scalars) and
walk_env_gpu.py (batched MuJoCo Warp env, torch tensors of shape (num_envs,)) both build a
`WalkState` and hand it to the same list of terms.

Why shared rather than one copy per backend: the two envs previously duplicated the whole reward
by hand, and they silently drifted apart -- a measured 0.74 reward difference on identical physics
states, meaning watch.py was showing behavior scored differently than the policy was trained on.
With one definition that class of bug can't happen: a term is either right for both or wrong for
both.

The terms are pure math. All backend-specific extraction (reading qpos, walking the contact list,
tracking air time) happens in the env and arrives here precomputed on WalkState, so nothing in
this file needs to know whether it's holding a float or a (num_envs,) tensor. The handful of
operations that differ between numpy and torch (exp, clamp, where) come in through WalkState.

To change what the walker optimizes, edit a constant or a `compute` here -- and nowhere else.
To add a term: write the class, add it to REWARD_TERMS. It's logged to TensorBoard automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WalkState:
    """Everything the reward terms need, already extracted from the physics backend.

    The env fills this in each step; the terms only do arithmetic on it. That split is what lets
    one set of term definitions serve both backends -- nothing below this line knows whether a
    field holds a numpy scalar (CPU, one env) or a torch tensor of shape (num_envs,) (GPU, batched).

    The few operations that differ between numpy and torch come in through the small method set at
    the bottom, overridden by NumpyState / TorchState.
    """

    # Command and root motion
    target_speed: object = 0.0
    forward_vel: object = 0.0
    lateral_vel: object = 0.0
    vertical_vel: object = 0.0

    # Control
    action: object = None
    prev_action: object = None

    # Actuated joints
    joint_accel: object = None
    joint_vel: object = None

    # Chest pose (NOT the root torso joint -- see OrientationPenalty)
    chest_pitch: object = 0.0
    chest_roll: object = 0.0
    chest_ang_vel_sq: object = 0.0

    # Balance: squared horizontal distance from CoM to the midpoint between the feet
    com_support_offset_sq: object = 0.0

    # Gait
    phase: object = 0.0
    foot_touching: dict = field(default_factory=dict)   # side -> bool
    air_time: dict = field(default_factory=dict)        # side -> seconds airborne, AFTER this step
    air_time_before: dict = field(default_factory=dict) # side -> seconds airborne BEFORE landing
    foot_slip_sq: dict = field(default_factory=dict)    # side -> squared horizontal foot speed

    fallen: object = False

    # --- backend operations (overridden per backend) ---
    def exp(self, x): raise NotImplementedError
    def where(self, cond, a, b): raise NotImplementedError
    def clamp_min(self, x, lo): raise NotImplementedError
    def clamp_max(self, x, hi): raise NotImplementedError
    def logical_and(self, a, b): raise NotImplementedError
    def logical_not(self, a): raise NotImplementedError
    def eq(self, a, b): raise NotImplementedError
    def mean_over_joints(self, x): raise NotImplementedError
    def to_float(self, x): raise NotImplementedError


class NumpyState(WalkState):
    """Single-env backend (walk_env.py): plain floats and 1-D numpy arrays."""

    def exp(self, x):
        import numpy as np
        return np.exp(x)

    def where(self, cond, a, b):
        import numpy as np
        return np.where(cond, a, b)

    def clamp_min(self, x, lo):
        import numpy as np
        return np.maximum(x, lo)

    def clamp_max(self, x, hi):
        import numpy as np
        return np.minimum(x, hi)

    def logical_and(self, a, b):
        import numpy as np
        return np.logical_and(a, b)

    def logical_not(self, a):
        import numpy as np
        return np.logical_not(a)

    def eq(self, a, b):
        return a == b

    def mean_over_joints(self, x):
        import numpy as np
        return float(np.mean(x))

    def to_float(self, x):
        return float(x)


class TorchState(WalkState):
    """Batched GPU backend (walk_env_gpu.py): torch tensors of shape (num_envs,)."""

    def exp(self, x):
        import torch
        return torch.exp(x)

    def where(self, cond, a, b):
        import torch
        return torch.where(cond, a, b)

    def clamp_min(self, x, lo):
        import torch
        return torch.clamp(x, min=lo)

    def clamp_max(self, x, hi):
        import torch
        return torch.clamp(x, max=hi)

    def logical_and(self, a, b):
        return a & b

    def logical_not(self, a):
        return ~a

    def eq(self, a, b):
        return a == b

    def mean_over_joints(self, x):
        return x.mean(dim=-1)

    def to_float(self, x):
        return x.float()


# --- Primary objective -------------------------------------------------------------------------

# Bounded velocity tracking, the shape used by "Learning to Walk in Minutes" (Rudin et al.,
# arXiv:2109.11978) -- exp(-error^2/scale), max 1.0. Replaced a bare `reward = forward_vel` that
# had no ceiling and got exploited into a leap-and-crash strategy (velocity climbed 0 -> 5.3 m/s
# with no plateau, because "faster" was always better until the crash).
#
# SCALE was widened 0.25 -> 1.0 after measuring that 0.25 left NO usable gradient from a standstill:
# a stationary agent against a 1.5 m/s command scored exp(-9) = 0.0001, so the only positive term
# in the whole reward was unreachable until you were already moving ~1 m/s. Every other term being
# a penalty, that made total per-step reward negative and "terminate immediately" optimal -- seen
# in training as ep_len_mean pinned flat at 9.4 steps.
FORWARD_VEL_TRACKING_SCALE = 1.0

# Commanded speed is resampled per episode and every COMMAND_RESAMPLE_INTERVAL seconds, rather than
# fixed. A fixed target let a canned motion (one big lunge per cycle) specialize on a single number;
# a lunge tuned for 1.5 m/s badly misses a 0.5 m/s command, so randomizing forces a controllable
# gait. The command is part of the observation -- without seeing it the task is unsolvable.
COMMAND_MIN_SPEED = 0.5
COMMAND_MAX_SPEED = 2.0
COMMAND_RESAMPLE_INTERVAL = 5.0  # seconds

# Small per-step reward just for being alive. Everything except velocity tracking is a penalty, so
# without this the per-step total hovers near zero and any pose with slightly-negative net reward
# makes ending the episode the optimal move. Deliberately far below tracking's 1.0 ceiling so
# standing still never beats walking.
ALIVE_BONUS = 0.1


class RewardTerm:
    """One named component of the reward. `compute` returns its SIGNED contribution (penalties
    return negative), so the total reward is just the sum over all terms."""

    name: str = "unnamed"

    def compute(self, s: "WalkState"):
        raise NotImplementedError


class Alive(RewardTerm):
    name = "alive"

    def compute(self, s):
        return ALIVE_BONUS + 0.0 * s.forward_vel  # keeps batch shape on the GPU backend


class VelocityTracking(RewardTerm):
    """How closely forward speed matches the commanded speed. The main objective."""

    name = "velocity_tracking"

    def compute(self, s):
        return s.exp(-((s.target_speed - s.forward_vel) ** 2) / FORWARD_VEL_TRACKING_SCALE)


# --- Posture / balance -------------------------------------------------------------------------

# Weights below follow the paper's Appendix A.3 pattern (many small shaping terms under one main
# objective) but NOT its numbers -- those are sized for a quadruped and would swamp everything here.
# All are first-pass values, retuned empirically rather than trusted because they came from a paper.

LATERAL_VEL_WEIGHT = 0.1   # sideways drift; nothing else discourages it (tracking is x-only)
VERTICAL_VEL_WEIGHT = 0.1  # vertical bobbing

# Penalizes the chest's ANGLE from upright, every step. Without it, tracking only cares about
# x-velocity, so a controlled forward topple that converts gravity into speed scores the same as an
# upright gait right up until it crosses the fall threshold.
ORIENTATION_WEIGHT = 0.5

# Penalizes the chest's angular VELOCITY (thrashing), complementing the angle term above.
ANGULAR_VEL_WEIGHT = 0.05

# Center of mass drifting horizontally away from the support polygon (approximated as the midpoint
# between the feet). This is the textbook static-stability criterion and the one thing no other term
# measured: orientation says "stay upright", the fall check says "don't be on the ground", but
# neither says "keep your weight over your feet" -- and CoM outside the support base IS what losing
# balance means. Measured: ~0.001 standing, ~0.08 while genuinely toppling.
BALANCE_WEIGHT = 0.5


class LateralVelocityPenalty(RewardTerm):
    name = "lateral_velocity"

    def compute(self, s):
        return -LATERAL_VEL_WEIGHT * s.lateral_vel ** 2


class VerticalVelocityPenalty(RewardTerm):
    name = "vertical_velocity"

    def compute(self, s):
        return -VERTICAL_VEL_WEIGHT * s.vertical_vel ** 2


class OrientationPenalty(RewardTerm):
    """Measures the CHEST's true world-space tilt, not the root torso joint.

    It used to read the root pitch/roll DOF directly. Once waist_bend became actuated (it sits
    between the root and the chest), that stopped meaning what it says: measured on a real
    checkpoint, root pitch stayed within +/-30 degrees all episode -- comfortably "upright" -- while
    the chest was actually at -49.8 degrees, right at the fall threshold, via a steady ~-30 degree
    waist arch. The policy had found a posture the penalty structurally could not see.
    """

    name = "orientation"

    def compute(self, s):
        return -ORIENTATION_WEIGHT * (s.chest_pitch ** 2 + s.chest_roll ** 2)


class AngularVelocityPenalty(RewardTerm):
    """Chest angular velocity -- same blind spot as OrientationPenalty had, same fix."""

    name = "angular_velocity"

    def compute(self, s):
        return -ANGULAR_VEL_WEIGHT * s.chest_ang_vel_sq


class BalancePenalty(RewardTerm):
    name = "balance"

    def compute(self, s):
        return -BALANCE_WEIGHT * s.com_support_offset_sq


# --- Effort / smoothness -----------------------------------------------------------------------

# These four are MEANS over the actuated joints, not sums. As sums their magnitude scaled with
# however many joints happened to be in the action space, so going from legs-only (5) to the whole
# body (15) silently tripled their weight against a tracking reward still capped at 1.0 -- which
# directly fights the reason the arms were added, by taxing "having more limbs" rather than
# "moving them wastefully". Weights are 5x their original values (the joint count they were
# calibrated at) so a mean lands where that calibration intended.
ENERGY_COST_WEIGHT = 0.005     # 5 * 0.001; action^2 as a stand-in for real joint torque
JOINT_ACCEL_WEIGHT = 1e-8      # 5 * 2e-9; raw qacc here reaches 3,000-11,000 rad/s^2, so squared is
                               # ~1e7-1e8 -- an earlier guess of 1e-5 made this one term -140 to
                               # -1760 per step, swamping everything else by 3-4 orders of magnitude
JOINT_VEL_WEIGHT = 5e-3        # 5 * 1e-3
ACTION_RATE_WEIGHT = 0.25      # 5 * 0.05; penalizes jerky step-to-step control changes


class EnergyPenalty(RewardTerm):
    name = "energy"

    def compute(self, s):
        return -ENERGY_COST_WEIGHT * s.mean_over_joints(s.action ** 2)


class JointAccelerationPenalty(RewardTerm):
    name = "joint_acceleration"

    def compute(self, s):
        return -JOINT_ACCEL_WEIGHT * s.mean_over_joints(s.joint_accel ** 2)


class JointVelocityPenalty(RewardTerm):
    name = "joint_velocity"

    def compute(self, s):
        return -JOINT_VEL_WEIGHT * s.mean_over_joints(s.joint_vel ** 2)


class ActionRatePenalty(RewardTerm):
    name = "action_rate"

    def compute(self, s):
        return -ACTION_RATE_WEIGHT * s.mean_over_joints((s.action - s.prev_action) ** 2)


# --- Gait --------------------------------------------------------------------------------------

# A clock that cycles once every GAIT_CYCLE_TIME. The right foot is expected planted for the first
# half, the left for the second. Air time alone only shapes HOW LONG a foot is up, never WHICH foot
# should be down WHEN -- so a shuffle or an asymmetric hop scored as well as a real alternating
# walk. The policy sees sin/cos of this phase in its observation.
GAIT_CYCLE_TIME = 0.8
GAIT_PHASE_WEIGHT = 0.3  # per foot per step: + if contact matches its expected half-cycle, - if not

# Paid once per touchdown, for how close that foot's airborne time was to a normal swing.
# TARGET_AIR_TIME is derived from the clock rather than set independently: it used to be a flat 0.5s
# (a quadruped value) while the clock implied 0.4s, so a foot obeying the schedule perfectly was
# paid a small NEGATIVE bonus for doing exactly what the gait term asked.
FEET_AIR_TIME_WEIGHT = 1.0
TARGET_AIR_TIME = GAIT_CYCLE_TIME / 2
AIR_TIME_CAP = 1.0  # (air_time - target) is otherwise unbounded above, so a foot held up forever
                    # would pay out an ever-larger one-off reward on landing

# Growing penalty for a foot hanging up past a normal swing. The landing bonus only prices what a
# touchdown is worth; nothing stopped a foot from simply staying airborne. Measured on the first
# working policy: both feet off the ground 40.3% of all steps -- bounding, not walking (a human walk
# is ~0%). Jumping is still allowed; this only bites past a normal swing, so brief flight is free.
MAX_AIR_TIME = 0.45  # just above TARGET_AIR_TIME so an on-schedule swing never triggers it
AIR_TIME_EXCESS_WEIGHT = 2.0

# A planted foot sliding instead of gripping ("skating").
FOOT_SLIP_WEIGHT = 0.1


class GaitPhase(RewardTerm):
    name = "gait_phase"

    def compute(self, s):
        right_should_stand = s.phase < 0.5
        matched_r = s.eq(s.foot_touching["r"], right_should_stand)
        matched_l = s.eq(s.foot_touching["l"], s.logical_not(right_should_stand))
        return GAIT_PHASE_WEIGHT * (s.where(matched_r, 1.0, -1.0) + s.where(matched_l, 1.0, -1.0))


class FeetAirTime(RewardTerm):
    """Landing bonus plus the hanging-too-long penalty -- both are about the same quantity."""

    name = "feet_air_time"

    def compute(self, s):
        total = 0.0 * s.forward_vel
        for side in ("r", "l"):
            landed = s.logical_and(s.foot_touching[side], s.air_time_before[side] > 0.0)
            capped = s.clamp_max(s.air_time_before[side], AIR_TIME_CAP)
            total = total + s.where(landed, FEET_AIR_TIME_WEIGHT * (capped - TARGET_AIR_TIME), 0.0)
            excess = s.clamp_min(s.air_time[side] - MAX_AIR_TIME, 0.0)
            total = total - AIR_TIME_EXCESS_WEIGHT * excess
        return total


class FootSlip(RewardTerm):
    name = "foot_slip"

    def compute(self, s):
        total = 0.0 * s.forward_vel
        for side in ("r", "l"):
            total = total - s.where(s.foot_touching[side], FOOT_SLIP_WEIGHT * s.foot_slip_sq[side], 0.0)
        return total


# --- Termination -------------------------------------------------------------------------------

# A one-off cost for falling, so a bad pose ends the episode instead of quietly bleeding reward.
FALL_PENALTY = 20.0


class FallPenalty(RewardTerm):
    name = "fall"

    def compute(self, s):
        return -FALL_PENALTY * s.to_float(s.fallen)


# The full reward. Order here is the order they appear in TensorBoard.
REWARD_TERMS: list[RewardTerm] = [
    Alive(),
    VelocityTracking(),
    GaitPhase(),
    FeetAirTime(),
    FootSlip(),
    BalancePenalty(),
    OrientationPenalty(),
    AngularVelocityPenalty(),
    LateralVelocityPenalty(),
    VerticalVelocityPenalty(),
    EnergyPenalty(),
    JointAccelerationPenalty(),
    JointVelocityPenalty(),
    ActionRatePenalty(),
    FallPenalty(),
]


def total_reward(state: "WalkState"):
    """Sum of every term. Use breakdown() instead when you want to see which term is doing what."""
    total = 0.0 * state.forward_vel
    for term in REWARD_TERMS:
        total = total + term.compute(state)
    return total


def breakdown(state: "WalkState") -> dict:
    """Every term's individual contribution, by name -- for TensorBoard and for debugging.

    Worth having: two separate bugs this session (a foot-slip term quietly dominating at -0.42/step,
    and velocity tracking sitting at 0.0001 with no gradient) were both invisible in the total and
    obvious the moment the terms were listed separately.
    """
    return {term.name: term.compute(state) for term in REWARD_TERMS}
