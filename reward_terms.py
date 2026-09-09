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
    action: object = None        # post-clamp -- what physics actually received
    prev_action: object = None
    raw_action: object = None    # pre-clamp -- what the policy commanded (see
                                 # ActionMagnitudePenalty; falls back to `action` if unset)

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
    knee_height: dict = field(default_factory=dict)     # side -> knee height above resting
    leg_lift_ema: dict = field(default_factory=dict)    # side -> EMA of that leg's knee lift
                                                       # (see GaitSymmetry)
    foot_height: dict = field(default_factory=dict)     # side -> foot height ABOVE its
                                                       # planted resting height (metres)

    # Joint positions and their authored limits, for JointLimitPenalty
    joint_pos: object = None
    joint_lower: object = None
    joint_upper: object = None

    fallen: object = False

    # --- backend operations (overridden per backend) ---
    def exp(self, x): raise NotImplementedError
    def where(self, cond, a, b): raise NotImplementedError
    def clamp_min(self, x, lo): raise NotImplementedError
    def clamp_max(self, x, hi): raise NotImplementedError
    def minimum(self, a, b): raise NotImplementedError
    def logical_and(self, a, b): raise NotImplementedError
    def logical_not(self, a): raise NotImplementedError
    def eq(self, a, b): raise NotImplementedError
    def mean_over_joints(self, x): raise NotImplementedError
    def sum_over_joints(self, x): raise NotImplementedError
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

    def minimum(self, a, b):
        import numpy as np
        return np.minimum(a, b)

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

    def sum_over_joints(self, x):
        import numpy as np
        return float(np.sum(x))

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

    def minimum(self, a, b):
        import torch
        return torch.minimum(a, torch.as_tensor(b, device=a.device, dtype=a.dtype))             if not isinstance(b, torch.Tensor) else torch.minimum(a, b)

    def logical_and(self, a, b):
        return a & b

    def logical_not(self, a):
        return ~a

    def eq(self, a, b):
        return a == b

    def mean_over_joints(self, x):
        return x.mean(dim=-1)

    def sum_over_joints(self, x):
        return x.sum(dim=-1)

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
# The command range now includes ZERO: "stand still" is a command the walker must be able to obey.
# It previously started at 0.5 m/s, so in 788M steps the policy was literally never once asked to
# stop -- it only ever knew how to keep moving. Standing is a genuinely different skill (hold a
# static double-support pose and reject disturbances, rather than cycle a gait), and it is the
# skill push-recovery is built on. Sampling zero as a discrete case rather than just lowering the
# minimum, because a uniform range would make true zero measure-zero and it would never actually
# be practised.
COMMAND_MIN_SPEED = 0.5
COMMAND_MAX_SPEED = 2.0
COMMAND_STAND_PROB = 0.3    # fraction of commands that are "stand still". Raised from 0.2
                            # because standing gets only this slice of the gradient signal,
                            # and it was measured collapsing (85% double-support at 2M steps
                            # down to 5% once walking was learned) rather than improving.
COMMAND_STAND_SPEED = 0.0
# Below this commanded speed the task is standing, not walking, and the gait terms switch off
# (see PeriodicGait). Slightly above zero so it is a clean test, not a float comparison.
STANDING_COMMAND_THRESHOLD = 0.05
COMMAND_RESAMPLE_INTERVAL = 5.0  # seconds

# --- Random pushes ------------------------------------------------------------------------------
# Every PUSH_INTERVAL seconds the torso gets a random horizontal velocity impulse. This is the
# standard robustness technique from legged_gym (`push_robots` / `max_push_vel_xy`) and it is what
# actually forces balance RECOVERY rather than a well-timed open-loop cycle: a policy that has only
# ever walked on flat ground with no disturbance never needs to learn how to catch itself.
# Applied as a velocity change rather than a force so its magnitude is interpretable and does not
# depend on the body's mass distribution.
# 3s against a 10s episode gives roughly 3 pushes per episode. At the previous 6.0 an episode
# saw about 1.6, which is a thin signal to learn recovery from -- most of every episode was
# undisturbed, so a policy could score well while never having to catch itself.
PUSH_INTERVAL = 3.0        # seconds between pushes
# Calibrated by measuring what a push actually does to a settled policy (single impulse, automatic
# pushes disabled), rather than picked by eye:
#     0.8 m/s -> 0.34 m of sideways displacement, recovered easily  (the old value; barely noticed)
#     2.0 m/s -> ~1-3 m, recovered
#     4.0 m/s -> knocked over sideways
# 2.0 sits where the disturbance is unmistakable but recovery is still possible, which is the point
# -- a push that always knocks it down teaches nothing except that falling is unavoidable, and the
# fall penalty would just swamp the episode. Sideways is the weaker axis (it falls to a 4.0 m/s
# sideways push but survives the same push from behind), which is what the hip_abduct joints and
# this term are together meant to fix.
PUSH_MAX_VEL = 2.0         # m/s, sampled uniformly in [-max, max] on x and y independently


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


# Penalizes what the policy COMMANDED, before the environment clamps it to [-1, 1]. Every other
# effort term sees the post-clamp action, which means commanding 24 costs exactly what commanding
# 1 costs -- nothing anywhere opposed the network's raw output growing without bound.
#
# It did exactly that. Measured across one training run's checkpoints:
#
#     step        mean|a|   saturated   actor out-layer |W|   log_std
#       2M          0.22        0.8%          0.045           +0.009
#     208M          6.10       70.2%          0.126           +0.209
#     936M         24.19       83.1%          0.377           +1.271
#
# A 110x growth in commanded magnitude, with 83% of joints pinned against the clamp permanently --
# a bang-bang controller with no proportional control left. It's self-reinforcing: once actions
# saturate, extra exploration noise stops changing behavior, so the entropy bonus can inflate
# log_std for free, which drives further saturation. The consequences were visible elsewhere and
# went undiagnosed for a long time: the policy couldn't respond to fine reward shaping (the
# air-time penalty moved nothing in 52M steps) and speed tracking was compressed to roughly one
# gait (commands 0.5-2.0 m/s produced only 0.93-1.59 m/s).
#
# The weight is a BARRIER, not a cost: near-free for a policy using its actuators sensibly, and
# expensive once the raw output starts drifting out of the usable range.
#
# Calibrating it is the whole game, and the first attempt (0.001) got it wrong by sizing against
# the ENDPOINT (|a| = 24, where it cost -0.58/step) rather than the DRIFT REGION where the growth
# actually happens. Measured over a real 300M-step run at 0.001: it slowed the runaway ~5x versus
# no barrier at all (mean|a| 2.11 vs 10.70 at matched steps, saturation 61% vs 80%) but never
# stopped it -- at |a| = 2.1 the penalty was only -0.008/step, about 1% of the ~0.84 velocity
# reward, so the policy correctly ignored it and kept climbing with no plateau.
#
# 0.01 puts the barrier where the drift is:
#     |a| = 1  ->  -0.01/step   (1% of the objective; a policy using full legal torque barely
#                                notices, which matters because this body's gear values were tuned
#                                so full torque just reaches each joint's real range)
#     |a| = 2  ->  -0.04
#     |a| = 3  ->  -0.09
#     |a| = 5  ->  -0.25
#
# Deliberately NOT larger: at w = 1.0, |a| = 1 would cost -1.00/step and cancel the entire
# velocity reward, pushing the policy to barely actuate at all (~30% of available torque).
#
# Confirmed unrecoverable after the fact: rescaling a saturated policy's output layer back into
# range destroyed its gait (949 -> 22 steps), so this has to be prevented during training rather
# than repaired later.
ACTION_SATURATION_WEIGHT = 0.01


class ActionMagnitudePenalty(RewardTerm):
    """Keeps the policy's raw output inside the range the environment can actually use."""

    name = "action_magnitude"

    def compute(self, s):
        raw = s.raw_action if s.raw_action is not None else s.action
        return -ACTION_SATURATION_WEIGHT * s.mean_over_joints(raw ** 2)


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
# Growing penalty for a foot hanging up past a normal swing. The landing bonus only prices what a
# touchdown is worth; nothing stopped a foot from simply staying airborne. Measured on the first
# working policy: both feet off the ground 40.3% of all steps -- bounding, not walking (a human walk
# is ~0%). Jumping is still allowed; this only bites past a normal swing, so brief flight is free.
MAX_AIR_TIME = 0.45  # just above TARGET_AIR_TIME so an on-schedule swing never triggers it
AIR_TIME_EXCESS_WEIGHT = 2.0
FEET_AIR_TIME_WEIGHT = 1.0
TARGET_AIR_TIME = GAIT_CYCLE_TIME / 2
# The landing bonus is capped at MAX_AIR_TIME, not at some larger independent value. It used to be
# 1.0s while MAX_AIR_TIME was 0.45s, which meant the bonus kept RISING (to +0.6) through the entire
# region the hang penalty was charging for -- the reward was simultaneously paying for and punishing
# the same behaviour between 0.45s and 1.0s. Deriving it removes that contradiction.
AIR_TIME_CAP = MAX_AIR_TIME


# A planted foot sliding instead of gripping ("skating").
FOOT_SLIP_WEIGHT = 0.1

# Rewards the SWING foot for actually getting up off the ground -- the standard foot-clearance term
# in legged-locomotion RL, and the piece this reward was missing entirely.
#
# Nothing here ever asked the walker to pick its feet up. It looked like it did, but only by
# accident: while any non-foot ground contact was an instant episode-ender, the agent was forced to
# hold its legs clear, which produced 0.256m of foot clearance and 80 degrees of knee range as a
# SIDE EFFECT of a termination rule. That rule was independently wrong (it killed upright agents --
# see walk_env.py's shin comment), and the moment it was relaxed the policy did exactly what the
# reward actually permitted: it dropped into a low scuff, dragging a shin on the floor 38% of all
# steps, riding just under the fall threshold, with knee range collapsing 82 -> 36 degrees and
# clearance 0.256 -> 0.045m at matched training steps.
#
# So clearance is stated as a reward instead of being implied by a death condition. Applied only
# while a foot is AIRBORNE, so it never fights the stance foot, and capped at the target so there
# is nothing to gain from flinging a leg unnaturally high.
FOOT_CLEARANCE_WEIGHT = 2.0    # first-pass, not tuned
FOOT_CLEARANCE_TARGET = 0.12   # metres; between a normal stride and the 0.256m the old strict
                               # rule was forcing. Foot bodies sit ~0.10m up when planted, so this
                               # is measured as height ABOVE that resting height, not absolute z.


# Periodic reward composition -- the formulation Siekmann et al. used to get clean bipedal gaits
# on Cassie ("Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition").
#
# The old GaitPhase was a crude version of this: a binary +/-0.3 for "is the right foot down during
# the first half of the cycle". It specified contact and nothing else, so a shuffle satisfied it as
# well as a stride, and it said nothing about what the swing foot should be doing.
#
# The proper form specifies, smoothly across the cycle, what each foot should be doing in BOTH
# phases, and does it entirely with penalties:
#   * during STANCE  the foot must not slide  -> penalise horizontal foot speed
#   * during SWING   the foot must not push   -> penalise ground contact
# Nothing here dictates joint angles or how the leg gets there; the positive drive stays with
# velocity tracking. That is the point -- it constrains the gait's *timing* without handing the
# policy a quantity it can maximise in a degenerate way, which is how the hand-added terms here
# kept getting gamed.
#
# DUTY FACTOR is what makes this a walk rather than a run. Each foot is in stance for this fraction
# of the cycle; the two windows are half a cycle apart, so any duty above 0.5 makes them OVERLAP,
# and that overlap is double support -- both feet down at once, which is the defining feature of
# walking. 0.6 gives ~20% double support. Drop it below 0.5 and the same reward asks for a run.
GAIT_DUTY_FACTOR = 0.6
GAIT_TRANSITION = 0.05          # cycle fraction spent blending between stance and swing
# Stated as a REWARD for correct timing, not a penalty for incorrect. That is the same function up
# to a constant, but the constant decides whether the agent wants to be alive: as pure penalties
# this term cost a standing agent -1.15/step, which made terminating immediately (-20 once) beat
# surviving 500 steps (-575) -- the exact failure that once pinned ep_len_mean at 9.4. Paying for
# what you want keeps the baseline non-negative, so a policy that cannot walk yet still prefers to
# keep trying rather than face-plant.
GAIT_STANCE_CONTACT_WEIGHT = 0.5  # paid while the foot is correctly planted during stance
GAIT_SWING_CONTACT_WEIGHT = 0.5   # paid while the foot is correctly clear during swing
# Slip stays a penalty (it is a continuous quantity with a genuine zero, not a binary
# right/wrong) but it must be BOUNDED. foot_slip_sq is a squared speed with no ceiling: at weight
# 1.0 an early policy whipping its feet around at a few m/s drove this single term to -9.4/step,
# swamping a reward whose other terms all live within +/-1. Same failure as the original
# JOINT_ACCEL_WEIGHT, which reached -1760/step before being measured. Capping it keeps the worst
# case comparable to what a correctly timed gait can earn.
GAIT_STANCE_SLIP_WEIGHT = 1.0
GAIT_SLIP_CAP = 0.25              # (0.5 m/s)^2 -- beyond this it is already maximally wrong


def _stance_weight(s, foot_phase):
    """Smooth 0..1 window: 1 while this foot should be in stance, 0 while it should swing.

    A trapezoid rather than a square wave -- a hard switch would put a discontinuity in the reward
    at the exact moment the policy is trying to time a footfall against it.
    """
    ramp_in = s.clamp_max(s.clamp_min(foot_phase / GAIT_TRANSITION, 0.0), 1.0)
    ramp_out = s.clamp_max(
        s.clamp_min((GAIT_DUTY_FACTOR + GAIT_TRANSITION - foot_phase) / GAIT_TRANSITION, 0.0), 1.0)
    return s.minimum(ramp_in, ramp_out)


class PeriodicGait(RewardTerm):
    """Constrains WHEN each foot may carry load and WHEN it may move, not how the leg gets there."""

    name = "periodic_gait"

    def compute(self, s):
        # When commanded to stand, the gait clock is the wrong thing to follow: stepping in place
        # is not standing still. Both feet are simply required to be down and quiet, which is the
        # same pair of quantities (contact, slip) scored against a different schedule.
        standing = s.target_speed < STANDING_COMMAND_THRESHOLD
        total = 0.0 * s.forward_vel
        for side, offset in (("r", 0.0), ("l", 0.5)):
            foot_phase = (s.phase + offset) % 1.0
            walk_stance = _stance_weight(s, foot_phase)
            # standing -> stance weight 1 for both feet, all the time
            stance = s.where(standing, 1.0, walk_stance)
            swing = 1.0 - stance
            contact = s.to_float(s.foot_touching[side])
            # Paid for doing the right thing at the right time: down during stance, clear
            # during swing. Max GAIT_STANCE_CONTACT_WEIGHT per foot for a correctly timed gait.
            total = total + GAIT_STANCE_CONTACT_WEIGHT * stance * contact
            total = total + GAIT_SWING_CONTACT_WEIGHT * swing * (1.0 - contact)
            slip = s.clamp_max(s.foot_slip_sq[side], GAIT_SLIP_CAP)
            total = total - GAIT_STANCE_SLIP_WEIGHT * stance * slip
        return total


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


class FootClearance(RewardTerm):
    """Pays a swing foot for its height above resting, up to FOOT_CLEARANCE_TARGET."""

    name = "foot_clearance"

    def compute(self, s):
        total = 0.0 * s.forward_vel
        for side in ("r", "l"):
            lifted = s.clamp_min(s.foot_height[side], 0.0)
            capped = s.clamp_max(lifted, FOOT_CLEARANCE_TARGET)
            # airborne only: a planted foot is at ~0 height anyway, but gating on contact keeps
            # this from quietly rewarding a stance foot that's on a bump or mid-transition.
            # Paid only while walking: lifting a foot is not something to reward when
            # the command is to stand still.
            lift_pay = s.where(s.foot_touching[side], 0.0, FOOT_CLEARANCE_WEIGHT * capped)
            total = total + s.where(s.target_speed < STANDING_COMMAND_THRESHOLD,
                                    0.0, lift_pay)
        return total


# Rewards lifting the KNEE, which is the part of a stride the foot-clearance term can't ask for.
#
# FootClearance prices sole height and nothing else, and there are two ways to raise a sole: swing
# the thigh forward at the hip (moves thigh+shin+foot) or fold the shin back at the knee (moves
# shin+foot only). The knee route is far cheaper under the joint-motion penalties and earns exactly
# the same clearance reward, so the policy took it -- measured, the sole lift correlates +0.72 with
# knee angle and -0.29 with hip angle, hip ROM sat at ~42 deg while knee ROM ran to ~99, and 74M
# further steps changed none of it. The result looks like a hamstring curl rather than a stride, and
# it limits how far the leg travels forward, because forward travel is hip work.
#
# Knee height has no cheap substitute: raising it requires hip flexion. Same shape as
# FootClearance -- swing leg only, capped, measured relative to the resting pose.
KNEE_LIFT_WEIGHT = 2.0     # first-pass, not tuned
KNEE_LIFT_TARGET = 0.10    # metres above the knee's resting height


class KneeLift(RewardTerm):
    """Pays the swing leg for raising its knee -- i.e. for actually using the hip."""

    name = "knee_lift"

    def compute(self, s):
        total = 0.0 * s.forward_vel
        for side in ("r", "l"):
            lifted = s.clamp_min(s.knee_height[side], 0.0)
            capped = s.clamp_max(lifted, KNEE_LIFT_TARGET)
            # Paid only while walking: lifting a foot is not something to reward when
            # the command is to stand still.
            lift_pay = s.where(s.foot_touching[side], 0.0, KNEE_LIFT_WEIGHT * capped)
            total = total + s.where(s.target_speed < STANDING_COMMAND_THRESHOLD,
                                    0.0, lift_pay)
        return total


# Penalises one leg doing the other's work.
#
# Every term added so far prices a TOTAL and gets satisfied by a lopsided extreme. FootClearance
# priced sole height without caring which joint raised it, so the policy folded the knee instead of
# using the hip. KneeLift then priced knee height without caring which LEG raised it, so the policy
# put everything into one leg: measured, hip_l swung through 119.6 deg while hip_r sat at +29.2 deg
# mean (pinned behind the body near its +25 limit), left knee lifting 0.098m on average while the
# right averaged -0.013m. It still collected +0.20/step of knee_lift -- half the theoretical max --
# entirely from one side. A genuine one-legged lunge that scores well.
#
# The measurement has to be over TIME, not instantaneous. A walking gait is supposed to be
# instantaneously asymmetric (one leg swings while the other stands); what is wrong is a sustained
# imbalance across many cycles. So each leg keeps an exponential moving average of its own lift and
# the penalty is on the difference between those averages -- normal alternation cancels out over a
# stride, a limp does not.
LEG_BALANCE_EMA_ALPHA = 0.005   # ~200-step memory, several 0.8s gait cycles
LEG_BALANCE_WEIGHT = 20.0       # sized so the observed 0.1m imbalance costs ~0.2/step, roughly
                                # cancelling what one-legged lifting was earning. First pass.


class GaitSymmetry(RewardTerm):
    """Penalises a sustained difference between what the two legs contribute."""

    name = "gait_symmetry"

    def compute(self, s):
        diff = s.leg_lift_ema["r"] - s.leg_lift_ema["l"]
        return -LEG_BALANCE_WEIGHT * diff ** 2


class FootSlip(RewardTerm):
    name = "foot_slip"

    def compute(self, s):
        total = 0.0 * s.forward_vel
        for side in ("r", "l"):
            total = total - s.where(s.foot_touching[side], FOOT_SLIP_WEIGHT * s.foot_slip_sq[side], 0.0)
        return total


# Penalises any joint pushed OUTSIDE its authored anatomical range.
#
# The body was built around real human range-of-motion limits, but MuJoCo joint limits are soft
# constraints and the gear values were deliberately tuned so max torque just *reaches* each limit --
# so sustained torque presses straight through. Measured on a real policy: hip +38 deg on a joint
# that stops at +25, knee -16.8 on one that stops at 0, ankle -44.0 on one that stops at -20 (more
# than double its legal travel), and getting worse with training, not better.
#
# Stiffening the solver does almost nothing (16.87 -> 16.23 deg of overshoot across every
# solref/solimp setting tried). Cutting gear works (16.87 -> 3.50 at x0.25) but would cripple the
# walker, since those gear values are what let it reach its range at all. So the limit is enforced
# by the reward instead: the policy gets a gradient telling it to stay inside its own anatomy,
# without weakening the actuators.
#
# Summed, not averaged, over joints: each joint out of range is independently wrong, and having
# three of them out at once should cost three times as much.
JOINT_LIMIT_WEIGHT = 2.0   # ~0.15/step for one joint 15 deg over; ~0.4 for three at once


class JointLimitPenalty(RewardTerm):
    """Keeps the walker inside the range of motion its body was actually modelled with."""

    name = "joint_limit"

    def compute(self, s):
        over = s.clamp_min(s.joint_pos - s.joint_upper, 0.0)
        under = s.clamp_min(s.joint_lower - s.joint_pos, 0.0)
        return -JOINT_LIMIT_WEIGHT * s.sum_over_joints(over ** 2 + under ** 2)


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
    PeriodicGait(),
    FootClearance(),
    KneeLift(),
    GaitSymmetry(),
    BalancePenalty(),
    OrientationPenalty(),
    AngularVelocityPenalty(),
    LateralVelocityPenalty(),
    VerticalVelocityPenalty(),
    EnergyPenalty(),
    JointAccelerationPenalty(),
    JointVelocityPenalty(),
    ActionRatePenalty(),
    ActionMagnitudePenalty(),
    JointLimitPenalty(),
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
