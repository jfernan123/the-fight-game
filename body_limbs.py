"""The humanoid's actual limbs, built from the primitives in body_parts.py.

This is the file to edit when you want to change the body. Every limb is a dataclass whose fields
are its real dimensions, so changing the walker is changing a number here -- no XML:

    leg = Leg("red", "r", y=0.11, facing=1.0, body_rgba=..., ankle_mode="single")
    leg.thigh_mass = 8.0        # heavier thigh
    leg.shin_length = 0.36      # longer shin

To replace a limb entirely, subclass it and override build(); Humanoid accepts whatever limbs you
give it, so a different arm design drops straight in without touching anything else.

Geometry, masses, ROM limits and spring/damping values are carried over unchanged from the original
hand-written MJCF -- this is a restructuring, verified against a captured reference model, not a
redesign. See fight_env.py for the empirical history behind the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from body_parts import Geom, Joint, Limb, Segment

# --- Anatomy constants -------------------------------------------------------------------------
# Real human range-of-motion limits (degrees) as (flexion_max, extension_max). Flexion (forward/up:
# raising an arm, bending a knee) and extension (backward) are deliberately asymmetric, because
# real joints are.
ROM = {
    "shoulder": (170.0, 50.0),
    "elbow": (145.0, 0.0),
    "hip": (120.0, 25.0),
    "knee": (145.0, 0.0),
    "waist_bend": (60.0, 30.0),   # trunk flexes forward much further than it extends back
    "ankle": (20.0, 50.0),        # dorsiflexion vs plantarflexion (toe-off)
}

# Which way each joint's angle runs relative to `facing`. The knee is flipped because it bends
# OPPOSITE the hip: a bent knee folds the shin back so the foot stays under the body. Same-sign
# would swing the leg in one continuous arc -- literally what "inverted knees" looks like.
RANGE_SIGN = {"shoulder": 1.0, "elbow": 1.0, "hip": 1.0, "knee": -1.0, "ankle": 1.0}

# Symmetric ranges (no facing mirror needed).
ANKLE_ROLL_MAX = 20.0        # foot inversion/eversion; only built for ankle_mode "dual"/"detailed"
SHOULDER_ABDUCT_MAX = 100.0  # arm out to the side -- the motion humans actually balance with
HIP_TWIST_MAX = 60.0         # pelvis yaw
WAIST_TWIST_MAX = 45.0       # chest yaw relative to pelvis (hip-shoulder separation)

STAND_HEIGHT = 0.95


def signed_range(facing: float, flex_max: float, extend_max: float) -> str:
    """MJCF range string, mirrored by which way the body faces."""
    if facing > 0:
        return f"{-flex_max} {extend_max}"
    return f"{-extend_max} {flex_max}"


def rest_pose_deg(facing: float) -> dict[str, float]:
    """Guard-stance angles the passive springs hold each limb toward.

    The elbow carries the SAME sign as the shoulder (a bicep curl keeps rotating the same way to
    bring the fist up). The knee carries the OPPOSITE sign from the hip -- see RANGE_SIGN.
    """
    lean = 25.0 * facing
    hip_lean = 10.0 * facing
    elbow_bend = -90.0 * facing
    knee_bend = 15.0 * facing
    return {
        "shoulder_r": -lean, "shoulder_l": -lean,
        "elbow_r": elbow_bend, "elbow_l": elbow_bend,
        "hip_r": -hip_lean, "hip_l": -hip_lean,
        "knee_r": knee_bend, "knee_l": knee_bend,
    }


# --- Limbs -------------------------------------------------------------------------------------


@dataclass
class Arm(Limb):
    """Upper arm -> forearm -> glove.

    The shoulder has two axes: flexion/extension (forward-back swing, the punch/arm-swing) and
    abduction/adduction (out to the side). The second was missing originally, so the arm could only
    move in one plane -- and that plane is not the one humans use to catch their balance.
    """

    prefix: str          # fighter name, e.g. "red"
    side: str            # "r" or "l"
    y: float             # sideways offset from the chest centre
    facing: float
    body_rgba: str
    glove_rgba: str

    shoulder_pos_z: float = 0.22
    upperarm_length: float = 0.24
    upperarm_radius: float = 0.06
    upperarm_mass: float = 3.5
    forearm_length: float = 0.22
    forearm_radius: float = 0.05
    forearm_mass: float = 2.4
    glove_offset: float = -0.26
    glove_radius: float = 0.09
    glove_mass: float = 1.3

    def build(self) -> Segment:
        p, s = self.prefix, self.side
        rest = rest_pose_deg(self.facing)
        forearm = Segment(
            name=f"{p}_forearm_{s}",
            pos=f"0 0 -{self.upperarm_length}",
            joints=[Joint(
                name=f"{p}_elbow_{s}", axis="0 1 0",
                range=signed_range(self.facing * RANGE_SIGN["elbow"], *ROM["elbow"]),
                stiffness=150, springref=rest[f"elbow_{s}"], damping=15, armature=0.06,
            )],
            geoms=[
                Geom(type="capsule", fromto=f"0 0 0 0 0 -{self.forearm_length}",
                     size=self.forearm_radius, mass=self.forearm_mass, rgba=self.body_rgba),
                Geom(name=f"{p}_glove_{s}", type="sphere", pos=f"0 0 {self.glove_offset}",
                     size=self.glove_radius, mass=self.glove_mass, rgba=self.glove_rgba),
            ],
        )
        return Segment(
            name=f"{p}_upperarm_{s}",
            pos=f"0 {self.y} {self.shoulder_pos_z}",
            joints=[
                Joint(name=f"{p}_shoulder_{s}", axis="0 1 0",
                      range=signed_range(self.facing * RANGE_SIGN["shoulder"], *ROM["shoulder"]),
                      stiffness=150, springref=rest[f"shoulder_{s}"], damping=15, armature=0.06),
                Joint(name=f"{p}_shoulder_abduct_{s}", axis="1 0 0",
                      range=f"-{SHOULDER_ABDUCT_MAX} {SHOULDER_ABDUCT_MAX}",
                      stiffness=100, springref=0, damping=10, armature=0.05),
            ],
            geoms=[Geom(type="capsule", fromto=f"0 0 0 0 0 -{self.upperarm_length}",
                        size=self.upperarm_radius, mass=self.upperarm_mass, rgba=self.body_rgba)],
            children=[forearm],
        )


@dataclass
class Leg(Limb):
    """Thigh -> shin -> foot, with an optional real ankle.

    `ankle_mode` decides how the foot attaches:
      None       - foot welded rigidly to the shin, no ankle DOF at all (the original body)
      "single"   - one sagittal hinge (dorsi/plantarflexion), the usual choice for RL locomotion
      "dual"     - adds inversion/eversion, mirroring how real bipeds (Cassie, Atlas, Digit) use
                   the ankle for fast small balance corrections before moving the hips
      "detailed" - same 2-DOF ankle, but the foot splits into separate heel and toe contact geoms
                   so it can roll heel-to-toe through a stride instead of landing as one blob
    """

    prefix: str
    side: str
    y: float
    facing: float
    body_rgba: str
    ankle_mode: str | None = None

    hip_pos_z: float = -0.22
    thigh_length: float = 0.32
    thigh_radius: float = 0.075
    thigh_mass: float = 6.9
    shin_length: float = 0.32
    shin_radius: float = 0.06
    shin_mass: float = 5.2
    foot_radius: float = 0.05
    foot_mass: float = 1.7
    foot_friction: str = "1.5 0.005 0.0001"   # grippier than default so feet don't skate

    @property
    def foot_x(self) -> float:
        """How far the foot sticks out in front, mirrored by facing."""
        return 0.14 * self.facing

    def _ankle_joints(self) -> list[Joint]:
        p, s = self.prefix, self.side
        pitch_range = signed_range(self.facing * RANGE_SIGN["ankle"], *ROM["ankle"])
        common = dict(stiffness=100, springref=0, damping=10, armature=0.05)
        if self.ankle_mode == "single":
            return [Joint(name=f"{p}_ankle_{s}", axis="0 1 0", range=pitch_range, **common)]
        return [
            Joint(name=f"{p}_ankle_pitch_{s}", axis="0 1 0", range=pitch_range, **common),
            Joint(name=f"{p}_ankle_roll_{s}", axis="1 0 0",
                  range=f"-{ANKLE_ROLL_MAX} {ANKLE_ROLL_MAX}", **common),
        ]

    def _foot_geoms(self) -> list[Geom]:
        p, s = self.prefix, self.side
        if self.ankle_mode == "detailed":
            return [
                Geom(name=f"{p}_heel_{s}", type="capsule",
                     fromto=f"0 0 0 {-0.06 * self.facing} 0 -0.02", size=0.045, mass=0.6,
                     friction=self.foot_friction, rgba=self.body_rgba),
                Geom(name=f"{p}_toe_{s}", type="capsule",
                     fromto=f"0 0 0 {self.foot_x} 0 -0.04", size=0.045, mass=1.1,
                     friction=self.foot_friction, rgba=self.body_rgba),
            ]
        return [Geom(name=f"{p}_foot_{s}", type="capsule",
                     fromto=f"0 0 0 {self.foot_x} 0 -0.04", size=self.foot_radius,
                     mass=self.foot_mass, friction=self.foot_friction, rgba=self.body_rgba)]

    def build(self) -> Segment:
        p, s = self.prefix, self.side
        rest = rest_pose_deg(self.facing)

        shin_geoms = [Geom(type="capsule", fromto=f"0 0 0 0 0 -{self.shin_length}",
                           size=self.shin_radius, mass=self.shin_mass, rgba=self.body_rgba)]
        shin_children: list[Segment] = []
        if self.ankle_mode is None:
            # No ankle DOF: the foot is just another geom welded onto the shin.
            shin_geoms.append(Geom(
                name=f"{p}_foot_{s}", type="capsule",
                fromto=f"0 0 -{self.shin_length} {self.foot_x} 0 -0.36",
                size=self.foot_radius, mass=self.foot_mass,
                friction=self.foot_friction, rgba=self.body_rgba))
        else:
            shin_children.append(Segment(
                name=f"{p}_ankle_{s}", pos=f"0 0 -{self.shin_length}",
                joints=self._ankle_joints(), geoms=self._foot_geoms()))

        shin = Segment(
            name=f"{p}_shin_{s}", pos=f"0 0 -{self.thigh_length}",
            joints=[Joint(name=f"{p}_knee_{s}", axis="0 1 0",
                          range=signed_range(self.facing * RANGE_SIGN["knee"], *ROM["knee"]),
                          stiffness=800, springref=rest[f"knee_{s}"], damping=180, armature=0.1)],
            geoms=shin_geoms, children=shin_children,
        )
        return Segment(
            name=f"{p}_thigh_{s}", pos=f"0 {self.y} {self.hip_pos_z}",
            joints=[Joint(name=f"{p}_hip_{s}", axis="0 1 0",
                          range=signed_range(self.facing * RANGE_SIGN["hip"], *ROM["hip"]),
                          stiffness=800, springref=rest[f"hip_{s}"], damping=180, armature=0.1)],
            geoms=[Geom(type="capsule", fromto=f"0 0 0 0 0 -{self.thigh_length}",
                        size=self.thigh_radius, mass=self.thigh_mass, rgba=self.body_rgba)],
            children=[shin],
        )


@dataclass
class Chest(Limb):
    """Chest, neck and head, hanging off the pelvis through the waist.

    The waist is a separate 2-DOF joint (twist + bend) rather than the torso being one rigid block,
    which is what lets the pelvis lead and the chest follow -- real hip-shoulder separation. Note
    this means the chest's world orientation is NOT the same as the root torso's: anything checking
    "is it upright" has to measure the chest, not the root joint (see reward_terms.OrientationPenalty).
    """

    prefix: str
    facing: float
    body_rgba: str
    accent_rgba: str
    arms: list[Arm] = field(default_factory=list)

    pos_z: float = -0.03
    head_radius: float = 0.115
    head_mass: float = 4.8

    def build(self) -> Segment:
        p = self.prefix
        return Segment(
            name=f"{p}_chest", pos=f"0 0 {self.pos_z}",
            joints=[
                Joint(name=f"{p}_waist_twist", axis="0 0 1",
                      range=f"-{WAIST_TWIST_MAX} {WAIST_TWIST_MAX}",
                      stiffness=200, springref=0, damping=20, armature=0.08),
                Joint(name=f"{p}_waist_bend", axis="0 1 0",
                      range=signed_range(-self.facing, *ROM["waist_bend"]),
                      stiffness=250, springref=0, damping=25, armature=0.08),
            ],
            geoms=[
                Geom(name=f"{p}_waist", type="capsule", fromto="0 -0.09 0 0 0.09 0",
                     size=0.075, mass=3.2, rgba=self.body_rgba),
                Geom(name=f"{p}_spine_up", type="capsule", fromto="0 0 0 0 0 0.22",
                     size=0.1, mass=4.3, rgba=self.body_rgba),
                Geom(name=f"{p}_chest_geom", type="capsule", fromto="0 -0.17 0.22 0 0.17 0.22",
                     size=0.09, mass=5.4, rgba=self.body_rgba),
                Geom(name=f"{p}_neck", type="capsule", fromto="0 0 0.22 0 0 0.31",
                     size=0.05, mass=0.9, rgba=self.body_rgba),
                Geom(name=f"{p}_head", type="sphere", pos="0 0 0.43",
                     size=self.head_radius, mass=self.head_mass, rgba=self.accent_rgba),
            ],
            children=[arm.build() for arm in self.arms],
        )


@dataclass
class Humanoid(Limb):
    """The whole fighter: a free-floating root (pelvis) carrying a chest and two legs.

    `free_torso` adds pitch and roll hinges to the root, giving it real tip-over physics instead of
    being mechanically locked upright. Unlike every other joint these have NO spring: a real fall
    isn't undone by a passive restoring force, the policy has to actively catch itself -- which is
    the entire point of the balance task. Off by default so the fighter model is unchanged.
    """

    name: str
    body_rgba: str
    accent_rgba: str
    glove_rgba: str
    facing: float
    start_x: float
    free_torso: bool = False
    ankle_mode: str | None = None
    y_offset: float = 0.0

    arm_y: float = 0.19
    leg_y: float = 0.11

    def make_arms(self) -> list[Arm]:
        """Override to swap in a different arm design."""
        return [Arm(self.name, s, y, self.facing, self.body_rgba, self.glove_rgba)
                for s, y in (("r", self.arm_y), ("l", -self.arm_y))]

    def make_legs(self) -> list[Leg]:
        """Override to swap in a different leg design."""
        return [Leg(self.name, s, y, self.facing, self.body_rgba, self.ankle_mode)
                for s, y in (("r", self.leg_y), ("l", -self.leg_y))]

    def build(self) -> Segment:
        n = self.name
        root_joints = [
            Joint(name=f"{n}_x", type="slide", axis="1 0 0", damping=5),
            Joint(name=f"{n}_y", type="slide", axis="0 1 0", damping=5),
            Joint(name=f"{n}_z", type="slide", axis="0 0 1"),
            Joint(name=f"{n}_hip_twist", axis="0 0 1",
                  range=f"-{HIP_TWIST_MAX} {HIP_TWIST_MAX}",
                  stiffness=300, springref=0, damping=30, armature=0.1),
        ]
        if self.free_torso:
            root_joints += [
                Joint(name=f"{n}_pitch", axis="0 1 0", range="-80 80", damping=8, armature=0.05),
                Joint(name=f"{n}_roll", axis="1 0 0", range="-80 80", damping=8, armature=0.05),
            ]
        chest = Chest(n, self.facing, self.body_rgba, self.accent_rgba, arms=self.make_arms())
        return Segment(
            name=n, pos=f"{self.start_x} {self.y_offset} {STAND_HEIGHT}",
            joints=root_joints,
            geoms=[
                Geom(name=f"{n}_pelvis", type="capsule", fromto="0 -0.12 -0.22 0 0.12 -0.22",
                     size=0.09, mass=6.5, rgba=self.body_rgba),
                Geom(name=f"{n}_spine_low", type="capsule", fromto="0 0 -0.22 0 0 -0.03",
                     size=0.085, mass=3.2, rgba=self.body_rgba),
            ],
            children=[chest.build()] + [leg.build() for leg in self.make_legs()],
        )
