"""The humanoid body, built from editable objects instead of one big f-string.

Each piece of the body is a small dataclass you can inspect, tweak, subclass, or swap out:

    Joint    one hinge/slide DOF        (name, axis, range, spring, damping)
    Geom     one collision/visual shape (capsule or sphere, size, mass, colour)
    Segment  one rigid body             (a position, its joints, its geoms, its children)
    Limb     a factory that builds a Segment tree (Arm, Leg, Torso, ...)

A Segment renders itself to MJCF recursively, so the whole skeleton is just nested objects and
you never touch XML strings to change the body.

To edit the body, change a Limb's fields:

    leg = Leg(side="r", y=0.11, facing=1.0, ankle_mode="single")
    leg.thigh_mass = 8.0            # heavier thigh
    leg.knee.range_deg = (150, 0)   # more knee flexion
    leg.shin_length = 0.36          # longer shin

To replace a limb wholesale, subclass it and override `build()` -- Humanoid takes whatever Limb
objects you hand it, so a different arm design drops straight in:

    class TentacleArm(Arm):
        def build(self): ...
    Humanoid(name="red", ..., arms=[TentacleArm("r", 0.19), TentacleArm("l", -0.19)])

Geometry, masses, ROM limits and gear values here are unchanged from the original hand-written
MJCF -- this is a restructuring, verified to produce a byte-identical model, not a redesign. See
fight_env.py for the empirical history behind the specific numbers (every gear value was swept
against its joint's real range, not guessed).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _fmt(v: float) -> str:
    """Match the original f-string formatting so generated MJCF stays identical."""
    return f"{v}"


@dataclass
class Joint:
    """One degree of freedom.

    `stiffness`/`springref` give a joint a passive spring pulling it toward a rest angle -- used
    for the arms/waist so they hold a guard stance. The torso's pitch/roll deliberately have NO
    spring: a real fall isn't undone by a spring, the policy has to catch itself, which is the
    whole point of the balance task.
    """

    name: str
    axis: str                       # e.g. "0 1 0"
    range: str | None = None        # e.g. "-30 60"; None = unlimited
    type: str = "hinge"             # "hinge" or "slide"
    stiffness: float | None = None
    springref: float | None = None
    damping: float | None = None
    armature: float | None = None

    def to_mjcf(self) -> str:
        parts = [f'<joint name="{self.name}" type="{self.type}" axis="{self.axis}"']
        if self.range is not None:
            parts.append(f'range="{self.range}"')
        for attr in ("stiffness", "springref", "damping", "armature"):
            val = getattr(self, attr)
            if val is not None:
                parts.append(f'{attr}="{_fmt(val)}"')
        return " ".join(parts) + " />"


@dataclass
class Geom:
    """One shape: collision, mass, and appearance.

    Feet carry a custom `friction` (higher than default) so they grip instead of skating.
    """

    type: str                       # "capsule" or "sphere"
    size: float
    mass: float
    rgba: str
    name: str | None = None
    fromto: str | None = None       # capsules
    pos: str | None = None          # spheres
    friction: str | None = None

    def to_mjcf(self) -> str:
        parts = ["<geom"]
        if self.name:
            parts.append(f'name="{self.name}"')
        parts.append(f'type="{self.type}"')
        if self.fromto is not None:
            parts.append(f'fromto="{self.fromto}"')
        if self.pos is not None:
            parts.append(f'pos="{self.pos}"')
        parts.append(f'size="{_fmt(self.size)}"')
        parts.append(f'mass="{_fmt(self.mass)}"')
        if self.friction is not None:
            parts.append(f'friction="{self.friction}"')
        parts.append(f'rgba="{self.rgba}"')
        return " ".join(parts) + " />"


@dataclass
class Segment:
    """One rigid body: where it sits, what DOFs connect it to its parent, its shapes, its children.

    This is the recursive unit -- a whole limb is just a Segment with children, and the entire
    humanoid is one root Segment.
    """

    name: str
    pos: str = "0 0 0"
    joints: list[Joint] = field(default_factory=list)
    geoms: list[Geom] = field(default_factory=list)
    children: list["Segment"] = field(default_factory=list)

    def to_mjcf(self, indent: int = 6) -> str:
        pad = " " * indent
        inner = " " * (indent + 2)
        lines = [f'{pad}<body name="{self.name}" pos="{self.pos}">']
        lines += [inner + j.to_mjcf() for j in self.joints]
        lines += [inner + g.to_mjcf() for g in self.geoms]
        lines += [c.to_mjcf(indent + 2) for c in self.children]
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    def walk(self):
        """Yield this segment and every descendant -- handy for inspecting or bulk-editing."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, name_suffix: str) -> "Segment | None":
        """First descendant whose name ends with `name_suffix` (e.g. "shin_r")."""
        return next((s for s in self.walk() if s.name.endswith(name_suffix)), None)


class Limb:
    """Builds one Segment tree. Subclass and override `build()` to replace a body part."""

    def build(self) -> Segment:
        raise NotImplementedError
