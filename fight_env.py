"""Minimal two-agent MuJoCo fighting environment and visual demo."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from body_limbs import Humanoid


# The body's anatomy -- ROM limits, joint ranges, limb dimensions -- now lives in body_limbs.py,
# where each limb is an editable object. Re-exported here so existing importers (walk_env.py,
# watch_all.py, the explore tooling below) keep working unchanged.
from body_limbs import (  # noqa: E402
    ANKLE_ROLL_MAX, HIP_TWIST_MAX, RANGE_SIGN, ROM, SHOULDER_ABDUCT_MAX, STAND_HEIGHT,
    WAIST_TWIST_MAX, rest_pose_deg as _rest_pose_deg, signed_range as _signed_range,
)

# +1 for a fighter whose front points toward +x, -1 toward -x, so both lean toward each other.
# The four ankle_mode names are for watch_all.py's side-by-side comparison scene -- they all face
# +x and share the same start_x (0.0) since they don't face off, just walk forward in parallel
# lanes differentiated by y_offset instead.
FACING = {"red": 1.0, "blue": -1.0, "baseline": 1.0, "single": 1.0, "dual": 1.0, "detailed": 1.0}
START_X = {"red": -2.0, "blue": 2.0, "baseline": 0.0, "single": 0.0, "dual": 0.0, "detailed": 0.0}


def _set_rest_pose_qpos(model: "mujoco.MjModel", data: "mujoco.MjData", name: str) -> None:
    """Start each hinge joint's qpos at its guard-stance rest angle instead of straight-limbed.

    Any controller driving these joints toward that same rest angle (a passive spring, or an
    explore-mode position actuator) needs qpos to already be there too, or the very first step
    has a huge gap to close -- with the strong gains involved, that snap is violent enough to
    launch the whole character (confirmed: an untouched-qpos start gave the torso a 24 m/s
    instantaneous velocity spike).
    """
    for joint_name, rest_deg in _rest_pose_deg(FACING[name]).items():
        qpos_adr = model.joint(f"{name}_{joint_name}").qposadr[0]
        data.qpos[qpos_adr] = np.radians(rest_deg)


def _humanoid_body(
    name: str, body_rgba: str, accent_rgba: str, glove_rgba: str,
    free_torso: bool = False, ankle_mode: str | None = None, y_offset: float = 0.0,
) -> str:
    """Build the MJCF for one humanoid fighter: torso, head, two legs, two arms, boxing gloves.

    The body itself is defined as editable limb objects in body_limbs.py -- this function just
    assembles them and renders the MJCF. To change the body (limb lengths, masses, joint ranges,
    or a whole limb design), edit body_limbs.py; nothing here needs to change.

    `free_torso` adds pitch/roll hinges to the root so the torso can actually tip over instead of
    being mechanically locked upright. Off by default so the fighter combat model is unchanged --
    it deliberately keeps fighters from falling from punches, which is a separate design decision
    from the walker's balance task.

    `ankle_mode` selects the foot design (None / "single" / "dual" / "detailed") -- see
    body_limbs.Leg for what each one physically builds.

    `y_offset` shifts the whole body sideways from its authored spawn; zero for the normal
    single-scene callers, used by watch_all.py to lay several agents out in parallel lanes.
    """
    return Humanoid(
        name=name,
        body_rgba=body_rgba,
        accent_rgba=accent_rgba,
        glove_rgba=glove_rgba,
        facing=FACING[name],
        start_x=START_X[name],
        free_torso=free_torso,
        ankle_mode=ankle_mode,
        y_offset=y_offset,
    ).build().to_mjcf(indent=4)


def _actuators(name: str, ankle_mode: str | None = None) -> str:
    """Punch-animation motors used by the scripted fight demo: torque-driven arms, plus separate
    hip and waist rotation so the pelvis can lead and the torso can follow -- hip-shoulder
    separation, not the whole body pivoting as one rigid block.

    Also includes leg torque motors (hip/knee) for locomotion training (walk_gym.py) -- inert
    for the scripted demo since main() never sets their ctrl, so the passive hip/knee springs
    (stiffness 800) keep holding the guard stance exactly as before this was added.

    Gear values were audited and tuned empirically for every joint, not guessed -- the original
    values (15 for shoulder/elbow, 130/90/110 for hip_twist/waist_twist/waist_bend, 250/200 for
    hip/knee) were tuned only for how they looked under the scripted demo's smooth sine-wave
    punch pulse, and turned out to reach barely any of each joint's actual authored range()
    limit when tested at sustained max torque -- shoulder/elbow were the worst, reaching only
    ~3-5% of their range at gear=15. For each joint the same empirical process was used: push
    gear up until sustained max-torque control reaches close to the joint's real mechanical
    limit (checked via jnt_range, both directions), while avoiding gear high enough to produce
    an unphysical instantaneous velocity spike (a joint snapping to 1000+ deg/s from a single
    control input in one step) or -- observed once with shoulder around gear=650 -- a numerical
    instability where more gear paradoxically stopped the joint from reaching the limit at all.
    Final values: shoulder=500 (was 15), elbow=250 (was 15), hip_twist=300 (was 130),
    waist_twist=250 (was 90), waist_bend=300 (was 110), hip=1500 (was 250), knee=1200 (was 200).
    This changes how the scripted two-fighter demo (main(), below) looks -- same sine-wave punch
    signal now swings arms through much more of their real range, not a small stylized jab.
    """
    lines = []
    for side in ("r", "l"):
        lines.append(f'    <motor name="{name}_shoulder_{side}_m" joint="{name}_shoulder_{side}" gear="500" ctrlrange="-1 1" />')
        # Unverified first-pass gear, same treatment as the ankle motors -- not yet empirically
        # swept the way shoulder/elbow/hip/knee above were.
        lines.append(f'    <motor name="{name}_shoulder_abduct_{side}_m" joint="{name}_shoulder_abduct_{side}" gear="200" ctrlrange="-1 1" />')
        lines.append(f'    <motor name="{name}_elbow_{side}_m" joint="{name}_elbow_{side}" gear="250" ctrlrange="-1 1" />')
    lines.append(f'    <motor name="{name}_hip_twist_m" joint="{name}_hip_twist" gear="300" ctrlrange="-1 1" />')
    lines.append(f'    <motor name="{name}_waist_twist_m" joint="{name}_waist_twist" gear="250" ctrlrange="-1 1" />')
    lines.append(f'    <motor name="{name}_waist_bend_m" joint="{name}_waist_bend" gear="300" ctrlrange="-1 1" />')
    for side in ("r", "l"):
        lines.append(f'    <motor name="{name}_hip_{side}_m" joint="{name}_hip_{side}" gear="1500" ctrlrange="-1 1" />')
        # Unverified first-pass gear, same treatment as the ankle/shoulder_abduct motors -- not yet
        # swept the way hip/knee were. Weaker than hip flexion (1500) because abduction is a
        # smaller muscle group and only needs to shift weight, not drive the stride.
        lines.append(f'    <motor name="{name}_hip_abduct_{side}_m" joint="{name}_hip_abduct_{side}" gear="600" ctrlrange="-1 1" />')
        lines.append(f'    <motor name="{name}_knee_{side}_m" joint="{name}_knee_{side}" gear="1200" ctrlrange="-1 1" />')
    # Ankle gear is an unverified first-pass guess (300), not yet swept the way every other joint
    # above was (see docstring) -- expect this to need the same empirical audit before trusting it.
    if ankle_mode == "single":
        for side in ("r", "l"):
            lines.append(f'    <motor name="{name}_ankle_{side}_m" joint="{name}_ankle_{side}" gear="300" ctrlrange="-1 1" />')
    elif ankle_mode in ("dual", "detailed"):
        for side in ("r", "l"):
            lines.append(f'    <motor name="{name}_ankle_pitch_{side}_m" joint="{name}_ankle_pitch_{side}" gear="300" ctrlrange="-1 1" />')
            lines.append(f'    <motor name="{name}_ankle_roll_{side}_m" joint="{name}_ankle_roll_{side}" gear="200" ctrlrange="-1 1" />')
    return "\n".join(lines)


def _explore_actuators(name: str, ankle_mode: str | None = None) -> str:
    """Position-controlled sliders for every limb joint, used by explore mode.

    Unlike the fight demo's torque motors, a position actuator's ctrl value directly commands a
    target angle, which is what makes a GUI slider intuitive: drag it to a degree, the joint goes
    there and holds. `kp` must be strong enough to overpower that joint's passive guard-stance
    spring (stiffness 150 for shoulder/elbow, 800 for hip/knee).

    Unlike a joint's `range` attribute, an actuator's `ctrlrange` is NOT auto-converted from the
    compiler's degree-authoring convention -- it's taken as radians literally. `_signed_range`
    (built for joint `range`) would leave this off by a factor of ~57, so the degrees are
    converted to radians here before formatting.

    `ankle_mode` is accepted (so this has the same signature as _actuators for _build_model_xml's
    call site) but not yet wired up to add ankle sliders -- walk_env.py's own explore() tool
    (torque motors, not these position sliders) is what's used to compare ankle_mode variants.
    """
    facing = FACING[name]
    # Weaker than you'd expect from the joint's own stiffness (800) -- a stronger kp could
    # brute-force a heavy leg through its full range fast enough to bounce the whole torso
    # (confirmed: at kp=5000 the torso visibly bounced by ~1m of vertical displacement when a
    # knee slider snapped to its extreme; 1800 cut that to ~0.1m while still moving substantially).
    kp = {"shoulder": 1200, "elbow": 1200, "hip": 1800, "knee": 1800}
    lines = []
    for side in ("r", "l"):
        for part in ("shoulder", "elbow", "hip", "knee"):
            joint = f"{name}_{part}_{side}"
            deg_lo, deg_hi = (float(v) for v in _signed_range(facing * RANGE_SIGN[part], *ROM[part]).split())
            ctrlrange = f"{np.radians(deg_lo)} {np.radians(deg_hi)}"
            lines.append(
                f'    <position name="{joint}_p" joint="{joint}" ctrlrange="{ctrlrange}" kp="{kp[part]}" />'
            )
    # kp=3000 could brute-force the pelvis through the feet's grip on the ground (confirmed:
    # dropping to 1500 cut the resulting sideways drag on a hip-twist slider by ~60% while still
    # reaching close to the commanded angle) -- friction alone barely helped, since no physically
    # realistic friction coefficient stops an actuator this strong from overpowering it.
    hip_twist_rad = np.radians(HIP_TWIST_MAX)
    lines.append(
        f'    <position name="{name}_hip_twist_p" joint="{name}_hip_twist" '
        f'ctrlrange="{-hip_twist_rad} {hip_twist_rad}" kp="1500" />'
    )
    waist_twist_rad = np.radians(WAIST_TWIST_MAX)
    lines.append(
        f'    <position name="{name}_waist_twist_p" joint="{name}_waist_twist" '
        f'ctrlrange="{-waist_twist_rad} {waist_twist_rad}" kp="1500" />'
    )
    bend_lo, bend_hi = (float(v) for v in _signed_range(-facing, *ROM['waist_bend']).split())
    lines.append(
        f'    <position name="{name}_waist_bend_p" joint="{name}_waist_bend" '
        f'ctrlrange="{np.radians(bend_lo)} {np.radians(bend_hi)}" kp="1800" />'
    )
    return "\n".join(lines)


FIGHTER_STYLE = {
    "red": ("0.85 0.16 0.12 1", "0.98 0.45 0.18 1", "1 0.82 0.2 1"),
    "blue": ("0.10 0.35 0.88 1", "0.25 0.75 1 1", "1 0.82 0.2 1"),
}
# Explore mode only spawns one fighter -- decluttered for testing a single character's range
# of motion rather than watching two of them at once.
EXPLORE_FIGHTERS = ("red",)


def _self_collision_excludes(name: str) -> str:
    """Body pairs on one fighter that should never collide with each other.

    Currently empty, deliberately.

    This previously excluded the forearms and upperarms from colliding with the chest and pelvis.
    That was a mistake: arm-versus-torso collision is real physics, and removing it let the
    forearms pass straight through the body -- measured at 6.7cm of interpenetration with zero
    contacts generated, which is exactly the "elbows going inside the body" that shows up on
    screen. The original justification was avoiding "contact-force jolts", but that was a guess,
    never verified, and it turns out to be wrong: with the exclusions removed the arm generates
    normal contacts (623 over a 600-step arm-across-body drive), penetration drops to a routine
    1.9cm of soft-contact depth, constraint forces stay bounded and the state stays finite.

    Kept as a function rather than deleted so there is an obvious place to add a genuine exclusion
    if one is ever actually needed -- and a record of why blanket-excluding was wrong.
    """
    return ""


def _build_model_xml(
    actuators_fn, fighters: tuple[str, ...] = ("red", "blue"),
    free_torso: bool = False, ankle_mode: str | None = None,
) -> str:
    bodies = "\n".join(
        _humanoid_body(name, *FIGHTER_STYLE[name], free_torso=free_torso, ankle_mode=ankle_mode) for name in fighters
    )
    actuators = "\n".join(actuators_fn(name, ankle_mode=ankle_mode) for name in fighters)
    excludes = "\n".join(_self_collision_excludes(name) for name in fighters)
    return f"""
<mujoco model="two_fighters">
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
    <geom name="floor" type="plane" size="8 8 0.1" material="floor_material" />
    {bodies}
  </worldbody>
  <contact>
{excludes}
  </contact>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


MODEL_XML = _build_model_xml(_actuators)
EXPLORE_MODEL_XML = _build_model_xml(_explore_actuators, fighters=EXPLORE_FIGHTERS)


class FightEnv:
    """Small continuous-control-style environment for two humanoid fighters."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.fighter_red = self.model.body("red").id
        self.fighter_blue = self.model.body("blue").id
        self._red_qpos = int(self.model.jnt_qposadr[self.model.body("red").jntadr[0]])
        self._red_qvel = int(self.model.jnt_dofadr[self.model.body("red").jntadr[0]])
        self._blue_qpos = int(self.model.jnt_qposadr[self.model.body("blue").jntadr[0]])
        self._blue_qvel = int(self.model.jnt_dofadr[self.model.body("blue").jntadr[0]])
        self.reset()

    def reset(self) -> np.ndarray:
        # mj_resetData alone already places each fighter correctly: the root body's authored
        # `pos` in the MJCF IS the standing spot, and a slide joint's qpos is a displacement
        # from that authored pos, so qpos=0 (the reset default) means "no displacement" --
        # exactly where it should stand. Only the limb joints still need their guard-pose qpos
        # applied explicitly, since qpos=0 there means straight-limbed, not standing.
        mujoco.mj_resetData(self.model, self.data)
        _set_rest_pose_qpos(self.model, self.data, "red")
        _set_rest_pose_qpos(self.model, self.data, "blue")
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self) -> np.ndarray:
        red = self.data.xpos[self.fighter_red]
        blue = self.data.xpos[self.fighter_blue]
        red_vel = self.data.qvel[self._red_qvel:self._red_qvel + 2]
        blue_vel = self.data.qvel[self._blue_qvel:self._blue_qvel + 2]
        return np.concatenate((red, blue, red_vel, blue_vel))

    def step(self, red_action: float, blue_action: float) -> np.ndarray:
        """Actions are horizontal intent values in [-1, 1]."""
        red_x = self.data.xpos[self.fighter_red, 0]
        blue_x = self.data.xpos[self.fighter_blue, 0]
        red_direction = np.sign(blue_x - red_x)
        blue_direction = np.sign(red_x - blue_x)

        self.data.xfrc_applied[self.fighter_red, 0] = 865 * red_action * red_direction
        self.data.xfrc_applied[self.fighter_blue, 0] = 865 * blue_action * blue_direction
        mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:, :] = 0
        return self.observation()


def main() -> None:
    env = FightEnv()
    print("MuJoCo fight demo running. Close the viewer window to stop.")

    waist_ids = {name: env.model.actuator(f"{name}_waist_twist_m").id for name in ("red", "blue")}

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 8
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18
        step_count = 0
        while viewer.is_running():
            # A simple scripted policy: approach, then throw punches once in range.
            distance = abs(env.data.xpos[env.fighter_blue, 0] - env.data.xpos[env.fighter_red, 0])
            approaching = distance > 1.3
            action = 1.0 if approaching else 0.2
            env.step(action, action)

            punch = 0.0 if approaching else (np.sin(step_count * 0.3) * 0.5 + 0.5)
            env.data.ctrl[:] = punch

            # Kinetic chain: the hip leads the rotation and the torso follows a beat later,
            # instead of both spinning together as one rigid block -- real hip-shoulder
            # separation, not just a slower version of the same motion.
            if not approaching:
                torso_follow = np.sin((step_count - 6) * 0.3) * 0.5 + 0.5
                for waist_id in waist_ids.values():
                    env.data.ctrl[waist_id] = torso_follow

            viewer.sync()
            step_count += 1
            time.sleep(env.model.opt.timestep)


WALK_FREQ = 0.06  # radians per sim step (dt=0.01) -- roughly a 1s stride cycle
HIP_SWING_DEG = 20.0
KNEE_LIFT_DEG = 30.0
ARM_SWING_DEG = 18.0
WALK_FORCE = 320.0  # forward newtons applied while walking -- without this it's just swaying
                     # limbs in place (foot-ground friction alone only crept forward ~0.26m over
                     # 3 seconds of cycling), which reads as dancing, not going anywhere.


def _walk_targets(rest_deg: dict[str, float], facing: float, phase: float) -> dict[str, float]:
    """One instant of a stylized walk cycle, as target angles (degrees) for the leg/arm joints.

    Legs swing 180 degrees out of phase (right forward while left is back, then swap). Real gait
    bends the knee in the *middle* of its own leg's swing (foot clearing the ground) and
    straightens it again by the time that leg reaches full forward extension for heel strike --
    not bent hardest exactly when most forward, which looks like a prance/skip instead of a walk.
    `cos(phase)` peaks a quarter-cycle before `sin(phase)` (hip forward) does, giving that lead.
    Arms counter-swing opposite their same-side leg, like natural walking.
    """
    swing_r = np.sin(phase)
    swing_l = -swing_r
    lift_r = np.cos(phase)
    lift_l = -lift_r
    return {
        "hip_r": rest_deg["hip_r"] - HIP_SWING_DEG * facing * swing_r,
        "hip_l": rest_deg["hip_l"] - HIP_SWING_DEG * facing * swing_l,
        "knee_r": rest_deg["knee_r"] + KNEE_LIFT_DEG * facing * max(0.0, lift_r),
        "knee_l": rest_deg["knee_l"] + KNEE_LIFT_DEG * facing * max(0.0, lift_l),
        "shoulder_r": rest_deg["shoulder_r"] - ARM_SWING_DEG * facing * swing_l,
        "shoulder_l": rest_deg["shoulder_l"] - ARM_SWING_DEG * facing * swing_r,
    }


def explore() -> None:
    """Interactive viewer for testing each joint's range of motion by hand.

    No scripted policy runs here by default, so the built-in Control panel sliders (one per
    limb joint) and Ctrl+right-click-drag on any body part are free to use. Press W to toggle a
    walking-cycle animation on the legs/arms; pressing it again snaps every joint straight back
    to the exact standing pose this session started from.
    """
    model = mujoco.MjModel.from_xml_string(EXPLORE_MODEL_XML)
    data = mujoco.MjData(model)

    # The root body's authored `pos` in the MJCF is already the standing spot (see
    # _humanoid_body), so only the limb joints need their guard-pose qpos applied here.
    rest_deg = {name: _rest_pose_deg(FACING[name]) for name in EXPLORE_FIGHTERS}
    rest_ctrl: dict[int, float] = {}
    for name in EXPLORE_FIGHTERS:
        _set_rest_pose_qpos(model, data, name)
        for joint_name, deg in rest_deg[name].items():
            actuator_id = model.actuator(f"{name}_{joint_name}_p").id
            data.ctrl[actuator_id] = np.radians(deg)
            rest_ctrl[actuator_id] = np.radians(deg)
    mujoco.mj_forward(model, data)
    # Full-state snapshot to restore on reset. Walking actually shuffles the character forward a
    # little (an emergent effect of the legs pushing against foot-ground friction, not something
    # scripted) so "reset to where he was" needs to restore position too, not just joint angles.
    initial_qpos = data.qpos.copy()

    state = {"walking": False, "phase": 0.0}

    def key_callback(keycode: int) -> None:
        if keycode == ord("W"):
            state["walking"] = not state["walking"]
            if not state["walking"]:
                data.qpos[:] = initial_qpos
                data.qvel[:] = 0.0
                for actuator_id, target in rest_ctrl.items():
                    data.ctrl[actuator_id] = target
                mujoco.mj_forward(model, data)
                state["phase"] = 0.0

    print("Explore mode: drag the sliders in the Control panel (right side) to move a joint.")
    print("Hold Ctrl + right-click-drag on any body part to push or pull it directly.")
    print("Press W to toggle a walking animation, Space to pause/resume. Close window to exit.")

    body_ids = {name: model.body(name).id for name in EXPLORE_FIGHTERS}

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if state["walking"]:
                for name in EXPLORE_FIGHTERS:
                    targets = _walk_targets(rest_deg[name], FACING[name], state["phase"])
                    for joint_name, deg in targets.items():
                        actuator_id = model.actuator(f"{name}_{joint_name}_p").id
                        data.ctrl[actuator_id] = np.radians(deg)
                    # Actually propel him forward -- the leg-cycling animation alone barely
                    # translates him (foot-ground friction only crept him along), which reads
                    # as swaying in place rather than walking somewhere.
                    data.xfrc_applied[body_ids[name], 0] = WALK_FORCE * FACING[name]
                state["phase"] += WALK_FREQ
            else:
                for body_id in body_ids.values():
                    data.xfrc_applied[body_id, 0] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo two-fighter demo.")
    parser.add_argument(
        "--explore", action="store_true",
        help="Open an interactive viewer with per-joint sliders instead of the scripted fight demo.",
    )
    args = parser.parse_args()
    explore() if args.explore else main()
