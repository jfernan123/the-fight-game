"""Hold the humanoid in a fixed stance and shove it, to find out which postures actually resist.

This is a physics sandbox, not a policy viewer -- no network is involved. Each stance is a set of
joint angles held by position servos, so what you are testing is the POSTURE and the body's
geometry (how wide the feet are, how low the mass sits, where the support polygon is), separated
from whatever a learned policy happens to do.

Worth knowing before reading the numbers: a stance by itself does NOT balance a biped. Measured,
this body with a free torso and no controller falls in ~200 steps (2s) whatever pose it is put in.
A horse stance is not stable because of its shape -- a person actively regulates their centre of
mass, and the shape only widens the region where that regulation can succeed. What the servos here
provide is a fixed, identical-for-every-stance amount of that regulation, so the comparison is
about the posture rather than about who got the better controller.

Two ways to run it:

    python stance_lab.py --benchmark
        Headless. For every stance, ramps the push up until it falls, and reports the threshold.
        This is the one that answers "is a horse stance really harder to push over".

    python stance_lab.py [--stance horse]
        Interactive viewer:
            TAB     cycle to the next stance
            X       random push        SHIFT+X  double strength
            G / H   push left / right
            V / B   push from behind / in front
            - / =   change push strength by 0.5 m/s
            R       reset to the stance
            P       print stance metrics (CoM height, support polygon, contacts)
"""

import argparse
import re
import time

import mujoco
import mujoco.viewer
import numpy as np

from fight_env import _actuators, _build_model_xml, _set_rest_pose_qpos
from reward_terms import PUSH_MAX_VEL
from walk_env import AGENT

# Joint angles in DEGREES; anything not named is left at its rest-pose angle. Signs follow the
# model's own convention, mirrored per side for the abduction joints. Measured, not assumed:
# hip_abduct_r=+40 / _l=-40 gives 0.66m of foot separation, -40/+40 gives 0.12m, 0/0 gives
# 0.22m. Getting this backwards silently swaps the horse and narrow stances, which is
# exactly what happened on the first pass here.
STANCES: dict[str, dict[str, float]] = {
    "neutral": {
        "hip_r": -10, "hip_l": -10, "knee_r": 15, "knee_l": 15,
    },
    "horse": {
        # Mabu. Feet well outside the hips, knees driven out over the feet, torso vertical.
        "hip_abduct_r": 40, "hip_abduct_l": -40,
        "hip_r": -45, "hip_l": -45, "knee_r": 70, "knee_l": 70,
        "ankle_pitch_r": -8, "ankle_pitch_l": -8,
    },
    "half_squat": {
        # Horse-stance knee bend at normal foot width -- isolates "low" from "wide".
        "hip_r": -45, "hip_l": -45, "knee_r": 70, "knee_l": 70,
        "ankle_pitch_r": -8, "ankle_pitch_l": -8,
    },
    "wide_tall": {
        # Wide feet, straight legs -- isolates "wide" from "low".
        "hip_abduct_r": 40, "hip_abduct_l": -40,
    },
    "staggered": {
        # Boxing stance: one foot forward, knees soft, hips turned.
        "hip_r": -35, "hip_l": 10, "knee_r": 35, "knee_l": 20,
        "hip_abduct_r": 12, "hip_abduct_l": -12, "hip_twist": 25,
    },
    "narrow": {
        # Feet together. Expected to be worst laterally; the control case.
        "hip_abduct_r": -18, "hip_abduct_l": 18,
    },
}

# Position servos, not the torque motors the policy uses. A torque-PD over gear-1500 motors
# saturates instantly and behaves bang-bang -- measured, it pinned |ctrl| at 1.00 every step and
# launched the body clear off the floor (ground contacts hit zero) before any push was applied.
# A position actuator commands an ANGLE, which is what "hold this stance" actually means. The kp
# values follow fight_env._explore_actuators, where they were tuned against the passive springs;
# kv damps the approach so the pose settles instead of snapping and bouncing.
STANCE_KP = {"hip": 1800, "knee": 1800, "hip_abduct": 1200, "ankle": 600, "ankle_pitch": 600,
             "ankle_roll": 400, "shoulder": 1200, "shoulder_abduct": 800, "elbow": 1200,
             "hip_twist": 1500, "waist_twist": 1500, "waist_bend": 1500}
STANCE_KV = 0.08     # velocity damping, as a fraction of kp

# The lab's own fall test. A deep stance is legitimately LOW, so walk_env's torso-height criterion
# would score a horse stance as already fallen.
FALL_TILT_DEG = 60.0


def _stance_actuators(name: str, ankle_mode: str | None = None) -> str:
    """A position servo on every limited hinge, so any stance can actually be held."""
    probe = mujoco.MjModel.from_xml_string(
        _build_model_xml(_actuators, fighters=(name,), free_torso=True, ankle_mode=ankle_mode))
    lines = []
    for j in range(probe.njnt):
        jname = probe.joint(j).name
        if (not jname or not jname.startswith(f"{name}_")
                or probe.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE
                or not probe.jnt_limited[j]):
            continue
        key = re.sub(r"_(r|l)$", "", jname[len(name) + 1:])
        kp = STANCE_KP.get(key, 1000)
        lo, hi = probe.jnt_range[j]
        lines.append(f'    <position name="{jname}_p" joint="{jname}" '
                     f'ctrlrange="{lo} {hi}" kp="{kp}" kv="{kp * STANCE_KV:.0f}" />')
    return "\n".join(lines)


class StanceRig:
    """Holds one stance with position servos and lets you shove it."""

    def __init__(self, stance: str = "horse", ankle_mode: str = "detailed") -> None:
        xml = _build_model_xml(_stance_actuators, fighters=(AGENT,),
                               free_torso=True, ankle_mode=ankle_mode)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jname = [m.actuator(i).name[:-2] for i in range(m.nu)]      # strip the trailing "_p"
        self.qadr = [int(m.joint(j).qposadr[0]) for j in self.jname]
        self.root = m.body(AGENT).id
        self.chest = m.body(f"{AGENT}_chest").id
        self.x_dof = int(m.jnt_dofadr[m.joint(f"{AGENT}_x").id])
        self.y_dof = int(m.jnt_dofadr[m.joint(f"{AGENT}_y").id])
        self.z_adr = int(m.joint(f"{AGENT}_z").qposadr[0])
        self.floor = m.geom("floor").id
        self.foot_geoms = {g for g in range(m.ngeom)
                           if any(k in (m.geom(g).name or "") for k in ("foot", "heel", "toe"))}
        self.names = list(STANCES)
        self.stance = stance
        self.push_vel = PUSH_MAX_VEL
        self.apply_stance(stance)

    # --- stance ---------------------------------------------------------------------------
    def apply_stance(self, name: str) -> None:
        self.stance = name
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        _set_rest_pose_qpos(m, d, AGENT)
        spec = STANCES[name]
        for i, jn in enumerate(self.jname):
            base = jn[len(AGENT) + 1:]
            if base in spec:
                d.qpos[self.qadr[i]] = np.radians(spec[base])
            d.ctrl[i] = float(np.clip(d.qpos[self.qadr[i]], *m.actuator_ctrlrange[i]))
        mujoco.mj_forward(m, d)
        # Posing the joints does not move the root, so a bent-knee stance leaves the feet hanging
        # in the air (measured 13-17cm for the horse stance); the body then drops and destabilises
        # before it can settle. Lower the root until the lowest foot geom rests on the floor.
        d.qpos[self.z_adr] -= self._lowest_foot_z()
        mujoco.mj_forward(m, d)
        self.settle()

    def _lowest_foot_z(self) -> float:
        m, d = self.model, self.data
        lows = [float(d.geom_xpos[g][2]
                      - np.abs(d.geom_xmat[g].reshape(3, 3)[2, :3]) @ m.geom_size[g][:3])
                for g in self.foot_geoms]
        return min(lows) if lows else 0.0

    # --- simulation -----------------------------------------------------------------------
    def step(self) -> bool:
        mujoco.mj_step(self.model, self.data)
        return self.fallen()

    def settle(self, steps: int = 400) -> bool:
        for _ in range(steps):
            if self.step():
                return True
        return False

    def fallen(self) -> bool:
        """Tilted past FALL_TILT_DEG, or something that is not a foot is touching the floor."""
        if np.degrees(np.arccos(np.clip(self.data.xmat[self.chest].reshape(3, 3)[2, 2],
                                        -1.0, 1.0))) > FALL_TILT_DEG:
            return True
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.floor in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.floor else c.geom1
                if other not in self.foot_geoms:
                    return True
        return False

    # --- pushing --------------------------------------------------------------------------
    def push(self, direction=None, scale: float = 1.0) -> tuple[float, float]:
        vx, vy = np.random.uniform(-1, 1, 2) if direction is None else direction
        vx, vy = vx * self.push_vel * scale, vy * self.push_vel * scale
        self.data.qvel[self.x_dof] += vx
        self.data.qvel[self.y_dof] += vy
        mujoco.mj_forward(self.model, self.data)
        return vx, vy

    # --- measurement ----------------------------------------------------------------------
    def metrics(self) -> dict:
        d = self.data
        pts = np.array([d.contact[i].pos for i in range(d.ncon)
                        if self.floor in (d.contact[i].geom1, d.contact[i].geom2)])
        return {
            "CoM height (m)": round(float(d.subtree_com[self.root][2]), 3),
            "support fore-aft (m)": round(float(np.ptp(pts[:, 0])), 3) if len(pts) else 0.0,
            "support lateral (m)": round(float(np.ptp(pts[:, 1])), 3) if len(pts) else 0.0,
            "ground contacts": int(len(pts)),
        }


def benchmark(ankle_mode: str) -> None:
    """For each stance, ramp the push until it collapses. The threshold is the comparable number."""
    directions = (("sideways", (0.0, 1.0)), ("from behind", (1.0, 0.0)))
    print("Ramping each push until the stance collapses. Higher = harder to push over.\n")
    header = (f"{'stance':12s} | {'CoM z':>6} {'supp FB':>8} {'supp LR':>8} |"
              + "".join(f" {lbl + ' (m/s)':>17}" for lbl, _ in directions))
    print(header)
    print("-" * len(header))
    for name in STANCES:
        rig = StanceRig(name, ankle_mode)
        if rig.fallen():
            print(f"{name:12s} | collapsed on its own before any push")
            continue
        mt = rig.metrics()
        row = (f"{name:12s} | {mt['CoM height (m)']:>6.2f} {mt['support fore-aft (m)']:>8.3f} "
               f"{mt['support lateral (m)']:>8.3f} |")
        for _, vec in directions:
            threshold = None
            for dv in np.arange(0.5, 12.1, 0.5):
                rig.apply_stance(name)
                rig.push_vel = float(dv)
                rig.push(direction=vec)
                if any(rig.step() for _ in range(300)):
                    threshold = float(dv)
                    break
            row += f" {(f'{threshold:.1f}' if threshold else '>12.0'):>17}"
        print(row, flush=True)
    print("\n(threshold = smallest impulse that knocks it down; '>12.0' means it never fell)")


def watch(stance: str, ankle_mode: str) -> None:
    rig = StanceRig(stance, ankle_mode)
    idx = {"i": rig.names.index(stance)}
    down = {"fallen": False}     # hold the fall instead of resetting every frame
    print(__doc__[__doc__.index("            TAB"):])
    print(f"\n[STANCE] {stance}   (push strength {rig.push_vel:.1f} m/s)")

    def reset_to(name: str) -> None:
        rig.apply_stance(name)
        down["fallen"] = False

    def on_key(keycode: int) -> None:
        key = chr(keycode) if 32 <= keycode < 127 else None
        if keycode in (258, 9):                             # TAB
            idx["i"] = (idx["i"] + 1) % len(rig.names)
            reset_to(rig.names[idx["i"]])
            print(f"\n[STANCE] {rig.stance}")
        elif key in ("r", "R"):
            reset_to(rig.stance)
            print(f"[RESET] {rig.stance}")
        elif key == "-":
            rig.push_vel = max(0.5, rig.push_vel - 0.5)
            print(f"[PUSH STRENGTH] {rig.push_vel:.1f} m/s")
        elif key == "=":
            rig.push_vel += 0.5
            print(f"[PUSH STRENGTH] {rig.push_vel:.1f} m/s")
        elif key in ("x", "X", "g", "G", "h", "H", "v", "V", "b", "B"):
            vec = {"g": (0.0, 1.0), "h": (0.0, -1.0),
                   "v": (1.0, 0.0), "b": (-1.0, 0.0)}.get(key.lower())
            if down["fallen"]:                              # a push also un-pauses a fall
                reset_to(rig.stance)
            vx, vy = rig.push(direction=vec, scale=2.0 if key == "X" else 1.0)
            print(f"[PUSH] dv = ({vx:+.2f}, {vy:+.2f}) m/s   strength {rig.push_vel:.1f}")
        elif key in ("p", "P"):
            print(f"[{rig.stance}]")
            for k, v in rig.metrics().items():
                print(f"    {k:22s} {v}")

    dt = float(rig.model.opt.timestep)
    with mujoco.viewer.launch_passive(rig.model, rig.data, key_callback=on_key) as viewer:
        # Frame the humanoid. The default camera looks at the world origin from close range, which
        # is most of why this window came up apparently empty.
        viewer.cam.lookat[:] = rig.data.xpos[rig.root]
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -15.0
        while viewer.is_running():
            frame_start = time.time()
            if not down["fallen"] and rig.step():
                down["fallen"] = True
                print(f"[FELL] {rig.stance} -- R to reset, TAB for the next stance")
            viewer.sync()
            # Pace to real time. Without this the loop steps physics as fast as the CPU allows,
            # so the whole episode is over before a single readable frame reaches the screen.
            time.sleep(max(0.0, dt - (time.time() - frame_start)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", action="store_true", help="headless push-resistance ranking")
    ap.add_argument("--stance", default="horse", choices=list(STANCES))
    ap.add_argument("--ankle", default="detailed", choices=["single", "dual", "detailed"])
    args = ap.parse_args()
    benchmark(args.ankle) if args.benchmark else watch(args.stance, args.ankle)


if __name__ == "__main__":
    main()
