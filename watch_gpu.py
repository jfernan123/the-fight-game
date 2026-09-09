"""Watch a GPU-trained (train_gpu.py) checkpoint run, with an interactive debugger.

Rendering uses the plain CPU WalkEnv (walk_env.py) -- a single visible episode doesn't need GPU
batching, only the policy network's forward pass comes from the GPU-trained checkpoint, and that's
cheap enough on CPU for one env at a time.

    python watch_gpu.py --ankle single                        # newest checkpoint in runs_gpu/
    python watch_gpu.py runs_gpu/single/<run>/walker_final.pt --ankle single

--ankle MUST match whatever the checkpoint was trained with -- the network's input/output sizes are
baked into its saved weights, so a mismatch fails to load rather than silently misbehaving.

INTERACTIVE CONTROLS (click the viewer window first so it has keyboard focus)

    SPACE   pause / resume physics
    N       while paused: advance exactly one step
    R       reset the episode
    S / L   save / restore a full state snapshot (qpos+qvel), so you can try something,
            see it fail, and jump straight back to the moment before it
    [ / ]   decrease / increase the commanded speed by 0.1 m/s
    P       print a full state dump: joint angles, velocities, contacts, and the per-term
            reward breakdown for the current step

  Shove it and see if it catches itself. The environment already pushes on its own every
  PUSH_INTERVAL seconds during training; these are the same impulse, on demand:

    X       random push (the training-strength impulse, random direction)
    SHIFT+X hard push -- double strength, for finding where recovery breaks down
    G / H   push from the left / right (the direction it has the least defence against)
    V / B   push from behind / in front
    - / =   scale every push down / up by 0.5 m/s, so you can walk the strength up until
            it actually falls instead of guessing

  Per-joint override -- hold one or more joints at a value you choose while the policy keeps
  driving everything else. This is the one for "how does forcing this limb change the gait":

    T       cycle which joint is selected
    O       override the selected joint on/off (starts at whatever the policy is commanding)
    , / .   nudge the selected joint's override down / up by 0.1 (clamped to [-1, 1])
    0       set the selected joint's override to 0
    C       clear every override, hand everything back to the policy

  Whole-body manual:

    M       manual control on/off -- the policy stops writing actuator commands entirely, so
            the viewer's own Control panel sliders take over every joint at once.

To move the character directly, use MuJoCo's own perturbation controls: double-click a body to
select it, then Ctrl+left-drag to rotate it or Ctrl+right-drag to translate it. Pause first
(SPACE) if you want to place it precisely before letting physics run again.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from reward_terms import PUSH_MAX_VEL
from train_gpu import ActorCritic
from walk_env import AGENT, MAX_EPISODE_STEPS, WalkEnv


def _latest_checkpoint(checkpoint_dir: str) -> str:
    # Recursive since train_gpu.py nests checkpoints under a per-run timestamp folder
    # (runs_gpu/<mode>/<run-id>/) -- "**" also matches zero intermediate dirs, so this still finds
    # checkpoints saved directly in checkpoint_dir by an older, un-timestamped run.
    candidates = glob.glob(os.path.join(checkpoint_dir, "**", "*.pt"), recursive=True)
    if not candidates:
        raise SystemExit(f"No checkpoint .pt files found under '{checkpoint_dir}'.")
    return max(candidates, key=os.path.getmtime)


class _Debugger:
    """Keyboard state for the interactive viewer.

    Kept as an object rather than globals because mujoco's key_callback is a plain function
    reference -- it needs somewhere to put the flags it toggles.
    """

    def __init__(self, env: WalkEnv) -> None:
        self.env = env
        self.paused = False
        self.manual = False
        self.step_once = False
        self.reset_requested = False
        self.saved_state: tuple[np.ndarray, np.ndarray, float, float] | None = None
        # Per-joint overrides: index into env.walk_joints -> held action value in [-1, 1].
        # Applied on top of the policy's action each step, so everything NOT listed here keeps
        # being driven normally. Kept as our own dict rather than reading the viewer's Control
        # panel, because the policy rewrites data.ctrl every step and would clobber slider edits.
        self.overrides: dict[int, float] = {}
        # Starts at the training strength so what you feel matches what it trains against;
        # adjustable live with - and = to find where recovery actually breaks down.
        self.push_vel = PUSH_MAX_VEL
        self.selected = 0
        self.last_action: np.ndarray | None = None

    # ---- per-joint override helpers ----

    def _joint_label(self, idx: int) -> str:
        name = self.env.walk_joints[idx]
        if idx in self.overrides:
            return f"{name} = {self.overrides[idx]:+.2f} (OVERRIDDEN)"
        return f"{name} (policy)"

    def _print_selection(self) -> None:
        held = ", ".join(f"{self.env.walk_joints[i]}={v:+.2f}" for i, v in sorted(self.overrides.items()))
        print(f"[SELECT {self.selected + 1}/{len(self.env.walk_joints)}] {self._joint_label(self.selected)}"
              + (f"   |  held: {held}" if held else ""))

    def apply_overrides(self, action: np.ndarray) -> np.ndarray:
        """Substitute the held values into the policy's action, leaving the rest untouched."""
        if not self.overrides:
            return action
        action = action.copy()
        for idx, value in self.overrides.items():
            action[idx] = value
        return action

    def key_callback(self, keycode: int) -> None:
        key = chr(keycode) if 32 <= keycode < 127 else None
        if keycode == 32:  # SPACE
            self.paused = not self.paused
            print(f"[{'PAUSED' if self.paused else 'RUNNING'}]")
        elif key in ("n", "N"):
            self.step_once = True
        elif key in ("m", "M"):
            self.manual = not self.manual
            if self.manual:
                print("[MANUAL] policy released the actuators -- drag the Control panel sliders")
            else:
                print("[AUTO] policy driving again")
        elif key in ("r", "R"):
            self.reset_requested = True
        elif key in ("s", "S"):
            self.saved_state = (
                self.env.data.qpos.copy(), self.env.data.qvel.copy(),
                self.env._phase, self.env._target_speed,
            )
            print("[SAVED] state snapshot stored (press L to restore)")
        elif key in ("l", "L"):
            if self.saved_state is None:
                print("[RESTORE] nothing saved yet -- press S first")
            else:
                qpos, qvel, phase, speed = self.saved_state
                self.env.data.qpos[:] = qpos
                self.env.data.qvel[:] = qvel
                self.env._phase, self.env._target_speed = phase, speed
                mujoco.mj_forward(self.env.model, self.env.data)
                print("[RESTORED] back to the saved snapshot")
        elif key == "[":
            self.env._target_speed = max(0.0, self.env._target_speed - 0.1)
            print(f"[COMMAND] target speed = {self.env._target_speed:.2f} m/s")
        elif key == "]":
            self.env._target_speed += 0.1
            print(f"[COMMAND] target speed = {self.env._target_speed:.2f} m/s")
        elif key in ("t", "T"):
            self.selected = (self.selected + 1) % len(self.env.walk_joints)
            self._print_selection()
        elif key in ("o", "O"):
            if self.selected in self.overrides:
                del self.overrides[self.selected]
                print(f"[RELEASED] {self.env.walk_joints[self.selected]} back to the policy")
            else:
                # Start from whatever the policy is currently commanding, so switching to
                # override doesn't jolt the joint -- you take the wheel at the current value.
                start = float(self.last_action[self.selected]) if self.last_action is not None else 0.0
                self.overrides[self.selected] = start
                print(f"[OVERRIDE] {self.env.walk_joints[self.selected]} held at {start:+.2f} "
                      f"(, and . to adjust)")
        elif key in (",", "."):
            if self.selected not in self.overrides:
                print(f"[hint] press O first to take over {self.env.walk_joints[self.selected]}")
            else:
                delta = -0.1 if key == "," else 0.1
                self.overrides[self.selected] = float(
                    np.clip(self.overrides[self.selected] + delta, -1.0, 1.0))
                print(f"[OVERRIDE] {self.env.walk_joints[self.selected]} = "
                      f"{self.overrides[self.selected]:+.2f}")
        elif key == "0":
            if self.selected in self.overrides:
                self.overrides[self.selected] = 0.0
                print(f"[OVERRIDE] {self.env.walk_joints[self.selected]} = +0.00")
        elif key in ("c", "C"):
            n = len(self.overrides)
            self.overrides.clear()
            print(f"[CLEARED] {n} override(s) released -- policy has full control again")
        elif key == "-":
            self.push_vel = max(0.5, self.push_vel - 0.5)
            print(f"[PUSH STRENGTH] {self.push_vel:.1f} m/s")
        elif key == "=":
            self.push_vel += 0.5
            print(f"[PUSH STRENGTH] {self.push_vel:.1f} m/s")
        elif key == "X":
            self._push(scale=2.0, label="HARD")
        elif key == "x":
            self._push()
        elif key in ("g", "G"):
            self._push(direction=(0.0, 1.0), label="from the left")
        elif key in ("h", "H"):
            self._push(direction=(0.0, -1.0), label="from the right")
        elif key in ("v", "V"):
            self._push(direction=(1.0, 0.0), label="from behind")
        elif key in ("b", "B"):
            self._push(direction=(-1.0, 0.0), label="from the front")
        elif key in ("p", "P"):
            self.dump()

    def _push(self, direction=None, scale=1.0, label="random"):
        """Apply the same velocity impulse the environment uses for its own random pushes.

        Written as a velocity change rather than a force for the same reason the env does it:
        the magnitude is directly interpretable in m/s and does not depend on how mass happens
        to be distributed through the body.
        """
        env = self.env
        if direction is None:
            vx, vy = np.random.uniform(-1.0, 1.0, size=2)
        else:
            vx, vy = direction
        vx *= self.push_vel * scale
        vy *= self.push_vel * scale
        env.data.qvel[env._x_dof] += vx
        env.data.qvel[env._y_dof] += vy
        mujoco.mj_forward(env.model, env.data)
        print(f"[PUSH {label}] dv = ({vx:+.2f}, {vy:+.2f}) m/s  (strength {self.push_vel:.1f}, - and = to change)")

    def dump(self) -> None:
        env = self.env
        d, m = env.data, env.model
        print("\n" + "=" * 78)
        print(f"step {env._step_count}   target_speed={env._target_speed:.2f}   phase={env._phase:.3f}")
        chest_pitch, chest_roll = env._chest_world_pitch_roll()
        print(f"forward_vel={d.qvel[env._x_dof]:+.3f}  lateral={d.qvel[env._y_dof]:+.3f}  "
              f"vertical={d.qvel[env._z_dof]:+.3f}")
        print(f"chest pitch={np.degrees(chest_pitch):+.1f} deg  roll={np.degrees(chest_roll):+.1f} deg  "
              f"torso_z={d.xpos[env.agent_id, 2]:.3f}")
        print(f"down_ema={env._down_ema:.3f} (fall at {env._down_ema >= 0.5})   "
              f"air_time r={env._air_time['r']:.2f}s l={env._air_time['l']:.2f}s")

        print("\n     joint                 angle(deg)    vel      ctrl   source")
        for i, (j, qadr, dadr, aid) in enumerate(zip(env.walk_joints, env._joint_qpos_adr,
                                                     env._joint_dof_adr, env._actuator_ids)):
            marker = ">" if i == self.selected else " "
            source = f"HELD {self.overrides[i]:+.2f}" if i in self.overrides else "policy"
            print(f"  {marker}  {j:20s} {np.degrees(d.qpos[qadr]):+8.1f}  {d.qvel[dadr]:+7.2f}  "
                  f"{d.ctrl[aid]:+6.2f}   {source}")

        contacts = []
        for i in range(d.ncon):
            c = d.contact[i]
            if env._floor_geom_id in (c.geom1, c.geom2):
                g = c.geom2 if c.geom1 == env._floor_geom_id else c.geom1
                nm = m.geom(g).name or m.body(m.geom_bodyid[g]).name
                contacts.append(nm)
        print(f"\n  floor contacts: {contacts if contacts else '(airborne)'}")

        if hasattr(env, "reward_breakdown"):
            print("\n  reward term            value")
            for k, v in sorted(env.reward_breakdown.items(), key=lambda kv: -abs(kv[1])):
                print(f"  {k:22s} {v:+.4f}")
            print(f"  {'TOTAL':22s} {sum(env.reward_breakdown.values()):+.4f}")
        print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a GPU-trained checkpoint, with an interactive debugger.")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to a saved .pt checkpoint. Omit to auto-pick the most recently modified one "
             "in --checkpoint-dir -- safe to run mid-training to see progress.",
    )
    parser.add_argument("--checkpoint-dir", default="runs_gpu")
    parser.add_argument(
        "--ankle", choices=["single", "dual", "detailed"], default=None,
        help="must match what this checkpoint was trained with (omit for no-ankle).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--speed", type=float, default=None,
        help="pin the commanded speed instead of letting it resample (adjustable live with [ and ])",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or _latest_checkpoint(args.checkpoint_dir)
    print(f"Loading {checkpoint}")

    env = WalkEnv(ankle_mode=args.ankle)   # no render_mode: this script drives the viewer itself
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    agent.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent.eval()

    dbg = _Debugger(env)
    print(__doc__[__doc__.index("INTERACTIVE CONTROLS"):])

    obs, _ = env.reset()
    if args.speed is not None:
        env._target_speed = args.speed
        env._resample_countdown = 10 ** 9   # effectively never resample

    episode, total_reward = 0, 0.0
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=dbg.key_callback) as viewer:
        viewer.cam.distance = 6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -18

        while viewer.is_running() and episode < args.episodes:
            frame_start = time.time()

            if dbg.reset_requested:
                obs, _ = env.reset()
                if args.speed is not None:
                    env._target_speed = args.speed
                    env._resample_countdown = 10 ** 9
                total_reward = 0.0
                dbg.reset_requested = False
                print("[RESET]")

            advance = (not dbg.paused) or dbg.step_once
            if advance:
                dbg.step_once = False
                if dbg.manual:
                    # Policy hands over: don't touch data.ctrl, so whatever the viewer's Control
                    # panel sliders are set to is what actually drives the joints. Physics still
                    # runs, so you see the consequences of the values you dial in.
                    mujoco.mj_step(env.model, env.data)
                    env._step_count += 1
                else:
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                        action = agent.actor_mean(obs_t).squeeze(0).numpy()  # deterministic
                    dbg.last_action = action
                    # Held joints get their value substituted in; everything else stays exactly
                    # what the policy asked for, so the gait keeps running around the override.
                    action = dbg.apply_overrides(action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    if args.speed is not None:
                        env._target_speed = args.speed
                    if terminated or truncated:
                        episode += 1
                        why = "fell" if terminated else f"reached {MAX_EPISODE_STEPS} steps"
                        print(f"Episode {episode}: total_reward={total_reward:.1f}  "
                              f"steps={env._step_count}  ({why})")
                        obs, _ = env.reset()
                        if args.speed is not None:
                            env._target_speed = args.speed
                            env._resample_countdown = 10 ** 9
                        total_reward = 0.0

            viewer.sync()
            # Real-time pacing. While paused we still sync so the window stays responsive to
            # camera drags, perturbations and key presses.
            elapsed = time.time() - frame_start
            time.sleep(max(0.0, env.dt - elapsed))


if __name__ == "__main__":
    main()
