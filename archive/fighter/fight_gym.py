"""Gymnasium environment: train a fighter (red) against the scripted opponent (blue).

Balance and locomotion aren't learning problems here -- the torso only has translation-only
slide joints (can't tip over) and moves via a direct applied force (not leg-driven propulsion),
both already true of the base model in fight_env.py. What's actually learned is combat: when to
advance/retreat and when/how to throw punches to win.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for fight_env

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from fight_env import FACING, MODEL_XML, START_X, STAND_HEIGHT, _rest_pose_deg, _set_rest_pose_qpos

AGENT = "red"
OPPONENT = "blue"

MOVE_FORCE = 865.0  # matches FightEnv.step's tuned locomotion force for this body weight

MAX_HP = 100.0
HEAD_DAMAGE = 12.0
BODY_DAMAGE = 6.0
HIT_COOLDOWN_STEPS = 15  # ~0.15s at dt=0.01 -- stops one continuous glove-on-body contact from
                          # scoring every single physics step, which would make "lean a glove on
                          # him" a free win condition instead of "land a punch"

# Opponent geoms that count as a landed hit if a glove touches them; anything else (an arm/leg
# in the way, i.e. blocked) doesn't score.
HITTABLE_GEOMS = {
    "pelvis": BODY_DAMAGE, "spine_low": BODY_DAMAGE, "waist": BODY_DAMAGE,
    "spine_up": BODY_DAMAGE, "chest_geom": BODY_DAMAGE, "head": HEAD_DAMAGE,
}

# All 7 torque actuators the fight demo already defines per fighter (see fight_env._actuators),
# exposed directly as the agent's punch/guard action space instead of the hardcoded sine pulse.
AGENT_ACTUATORS = [
    "shoulder_r", "elbow_r", "shoulder_l", "elbow_l", "hip_twist", "waist_twist", "waist_bend",
]

MAX_EPISODE_STEPS = 2000  # 20s at dt=0.01


class FightGymEnv(gym.Env):
    """Single-agent Gymnasium env: red is the learning agent, blue runs the same scripted
    approach-and-punch policy as fight_env.main(), reused verbatim."""

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._viewer = None

        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.agent_id = self.model.body(AGENT).id
        self.opp_id = self.model.body(OPPONENT).id
        self._agent_qvel = int(self.model.jnt_dofadr[self.model.body(AGENT).jntadr[0]])
        self._opp_qvel = int(self.model.jnt_dofadr[self.model.body(OPPONENT).jntadr[0]])

        self._agent_actuator_ids = [self.model.actuator(f"{AGENT}_{j}_m").id for j in AGENT_ACTUATORS]
        self._opp_all_actuator_ids = [
            self.model.actuator(f"{OPPONENT}_{j}_m").id
            for j in AGENT_ACTUATORS  # same 7 joints exist on both fighters
        ]
        self._opp_waist_twist_id = self.model.actuator(f"{OPPONENT}_waist_twist_m").id

        self._agent_glove_ids = [self.model.geom(f"{AGENT}_glove_{s}").id for s in ("r", "l")]
        self._opp_glove_ids = [self.model.geom(f"{OPPONENT}_glove_{s}").id for s in ("r", "l")]
        self._agent_hittable = {self.model.geom(f"{AGENT}_{g}").id: dmg for g, dmg in HITTABLE_GEOMS.items()}
        self._opp_hittable = {self.model.geom(f"{OPPONENT}_{g}").id: dmg for g, dmg in HITTABLE_GEOMS.items()}

        self._agent_joint_ids = self._body_joint_ids(AGENT)
        self._opp_key_geoms = [
            self.model.geom(f"{OPPONENT}_glove_r").id,
            self.model.geom(f"{OPPONENT}_glove_l").id,
            self.model.geom(f"{OPPONENT}_head").id,
        ]

        self._agent_hp = MAX_HP
        self._opp_hp = MAX_HP
        self._step_count = 0
        self._glove_cooldown: dict[int, int] = {}

        n_obs = len(self._build_observation())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1 + len(AGENT_ACTUATORS),), dtype=np.float32)

    def _body_joint_ids(self, name: str) -> list[int]:
        """All hinge joints belonging to this fighter (limbs + torso twist/bend), by name --
        every joint fight_env.py defines for a fighter other than the root x/y/z slides."""
        joint_names = []
        for side in ("r", "l"):
            joint_names += [f"shoulder_{side}", f"elbow_{side}", f"hip_{side}", f"knee_{side}"]
        joint_names += ["hip_twist", "waist_twist", "waist_bend"]
        return [self.model.joint(f"{name}_{j}").id for j in joint_names]

    # ---- Gymnasium API ----

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        _set_rest_pose_qpos(self.model, self.data, AGENT)
        _set_rest_pose_qpos(self.model, self.data, OPPONENT)
        mujoco.mj_forward(self.model, self.data)

        self._agent_hp = MAX_HP
        self._opp_hp = MAX_HP
        self._step_count = 0
        self._glove_cooldown = {}

        if self.render_mode == "human":
            self._render_frame()
        return self._build_observation(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_agent_action(action)
        self._apply_scripted_opponent()

        mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:, :] = 0
        self._step_count += 1

        agent_hit_scored, agent_damage_taken = self._resolve_hits()
        reward, terminated = self._compute_reward(agent_hit_scored, agent_damage_taken)
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

    def _apply_agent_action(self, action: np.ndarray) -> None:
        agent_x = self.data.xpos[self.agent_id, 0]
        opp_x = self.data.xpos[self.opp_id, 0]
        toward_opponent = np.sign(opp_x - agent_x)
        # action[0] > 0 advances, < 0 retreats -- FightEnv.step's own force formula already
        # supports this (nothing stopped a negative action before; the scripted demo just never
        # passed one), so this is the same mechanic, just actually used both directions.
        self.data.xfrc_applied[self.agent_id, 0] = MOVE_FORCE * action[0] * toward_opponent
        for actuator_id, value in zip(self._agent_actuator_ids, action[1:]):
            self.data.ctrl[actuator_id] = value

    def _apply_scripted_opponent(self) -> None:
        """Identical to fight_env.main()'s policy for a single fighter -- approach until close,
        then throw punches with a hip-leads-torso kinetic chain."""
        agent_x = self.data.xpos[self.agent_id, 0]
        opp_x = self.data.xpos[self.opp_id, 0]
        distance = abs(agent_x - opp_x)
        approaching = distance > 1.3
        move = 1.0 if approaching else 0.2
        direction = np.sign(agent_x - opp_x)
        self.data.xfrc_applied[self.opp_id, 0] = MOVE_FORCE * move * direction

        punch = 0.0 if approaching else (np.sin(self._step_count * 0.3) * 0.5 + 0.5)
        for actuator_id in self._opp_all_actuator_ids:
            self.data.ctrl[actuator_id] = punch
        if not approaching:
            torso_follow = np.sin((self._step_count - 6) * 0.3) * 0.5 + 0.5
            self.data.ctrl[self._opp_waist_twist_id] = torso_follow

    def _resolve_hits(self) -> tuple[float, float]:
        """Scan contacts this step; a glove touching a hittable opponent geom scores, subject to
        a per-glove cooldown so continuous contact can't be farmed for repeated damage."""
        for glove_id in list(self._glove_cooldown):
            self._glove_cooldown[glove_id] = max(0, self._glove_cooldown[glove_id] - 1)

        agent_reward = 0.0
        agent_damage_taken = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            for glove_ids, hittable, cooldown_key, scores_for_agent in (
                (self._agent_glove_ids, self._opp_hittable, "agent", True),
                (self._opp_glove_ids, self._agent_hittable, "opp", False),
            ):
                for glove_id, target_id in ((g1, g2), (g2, g1)):
                    if glove_id not in glove_ids or target_id not in hittable:
                        continue
                    if self._glove_cooldown.get(glove_id, 0) > 0:
                        continue
                    damage = hittable[target_id]
                    self._glove_cooldown[glove_id] = HIT_COOLDOWN_STEPS
                    if scores_for_agent:
                        self._opp_hp = max(0.0, self._opp_hp - damage)
                        agent_reward += damage
                    else:
                        self._agent_hp = max(0.0, self._agent_hp - damage)
                        agent_damage_taken += damage
        return agent_reward, agent_damage_taken

    def _compute_reward(self, hit_scored: float, damage_taken: float) -> tuple[float, bool]:
        reward = hit_scored - damage_taken
        reward -= 0.001  # small time penalty -- discourages stalling out the clock
        distance = abs(self.data.xpos[self.agent_id, 0] - self.data.xpos[self.opp_id, 0])
        reward -= 0.001 * max(0.0, distance - 1.3)  # gentle pull toward engagement range

        terminated = False
        if self._opp_hp <= 0.0:
            reward += 50.0
            terminated = True
        elif self._agent_hp <= 0.0:
            reward -= 50.0
            terminated = True
        return reward, terminated

    def _build_observation(self) -> np.ndarray:
        agent_pos = self.data.xpos[self.agent_id]
        opp_pos = self.data.xpos[self.opp_id]
        rel_pos = opp_pos - agent_pos
        agent_vel = self.data.qvel[self._agent_qvel:self._agent_qvel + 3]
        opp_vel = self.data.qvel[self._opp_qvel:self._opp_qvel + 3]

        own_joints = []
        for joint_id in self._agent_joint_ids:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            dof_adr = self.model.jnt_dofadr[joint_id]
            own_joints.append(self.data.qpos[qpos_adr])
            own_joints.append(self.data.qvel[dof_adr])

        opp_key_rel = []
        for geom_id in self._opp_key_geoms:
            opp_key_rel.append(self.data.geom_xpos[geom_id] - agent_pos)

        return np.concatenate([
            rel_pos,
            agent_vel,
            opp_vel[:2],
            np.array(own_joints, dtype=np.float64),
            np.concatenate(opp_key_rel),
            [self._agent_hp / MAX_HP, self._opp_hp / MAX_HP],
            [np.linalg.norm(rel_pos[:2])],
            [1.0 - self._step_count / MAX_EPISODE_STEPS],
        ]).astype(np.float32)

    def _info(self) -> dict:
        return {"agent_hp": self._agent_hp, "opponent_hp": self._opp_hp, "step": self._step_count}

    def _render_frame(self) -> None:
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.distance = 8
            self._viewer.cam.azimuth = 90
            self._viewer.cam.elevation = -18
        self._viewer.sync()
        time.sleep(self.dt)
