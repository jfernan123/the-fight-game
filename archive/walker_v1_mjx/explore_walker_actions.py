"""Interactively drag torque sliders for the walker's action-space joints and watch them respond
-- uses the exact same torque-actuator model walk_gym.py trains against, not fight_env.py's
--explore mode (which uses position actuators and behaves completely differently).

    python explore_walker_actions.py

Drag sliders in the viewer's "Control" panel for:
    red_hip_r_m, red_hip_l_m, red_knee_r_m, red_knee_l_m, red_hip_twist_m
Each ranges -1 to 1 -- a torque *fraction*, not a target angle (this is a motor, not a position
actuator: holding it at 1 applies constant maximum torque, it does not drive to and hold "the
joint at 1"). The model has other actuators (arms, waist) too; they exist because this is the
same shared body-builder as the rest of the project, but they're not part of the walker's action
space -- ignore them here.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root, for fight_env

import mujoco
import mujoco.viewer

from fight_env import _actuators, _build_model_xml, _set_rest_pose_qpos

AGENT = "red"

model = mujoco.MjModel.from_xml_string(_build_model_xml(_actuators, fighters=(AGENT,), free_torso=True))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
_set_rest_pose_qpos(model, data, AGENT)
mujoco.mj_forward(model, data)

print("Drag sliders in the viewer's Control panel for:")
print("  red_hip_r_m, red_hip_l_m, red_knee_r_m, red_knee_l_m, red_hip_twist_m")
print("(other actuators exist on the shared model but aren't part of the walker's action space)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 6
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -18
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
