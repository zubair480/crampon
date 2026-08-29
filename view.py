"""Interactive MuJoCo viewer for the G1 ice scene, with live physics.

Playground is a headless library -- this is the only window you get.

Note on geom groups: the G1 model keeps its visual meshes and its collision
geoms in different groups, and the viewer disables some by default. That is
why a bare launch shows only two dark blobs (the foot collision geoms) instead
of a robot. We force every group visible.

Run:  python view.py
Keys: 0-4 toggle geom groups, space pauses, left-drag orbits, scroll zooms,
      ctrl+left-drag shoves the robot, backspace resets.
"""

import time

import mujoco
import mujoco.viewer

from crampon.ice_env import G1Ice

MU = 0.05  # ice

env = G1Ice()
model = env.mj_model
data = mujoco.MjData(model)

# Pin the floor to ice so what you watch matches what we train on.
model.pair_friction[0:2, 0:2] = MU

if model.nkey > 0:
  mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Position servos: command the default pose so it tries to hold itself up.
data.ctrl[:] = data.qpos[7:]

print(f"viewer open | mu={MU} | {model.ngeom} geoms", flush=True)

with mujoco.viewer.launch_passive(model, data) as viewer:
  viewer.opt.geomgroup[:] = 1  # <-- show ALL geom groups, meshes included
  viewer.cam.distance = 3.2
  viewer.cam.elevation = -12
  viewer.cam.azimuth = 135
  viewer.cam.lookat[:] = [0.0, 0.0, 0.7]

  while viewer.is_running():
    step_start = time.time()
    mujoco.mj_step(model, data)
    viewer.sync()
    dt = model.opt.timestep - (time.time() - step_start)
    if dt > 0:
      time.sleep(dt)
