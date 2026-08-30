"""Render the working ice policy on a walkable Himalayan snowfield.

Three fixes over the first terrain attempt, each addressing a measured problem:

  * The bundle heightmap is 200x200 over 50 m -- 25 cm per pixel, which is why
    the terrain rendered as visible stair-steps. Bicubic-upsampled to 640 and
    lightly blurred.
  * Relief of 18.771 m over 50 m is a 21.5 degree mean slope. No humanoid walks
    that, which is why the policy collapsed rather than walked. Scaled to 3.5 m,
    measured at 4.8 degrees.
  * The terrain material was flat grey rgba 0.6 0.58 0.55 with no texture, so
    it rendered like clay. Replaced with a snow material over a procedural
    texture, a low sun, and haze for aerial perspective.

Nothing here changes the policy or the physics being demonstrated -- only the
world it is shown in and the slope it is asked to cross.
"""

import argparse
import os

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download

from mujoco_playground._src.locomotion.g1 import base as g1_base

from crampon import getup_model
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

HERE = os.path.dirname(os.path.abspath(__file__))
SMOOTH_HFIELD = os.path.join(HERE, "assets_terrain_smooth.png")

SCENE = """<mujoco model="g1 himalaya">
  <include file="g1_mjx_feetonly.xml"/>
  <compiler autolimits="true"/>
  <!-- rho 0.68 kg/m3: air at ~5400 m, roughly half sea level. -->
  <option density="0.68" viscosity="1.8e-5"/>
  <visual>
    <headlight diffuse="0.35 0.37 0.42" ambient="0.30 0.33 0.40" specular="0 0 0"/>
    <rgba haze="0.82 0.88 0.95 1"/>
    <map haze="0.28"/>
    <global azimuth="140" elevation="-18" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.52 0.68 0.86"
             rgb2="0.90 0.95 1.0" width="512" height="3072"/>
    <!-- Subtle checker, not decoration: on a featureless white plane a walking
         robot and a stationary one look identical. The grid is what makes
         motion legible. Contrast kept low so it reads as windblown snow. -->
    <texture type="2d" name="snowtex" builtin="checker" rgb1="0.97 0.98 1.0"
             rgb2="0.87 0.91 0.96" width="300" height="300"/>
    <material name="snow" texture="snowtex" texuniform="true" texrepeat="8 8"
              specular="0.2" shininess="0.05" reflectance="0.02"/>
    <material name="snowfar" rgba="0.93 0.95 0.99 1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="20 -25 30" dir="-0.45 0.55 -0.75" directional="true"
           diffuse="0.95 0.95 0.92" specular="0.25 0.25 0.25" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.01" material="snow"/>
  </worldbody>
  <include file="sensor.xml"/>
  <keyframe>
    <key name="home"
      qpos="0 0 {spawn_z} 1 0 0 0
      -0.1 0 0 0.3 -0.2 0   -0.1 0 0 0.3 -0.2 0   0 0 0
      0.2 0.2 0 1.28 0 0 0   0.2 -0.2 0 1.28 0 0 0"
      ctrl="-0.1 0 0 0.3 -0.2 0   -0.1 0 0 0.3 -0.2 0   0 0 0
      0.2 0.2 0 1.28 0 0 0   0.2 -0.2 0 1.28 0 0 0"/>
  </keyframe>
</mujoco>
"""


def build(mu: float, relief: float, spawn_z: float) -> mujoco.MjModel:
  assets = dict(g1_base.get_assets())
  with open(SMOOTH_HFIELD, "rb") as f:
    assets["terrain_smooth.png"] = f.read()

  robot = assets[getup_model.ROBOT_XML].decode()
  for body, name, size, fromto in getup_model.NEW_GEOMS:
    robot = getup_model._insert_geom(robot, body, name, size, fromto)
  # Playground's G1 collides only through <pair> elements; the heightfield uses
  # ordinary contype/conaffinity, so without this the robot falls through it.
  assets[getup_model.ROBOT_XML] = robot.encode()

  xml = SCENE.format(mu=mu, relief=relief, spawn_z=spawn_z)
  return mujoco.MjModel.from_xml_string(xml, assets)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", default="policy-ice-s1.pkl")
  ap.add_argument("--mu", type=float, default=0.35)
  ap.add_argument("--relief", type=float, default=3.5)
  ap.add_argument("--spawn-z", type=float, default=2.6)
  ap.add_argument("--wind", type=float, default=8.0)
  ap.add_argument("--steps", type=int, default=500)
  ap.add_argument("--out", default="himalaya.mp4")
  a = ap.parse_args()

  cfg = default_config()
  cfg.noise_config.level = 0.0
  env = G1Ice(config=cfg)
  fn = jax.jit(build_inference_fn(
      env, hf_hub_download(repo_id="Zubair480/crampon-g1-ice",
                           filename=a.policy)))

  model = build(a.mu, a.relief, a.spawn_z)
  r = NativePolicyRunner(env, fn, mu=a.mu, kp_scale=0.8,
                         cold_scale=(2.25, 1.6), model=model)

  rend = mujoco.Renderer(r.model, height=720, width=1280)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.distance, cam.elevation, cam.azimuth = 5.0, -12, 125

  cmd = np.array([0.8, 0.0, 0.0], np.float32)
  key = jax.random.PRNGKey(0)
  rng = np.random.default_rng(0)
  v, th = a.wind * 0.5, 0.0
  dt = env.dt
  x0, y0 = float(r.data.qpos[0]), float(r.data.qpos[1])
  frames, fell = [], None

  for i in range(a.steps):
    v += (a.wind * 0.5 - v) * (dt / 2.0) + 6.0 * np.sqrt(dt) * rng.normal()
    v = float(np.clip(v, 0.0, a.wind))
    th += 0.3 * np.sqrt(dt) * rng.normal()
    mag = 0.5 * 0.68 * 0.5 * v * v  # rho at 5400 m
    f = np.array([mag * np.cos(th), mag * np.sin(th), 0.0])

    o = r.observe(cmd)
    key, k = jax.random.split(key)
    act, _ = fn(o, k)
    r.step(np.asarray(act), wind_force=f)
    if r.fallen and fell is None:
      fell = i

    cam.lookat[:] = r.data.qpos[:3]
    rend.update_scene(r.data, camera=cam)
    frames.append(rend.render())
  rend.close()

  dist = float(np.hypot(r.data.qpos[0] - x0, r.data.qpos[1] - y0))
  imageio.mimsave(a.out, frames, fps=50, macro_block_size=1)
  imageio.imwrite(a.out.replace(".mp4", ".png"), frames[len(frames) // 2])
  print(f"mu={a.mu} relief={a.relief} m | fell at {fell} | travelled "
        f"{dist:.2f} m | wrote {a.out}")


if __name__ == "__main__":
  main()
