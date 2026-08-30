"""A complete ice mountain scene: summit, ridgelines, and a hikeable traverse.

Collision follows what was measured on the Khumbu DEM: only the feet touch the
world, because with every group-3 geom colliding the robot's shin and torso
capsules catch on the terrain and it walks 0.81 m in a crouch instead of 7.30 m
upright. Full-body contact is still available (mode="body") since fall recovery
needs it -- the two tasks genuinely want different collision models.
"""

import os
import re

import mujoco
import numpy as np

from mujoco_playground._src.locomotion.g1 import base as g1_base

from crampon import getup_model

HERE = os.path.dirname(os.path.abspath(__file__))
HFIELD = os.path.join(HERE, "hfield_mountain.bin")

SIZE_M = 400.0
PEAK_M = 120.0
BENCH_R = 110.0


def _grid():
  hdr = np.fromfile(HFIELD, dtype="int32", count=2)
  return np.fromfile(HFIELD, dtype="float32", offset=8).reshape(hdr[0], hdr[1])


def bench_spawn(theta_deg: float = 200.0):
  """A point on the graded traverse, with its terrain height.

  Returns (x, y, z_terrain, local_slope_deg). Row index maps directly to +y --
  verified on the Khumbu field by dropping the robot and comparing where it
  settled against prediction (2.6 m out mirrored, 0.12 m correct).
  """
  a = _grid()
  res = a.shape[0]
  mpp = SIZE_M / res
  th = np.radians(theta_deg)
  x, y = BENCH_R * np.cos(th), BENCH_R * np.sin(th)
  cj = int(round(x / mpp + (res - 1) / 2.0))
  ci = int(round(y / mpp + (res - 1) / 2.0))
  ci = int(np.clip(ci, 1, res - 2))
  cj = int(np.clip(cj, 1, res - 2))
  gy, gx = np.gradient(a * PEAK_M, mpp)
  slope = float(np.degrees(np.arctan(np.hypot(gy[ci, cj], gx[ci, cj]))))
  return float(x), float(y), float(a[ci, cj]) * PEAK_M, slope


SCENE = """<mujoco model="g1 ice mountain">
  <include file="g1_mjx_feetonly.xml"/>
  <compiler autolimits="true"/>
  <!-- rho 0.68 kg/m3: air at ~5400 m, roughly half sea level. -->
  <option density="0.68" viscosity="1.8e-5"/>
  <visual>
    <headlight diffuse="0.18 0.20 0.24" ambient="0.13 0.16 0.22" specular="0.05 0.05 0.05"/>
    <rgba haze="0.80 0.86 0.94 1"/>
    <global azimuth="140" elevation="-15" offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
    <map shadowclip="120" shadowscale="0.5" haze="0.12"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.20 0.42 0.75"
             rgb2="0.72 0.85 0.97" width="768" height="4096"/>
    <texture type="2d" name="snowtex" builtin="flat" mark="random" random="0.05"
             rgb1="0.93 0.95 0.98" markrgb="0.80 0.85 0.92"
             width="1024" height="1024"/>
    <material name="ice" texture="snowtex" texrepeat="90 90" texuniform="true"
              rgba="0.97 0.98 1 1" specular="0.15" shininess="0.06"
              reflectance="0.0"/>
    <hfield name="mountain" file="hfield_mountain.bin"
            size="{half} {half} {peak} 2.0"/>
  </asset>
  <worldbody>
    <light name="sun" pos="120 -180 260" dir="-0.35 0.5 -0.79" directional="true"
           diffuse="0.98 0.97 0.93" specular="0.30 0.30 0.28" castshadow="true"/>
    <geom name="snow" type="hfield" hfield="mountain" material="ice"
          pos="0 0 0" contype="1" conaffinity="2" condim="3"
          friction="{mu} {tors} 0.001"
          priority="1" solref="0.02 1.0" solimp="0.9 0.99 0.004 0.5 2"/>
  </worldbody>
  <include file="sensor.xml"/>
  <keyframe>
    <key name="home"
      qpos="{sx} {sy} {sz} {qw} 0 0 {qz}
      -0.1 0 0 0.3 -0.2 0   -0.1 0 0 0.3 -0.2 0   0 0 0
      0.2 0.2 0 1.28 0 0 0   0.2 -0.2 0 1.28 0 0 0"
      ctrl="-0.1 0 0 0.3 -0.2 0   -0.1 0 0 0.3 -0.2 0   0 0 0
      0.2 0.2 0 1.28 0 0 0   0.2 -0.2 0 1.28 0 0 0"/>
  </keyframe>
</mujoco>
"""


def build_model(mu: float = 0.35, mode: str = "feet", theta_deg: float = 200.0,
                stand_h: float = 0.80, tors: float = 0.6) -> mujoco.MjModel:
  """mode='feet' for walking, 'body' for fall recovery."""
  assert mode in ("feet", "body", "hybrid")
  assets = dict(g1_base.get_assets())
  with open(HFIELD, "rb") as f:
    assets["hfield_mountain.bin"] = f.read()

  robot = assets[getup_model.ROBOT_XML].decode()
  if mode == "hybrid":
    # Feet plus the torso/pelvis/forearm capsules, but NOT thigh or shin.
    # Measured on this traverse: with every group-3 geom colliding the robot
    # falls at step 55 even on grippy snow, because thighs and shins pass
    # close to the ground during normal swing and catch. Those two are the
    # only ones that break the gait; the upper-body capsules never touch while
    # walking, and they are exactly what stops a fallen torso sinking through.
    for b, n, sz, ft in getup_model.NEW_GEOMS:
      robot = getup_model._insert_geom(robot, b, n, sz, ft)
      robot = robot.replace(
          f'<geom name="{n}" class="collision"',
          f'<geom name="{n}" class="collision" contype="2" conaffinity="1" '
          f'condim="3" friction="{mu} {tors} 0.001"')
    robot = robot.replace(
        '<geom size="0.085 0.03 0.005"/>',
        '<geom size="0.085 0.03 0.005" contype="2" conaffinity="1" '
        f'condim="3" friction="{mu} {tors} 0.001"/>')
  elif mode == "body":
    for b, n, s, ft in getup_model.NEW_GEOMS:
      robot = getup_model._insert_geom(robot, b, n, s, ft)
    robot = robot.replace(
        '<geom group="3" rgba=".2 .6 .2 .3"/>',
        '<geom group="3" rgba=".2 .6 .2 .3" contype="1" conaffinity="1" '
        'condim="3"/>')
  else:
    robot = robot.replace(
        '<geom size="0.085 0.03 0.005"/>',
        '<geom size="0.085 0.03 0.005" contype="1" conaffinity="1" '
        f'condim="3" friction="{mu} {tors} 0.001"/>')
  robot = re.sub(r'\s*<pair[^>]*geom2="floor"[^>]*/>', "", robot)
  assets[getup_model.ROBOT_XML] = robot.encode()

  s = assets["sensor.xml"].decode()
  s = (s.replace('geom1="floor"', 'geom1="snow"')
        .replace('geom2="floor"', 'geom2="snow"'))
  assets["sensor.xml"] = s.encode()

  sx, sy, sz, _ = bench_spawn(theta_deg)
  # Face along the traverse (tangent to the bench) rather than into the slope.
  yaw = np.radians(theta_deg) + np.pi / 2.0
  xml = SCENE.format(half=SIZE_M / 2, peak=PEAK_M, mu=mu, tors=tors,
                     sx=sx, sy=sy, sz=sz + stand_h,
                     qw=float(np.cos(yaw / 2)), qz=float(np.sin(yaw / 2)))
  return mujoco.MjModel.from_xml_string(xml, assets)
