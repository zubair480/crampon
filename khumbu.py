"""Playground's 29-DoF G1 on the real Khumbu DEM, with the soft snow layer.

The supplied scene_himalaya.xml pairs that terrain with g1_12dof.xml -- a
legs-only G1. Our policy emits 29 actions and reads 29 joint angles, so it
cannot drive that robot. This keeps Playground's 29-DoF model and takes only
the world: the 512x512 Khumbu heightfield (0.234 m/cell, from a float32 binary
rather than an 8-bit PNG, so no quantisation stair-stepping) and the two-layer
ground.

The snow layer is the interesting part and is worth preserving exactly: an 8 cm
skin over the rigid base with priority=1 and soft contact parameters
(solref 0.280 1.0, solimp 0.03 0.95 0.160 0.5 2), so the foot sinks into it
instead of striking a rigid plane.

Collision has to be reconciled. Playground's G1 carries contype=0 on every geom
and collides only through explicit <pair> elements naming "floor"; this terrain
uses ordinary contype/conaffinity. Two ways to bridge that, and they are not
equivalent:

  mode="pair"    rename the snow geom to "floor" so the trained pairs bind.
                 Keeps exactly the contact model the policy learned, but pair
                 parameters override the snow's softness.
  mode="contype" enable ordinary collision on the robot. The snow layer's
                 priority and solref/solimp then govern the contact, which is
                 the whole point of the soft layer -- but it is not what the
                 policy trained against.

Both are built here so the difference can be measured rather than assumed.
"""

import os

import mujoco
import numpy as np

from mujoco_playground._src.locomotion.g1 import base as g1_base

from crampon import getup_model

DEM = os.environ.get(
    "KHUMBU_DEM",
    r"C:\Users\zubai\Downloads\dem_extract\dem\hfield_khumbu.bin")

SIZE = 120.0          # arena side, metres
RELIEF = 36.314375    # vertical relief, metres
BASE_Z = -13.3864     # rigid base offset from the supplied scene
SNOW_Z = -13.3064     # 8 cm of snow above it


def flattest_spawn(margin_m: float = 6.0):
  """World (x, y, z) of the flattest patch -- the only sane place to start."""
  hdr = np.fromfile(DEM, dtype="int32", count=2)
  a = np.fromfile(DEM, dtype="float32", offset=8).reshape(hdr[0], hdr[1])
  mpp = SIZE / a.shape[0]
  gy, gx = np.gradient(a * RELIEF, mpp)
  slope = np.degrees(np.arctan(np.hypot(gy, gx)))
  k = max(2, int(margin_m / mpp))
  from numpy.lib.stride_tricks import sliding_window_view
  w = sliding_window_view(slope, (k, k)).mean(axis=(2, 3))
  i, j = np.unravel_index(np.argmin(w), w.shape)
  ci, cj = i + k // 2, j + k // 2
  x = (cj / a.shape[1] - 0.5) * SIZE
  # MuJoCo hfield rows run along +y starting at -y, so row index maps
  # directly. Verified by dropping the robot: at the mirrored y it settled
  # 2.6 m above the predicted surface, at this one it matched to 0.12 m.
  y = (ci / a.shape[0] - 0.5) * SIZE
  z = float(a[ci, cj]) * RELIEF + SNOW_Z
  return float(x), float(y), z, float(w[i, j])


SCENE = """<mujoco model="g1 khumbu">
  <include file="g1_mjx_feetonly.xml"/>
  <compiler autolimits="true"/>
  <!-- rho 0.68 kg/m3: air at ~5400 m, roughly half sea level. -->
  <option density="0.68" viscosity="1.8e-5"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.38" ambient="0.22 0.24 0.30" specular="0.1 0.1 0.1"/>
    <rgba haze="0.88 0.90 0.95 1"/>
    <global azimuth="140" elevation="-18" offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
    <map shadowclip="60" shadowscale="0.5" haze="0.25"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.68 0.85"
             rgb2="0.90 0.94 1.0" width="512" height="3072"/>
    <texture type="2d" name="snowtex" builtin="flat" mark="random" random="0.08"
             rgb1="0.86 0.89 0.94" markrgb="0.72 0.78 0.88"
             width="512" height="512"/>
    <material name="ground" texture="snowtex" texrepeat="40 40" texuniform="true"
              rgba="0.95 0.96 1 1" reflectance="0.0" specular="0.2" shininess="0.1"/>
    <hfield name="terrain" file="hfield_khumbu.bin"
            size="{half} {half} {relief} 0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 40" dir="-0.4 0.3 -1" directional="true"
           diffuse="0.6 0.6 0.66" castshadow="true"/>
    <geom name="rock" type="hfield" hfield="terrain" material="ground"
          pos="0 0 {base_z}" group="3"
          contype="{ct}" conaffinity="{ct}" condim="3" friction="{mu} 0.01 0.001"/>
    <!-- 8 cm of soft snow over the rigid base; priority 1 so it wins the
         contact, with compliant solref/solimp so the foot sinks in. -->
    <geom name="{snow_name}" type="hfield" hfield="terrain" material="ground"
          pos="0 0 {snow_z}"
          contype="{ct}" conaffinity="{ct}" condim="3" friction="{mu} 0.01 0.001"
          priority="1" solref="0.280 1.0" solimp="0.03 0.95 0.160 0.5 2"/>
  </worldbody>
  <include file="sensor.xml"/>
  <keyframe>
    <key name="home"
      qpos="{sx} {sy} {sz} 1 0 0 0
      -0.1 0 0 0.3 -0.2 0   -0.1 0 0 0.3 -0.2 0   0 0 0
      0.2 0.2 0 1.28 0 0 0   0.2 -0.2 0 1.28 0 0 0"
      ctrl="-0.1 0 0 0.3 -0.2 0   -0.1 0 0 0.3 -0.2 0   0 0 0
      0.2 0.2 0 1.28 0 0 0   0.2 -0.2 0 1.28 0 0 0"/>
  </keyframe>
</mujoco>
"""


def build_model(mode: str = "pair", mu: float = 0.5,
                stand_h: float = 0.80) -> mujoco.MjModel:
  assert mode in ("pair", "contype", "feet")
  assets = dict(g1_base.get_assets())
  with open(DEM, "rb") as f:
    assets["hfield_khumbu.bin"] = f.read()

  robot = assets[getup_model.ROBOT_XML].decode()
  if mode != "feet":
    for body, name, size, fromto in getup_model.NEW_GEOMS:
      robot = getup_model._insert_geom(robot, body, name, size, fromto)
  if mode == "feet":
    # Only the feet collide with the world, which is exactly what the policy
    # trained against. Measured: with every group-3 geom colliding the robot
    # walks 0.81 m and stays crouched, because its shin and torso capsules
    # catch on the terrain; feet-only it walks 7.30 m over 367 steps.
    # The full-body capsules are still required for fall recovery -- the two
    # tasks genuinely want different collision models.
    robot = robot.replace(
        '<geom size="0.085 0.03 0.005"/>',
        '<geom size="0.085 0.03 0.005" contype="1" conaffinity="1" '
        'condim="3" friction="0.6 0.6 0.001"/>')
  if mode == "contype":
    robot = robot.replace(
        '<geom group="3" rgba=".2 .6 .2 .3"/>',
        '<geom group="3" rgba=".2 .6 .2 .3" contype="1" conaffinity="1" '
        'condim="3"/>')
  assets[getup_model.ROBOT_XML] = robot.encode()

  sx, sy, sz, _ = flattest_spawn()
  xml = SCENE.format(
      half=SIZE / 2, relief=RELIEF, base_z=BASE_Z, snow_z=SNOW_Z, mu=mu,
      ct=0 if mode == "pair" else 1,
      # In "pair" mode the snow geom must be called floor, because that is the
      # name every trained contact pair refers to.
      snow_name="floor" if mode == "pair" else "snow",
      sx=sx, sy=sy, sz=sz + stand_h)
  if mode in ("contype", "feet"):
    # No geom named floor in these modes; drop the pairs that reference it and
    # repoint the foot-contact sensors at the snow layer.
    import re
    robot = re.sub(r'\s*<pair[^>]*geom2="floor"[^>]*/>', "",
                   assets[getup_model.ROBOT_XML].decode())
    assets[getup_model.ROBOT_XML] = robot.encode()
    s = assets["sensor.xml"].decode()
    s = (s.replace('geom1="floor"', 'geom1="snow"')
          .replace('geom2="floor"', 'geom2="snow"'))
    assets["sensor.xml"] = s.encode()
  return mujoco.MjModel.from_xml_string(xml, assets)
