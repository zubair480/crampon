"""Everest Base Camp approach scene: moraine trail, seracs, Everest and Nuptse.

Materials follow the real place rather than a snow-globe idea of it. The lower
Khumbu is rock- and dust-covered -- guidebooks describe it as looking more like
a grey desert than a river of ice -- so the moraine is grey rubble, the icefall
apron and seracs are blue-white ice, and the high faces are snow. Three
materials, not one.

Collision uses the hybrid scheme established on the earlier mountain: feet plus
the torso/pelvis/forearm capsules, with bitmasks so those capsules hit the
terrain but never each other. Thighs and shins stay non-colliding because they
pass close to the ground during swing and kill the gait.
"""

import os
import re

import mujoco
import numpy as np

from mujoco_playground._src.locomotion.g1 import base as g1_base

from crampon import getup_model
import make_everest

HERE = os.path.dirname(os.path.abspath(__file__))
HFIELD = os.path.join(HERE, "hfield_everest.bin")

SIZE_M = 400.0
MORAINE_Y = -70.0


def _grid():
  hdr = np.fromfile(HFIELD, dtype="int32", count=2)
  return np.fromfile(HFIELD, dtype="float32", offset=8).reshape(hdr[0], hdr[1])


def _relief():
  """Recompute the relief the generator used, so heights match the hfield."""
  _, meta = make_everest.build(res=_grid().shape[0], size_m=SIZE_M)
  return meta["relief_m"]


RELIEF_M = 188.0  # from make_everest report; keep in sync with the .bin


def wander(x):
  return 16.0 * np.sin(x / 95.0) + 7.0 * np.sin(x / 41.0 + 1.3)


def trail_point(x_m: float):
  """(x, y, z, slope_deg) on the moraine crest at a given easting."""
  a = _grid()
  res = a.shape[0]
  mpp = SIZE_M / res
  y = MORAINE_Y + float(wander(x_m))
  cj = int(np.clip(round(x_m / mpp + (res - 1) / 2.0), 1, res - 2))
  ci = int(np.clip(round(y / mpp + (res - 1) / 2.0), 1, res - 2))
  gy, gx = np.gradient(a * RELIEF_M, mpp)
  slope = float(np.degrees(np.arctan(np.hypot(gy[ci, cj], gx[ci, cj]))))
  return float(x_m), y, float(a[ci, cj]) * RELIEF_M, slope


def seracs(n=26, seed=5):
  """Ice towers flanking the moraine, as they do below Base Camp."""
  rng = np.random.default_rng(seed)
  a = _grid()
  res = a.shape[0]
  mpp = SIZE_M / res
  out = []
  for _ in range(n):
    x = rng.uniform(-150, 150)
    side = rng.choice([-1.0, 1.0])
    y = MORAINE_Y + float(wander(x)) + side * rng.uniform(13.0, 34.0)
    cj = int(np.clip(round(x / mpp + (res - 1) / 2.0), 1, res - 2))
    ci = int(np.clip(round(y / mpp + (res - 1) / 2.0), 1, res - 2))
    z = float(a[ci, cj]) * RELIEF_M
    h = rng.uniform(1.6, 5.2)
    w = rng.uniform(0.7, 2.0)
    d = rng.uniform(0.7, 1.8)
    # Sunk in a little and tilted, the way real seracs lean.
    out.append((x, y, z + h * 0.55, w, d, h,
                rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(0, 180)))
  return out


def boulders(n=70, seed=9):
  """Rocks breaking through the snow along and beside the moraine.

  Kept clear of the walking line itself -- the policy has never seen an
  obstacle and would simply trip on every one, which is not the demo. They sit
  beside the trail, which is also where they sit on the real moraine.
  """
  rng = np.random.default_rng(seed)
  a = _grid()
  res = a.shape[0]
  mpp = SIZE_M / res
  out = []
  for _ in range(n):
    x = rng.uniform(-165, 165)
    side = rng.choice([-1.0, 1.0])
    y = MORAINE_Y + float(wander(x)) + side * rng.uniform(4.5, 26.0)
    cj = int(np.clip(round(x / mpp + (res - 1) / 2.0), 1, res - 2))
    ci = int(np.clip(round(y / mpp + (res - 1) / 2.0), 1, res - 2))
    z = float(a[ci, cj]) * RELIEF_M
    r = rng.uniform(0.25, 1.1)
    # Sunk 35-60% into the snow, so they read as embedded not dropped on top.
    out.append((x, y, z + r * rng.uniform(0.40, 0.65), r,
                rng.uniform(0.65, 1.0), rng.uniform(0.6, 1.0),
                rng.uniform(0, 180), rng.uniform(-25, 25)))
  return out


SCENE = """<mujoco model="everest base camp approach">
  <include file="g1_mjx_feetonly.xml"/>
  <compiler autolimits="true"/>
  <!-- rho 0.60 kg/m3: air at ~5400 m, roughly half sea level. -->
  <option density="0.60" viscosity="1.8e-5"/>
  <visual>
    <headlight diffuse="0.22 0.24 0.29" ambient="0.26 0.30 0.38" specular="0.03 0.03 0.03"/>
    <rgba haze="0.74 0.82 0.92 1"/>
    <global azimuth="120" elevation="-12" offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
    <map shadowclip="200" shadowscale="0.4" haze="0.22"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.13 0.32 0.66"
             rgb2="0.68 0.82 0.96" width="768" height="4096"/>
    <!-- Texture painted from the heightfield itself: snow above the snowline
         and on anything not too steep, bare rock below and on near-vertical
         faces. That is why the moraine reads as grey rubble while the Nuptse
         wall is striped -- the same reason it looks that way in photographs.
         texrepeat 1 1 so the image maps once across the whole terrain. -->
    <texture type="2d" name="terraintex" file="everest_tex.png"/>
    <material name="moraine" texture="terraintex" texrepeat="1 1"
              texuniform="false" specular="0.05" shininess="0.02"/>
    <material name="ice" rgba="0.80 0.90 0.97 1" specular="0.55"
              shininess="0.55" reflectance="0.06"/>
    <material name="rock" rgba="0.33 0.31 0.29 1" specular="0.08"
              shininess="0.05"/>
    <hfield name="ebc" file="hfield_everest.bin"
            size="{half} {half} {relief} 3.0"/>
  </asset>
  <worldbody>
    <light name="sun" pos="-120 -300 300" dir="0.28 0.72 -0.63" directional="true"
           diffuse="1.0 0.98 0.94" specular="0.35 0.35 0.33" castshadow="true"/>
    <geom name="snow" type="hfield" hfield="ebc" material="moraine"
          pos="0 0 0" contype="1" conaffinity="2" condim="3"
          friction="{mu} {tors} 0.001"/>
{seracs}
{boulders}
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


def build_model(mu: float = 0.35, mode: str = "hybrid", start_x: float = -120.0,
                stand_h: float = 0.85, tors: float = 0.6,
                n_seracs: int = 26) -> mujoco.MjModel:
  assert mode in ("feet", "hybrid")
  assets = dict(g1_base.get_assets())
  with open(HFIELD, "rb") as f:
    assets["hfield_everest.bin"] = f.read()
  with open(os.path.join(HERE, "everest_tex.png"), "rb") as f:
    assets["everest_tex.png"] = f.read()

  robot = assets[getup_model.ROBOT_XML].decode()
  if mode == "hybrid":
    for b, n, sz, ft in getup_model.NEW_GEOMS:
      robot = getup_model._insert_geom(robot, b, n, sz, ft)
      robot = robot.replace(
          f'<geom name="{n}" class="collision"',
          f'<geom name="{n}" class="collision" contype="2" conaffinity="1" '
          f'condim="3" friction="{mu} {tors} 0.001"')
    # Knee caps: contact when kneeling, clear while walking. Training and
    # deployment must expose the same contacts or the getup policy reaches for
    # geometry that is not there.
    for b, n, sz, pos in getup_model.KNEE_GEOMS:
      i = robot.index(f'<body name="{b}"')
      j = robot.index(">", i) + 1
      cap = (chr(10) + f'      <geom name="{n}" class="collision" '
             f'type="sphere" size="{sz}" pos="{pos}" contype="2" '
             f'conaffinity="1" condim="3" friction="{mu} {tors} 0.001"/>')
      robot = robot[:j] + cap + robot[j:]

    # Hands too: getup pushes off them, so deployment must expose the same
    # contacts. Hands sit high during walking and cost nothing.
    for hn in ("left_hand_collision", "right_hand_collision"):
      robot = robot.replace(
          f'<geom name="{hn}" class="collision"',
          f'<geom name="{hn}" class="collision" contype="2" conaffinity="1" '
          f'condim="3" friction="{mu} {tors} 0.001"')

  robot = robot.replace(
      '<geom size="0.085 0.03 0.005"/>',
      '<geom size="0.085 0.03 0.005" contype="2" conaffinity="1" '
      f'condim="3" friction="{mu} {tors} 0.001"/>')
  robot = re.sub(r'\s*<pair[^>]*geom2="floor"[^>]*/>', "", robot)
  assets[getup_model.ROBOT_XML] = robot.encode()

  s = assets["sensor.xml"].decode()
  s = (s.replace('geom1="floor"', 'geom1="snow"')
        .replace('geom2="floor"', 'geom2="snow"'))
  assets["sensor.xml"] = s.encode()

  blocks = "\n".join(
      f'    <geom name="serac{i}" type="box" material="ice" '
      f'size="{w:.2f} {d:.2f} {h:.2f}" pos="{x:.2f} {y:.2f} {z:.2f}" '
      f'euler="{ex:.0f} {ey:.0f} {ez:.0f}" contype="1" conaffinity="2" '
      f'condim="3" friction="0.15 0.05 0.001"/>'
      for i, (x, y, z, w, d, h, ex, ey, ez) in enumerate(seracs(n_seracs)))

  rocks = chr(10).join(
      f'    <geom name="rock{i}" type="ellipsoid" material="rock" '
      f'size="{r:.2f} {r*sy_:.2f} {r*sz_:.2f}" pos="{x:.2f} {y:.2f} {z:.2f}" '
      f'euler="{ex:.0f} 0 {ez:.0f}" contype="1" conaffinity="2" condim="3" '
      f'friction="0.8 0.4 0.001"/>'
      for i, (x, y, z, r, sy_, sz_, ez, ex) in enumerate(boulders()))

  sx, sy, sz, _ = trail_point(start_x)
  yaw = 0.0  # walk east along the moraine
  xml = SCENE.format(half=SIZE_M / 2, relief=RELIEF_M, mu=mu, tors=tors,
                     seracs=blocks, boulders=rocks, sx=sx, sy=sy, sz=sz + stand_h,
                     qw=float(np.cos(yaw / 2)), qz=float(np.sin(yaw / 2)))
  return mujoco.MjModel.from_xml_string(xml, assets)
