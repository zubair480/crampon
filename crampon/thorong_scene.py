"""Put Playground's G1 on real Thorong La terrain.

The terrain bundle ships a 50 m patch of Thorong La (5416 m, Annapurna
Circuit) built from SRTM elevation data, plus 45 procedural rock obstacles and
air density 0.68 kg/m3. It also vendors the raw Unitree G1 -- but that model
has none of the sensors our policies read (gyro_pelvis, local_linvel_pelvis,
upvector_torso), so we keep Playground's robot and splice in only the world.

The collision models differ and that is the crux. Playground's G1 collides
through explicit <pair> elements and carries contype=0/conaffinity=0 on every
geom; the terrain and rocks use ordinary contype/conaffinity. Left alone the
robot falls straight through the mountain. We therefore turn ordinary
collision back on for the robot's collision geoms only (group 3), leaving its
visual meshes untouched.
"""

import os
import re

import mujoco

from mujoco_playground._src.locomotion.g1 import base as g1_base

from crampon import getup_model

BUNDLE = os.environ.get(
    "TERRAIN_BUNDLE",
    r"C:\Users\zubai\Downloads\terrain_bundle\terrain")

# Level footing graded into the slope by the bundle's generator.
SPAWN_XY = (-17.462, 0.879)


def _bundle_assets() -> dict:
  """terrain.png plus the rock meshes, as bytes for the XML compiler."""
  out = {}
  adir = os.path.join(BUNDLE, "assets")
  for name in os.listdir(adir):
    if name.endswith((".png", ".obj")):
      with open(os.path.join(adir, name), "rb") as f:
        out[name] = f.read()
  return out


def _extract(xml: str, tag: str) -> str:
  """Pull the inner text of the first <tag>...</tag> block."""
  m = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.S)
  return m.group(1) if m else ""


def build_assets(mu: float = 0.4, spawn_z: float = 0.90):
  """Return (scene_xml, assets) for Playground's G1 on Thorong La."""
  assets = dict(g1_base.get_assets())
  assets.update(_bundle_assets())

  # Robot XML with full-body collision capsules (needed for fall recovery),
  # then ordinary collision enabled so it can touch terrain and rocks at all.
  robot = assets[getup_model.ROBOT_XML].decode()
  for body, name, size, fromto in getup_model.NEW_GEOMS:
    robot = getup_model._insert_geom(robot, body, name, size, fromto)
  robot = robot.replace(
      '<geom group="3" rgba=".2 .6 .2 .3"/>',
      f'<geom group="3" rgba=".2 .6 .2 .3" contype="1" conaffinity="1" '
      f'condim="3" friction="{mu} {mu} 0.001"/>')
  # Drop every pair against "floor": that geom does not exist in this scene
  # (the ground here is the hfield "terrain"), and with ordinary collision
  # enabled above the pairs are redundant anyway.
  robot = re.sub(r'\s*<pair[^>]*geom2="floor"[^>]*/>', "", robot)
  assets[getup_model.ROBOT_XML] = robot.encode()

  # sensor.xml has foot-contact sensors that name the geom "floor"; here the
  # ground is the hfield "terrain". Repoint the geom references only -- the
  # sensor NAMES stay as they are, because the env looks them up by name.
  sensors = assets["sensor.xml"].decode()
  sensors = sensors.replace('geom1="floor"', 'geom1="terrain"').replace('geom2="floor"', 'geom2="terrain"')
  assets["sensor.xml"] = sensors.encode()

  with open(os.path.join(BUNDLE, "scene_g1_terrain.xml"), encoding="utf-8") as f:
    bundle_scene = f.read()
  world = _extract(bundle_scene, "worldbody")
  asset_block = _extract(bundle_scene, "asset")
  # Drop the bundle's skybox; we keep a bright one for a snow scene.
  asset_block = re.sub(r"<texture type=\"skybox\".*?/>", "", asset_block, flags=re.S)

  scene = f"""<mujoco model="g1 thorong la">
  <include file="{getup_model.ROBOT_XML}"/>
  <compiler autolimits="true"/>
  <!-- rho 0.68 kg/m3: air at ~5400 m, about half sea level. -->
  <option density="0.68" viscosity="1.8e-5"/>
  <visual>
    <headlight diffuse="0.6 0.62 0.68" ambient="0.35 0.37 0.42" specular="0.1 0.1 0.1"/>
    <rgba haze="0.86 0.91 0.97 1"/>
    <global azimuth="140" elevation="-20" offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.85 0.95" rgb2="1 1 1"
             width="512" height="3072"/>
    {asset_block}
  </asset>
  {f"<worldbody>{world}</worldbody>"}
  <include file="sensor.xml"/>
  <keyframe>
    <key name="home"
      qpos="{SPAWN_XY[0]} {SPAWN_XY[1]} {spawn_z} 1 0 0 0
      -0.1 0 0 0.3 -0.2 0
      -0.1 0 0 0.3 -0.2 0
      0 0 0
      0.2 0.2 0 1.28 0 0 0
      0.2 -0.2 0 1.28 0 0 0"
      ctrl="
      -0.1 0 0 0.3 -0.2 0
      -0.1 0 0 0.3 -0.2 0
      0 0 0
      0.2 0.2 0 1.28 0 0 0
      0.2 -0.2 0 1.28 0 0 0"/>
  </keyframe>
</mujoco>
"""
  return scene, assets


def build_model(mu: float = 0.4, spawn_z: float = 0.90) -> mujoco.MjModel:
  scene, assets = build_assets(mu, spawn_z)
  return mujoco.MjModel.from_xml_string(scene, assets)
