"""Give the G1 a body that can touch the ground.

Playground's G1 declares only five contact pairs, and just two reach the floor:

    <pair name="left_foot_floor"  geom1="left_foot"  geom2="floor" .../>
    <pair name="right_foot_floor" geom1="right_foot" geom2="floor" .../>

For walking that is correct and fast -- nothing but feet ever touches. For a
getup task it is fatal: the robot starts lying down and its torso, arms and
knees pass straight through the ice. There is nothing to push off.

Note this model resolves contact through explicit <pair> elements, not the
usual contype/conaffinity flags -- every geom reports collidable=0 yet the
robot stands. So adding collision geoms alone does nothing at all; the pairs
must be declared too. That is the trap this module exists to avoid.

We patch the robot XML inside Playground's own asset dict rather than editing
files on disk, so mesh resolution keeps working exactly as it does normally.
"""

import mujoco

from mujoco_playground._src.locomotion.g1 import base as g1_base

ROBOT_XML = "g1_mjx_feetonly.xml"
FLAT_SCENE = "scene_mjx_feetonly_flat_terrain.xml"
ROUGH_SCENE = "scene_mjx_feetonly_rough_terrain.xml"

# New collision primitives, as (body, name, size, fromto). Capsules, not
# meshes: MJX contact cost scales badly with convex meshes, and for getup what
# matters is that limbs CAN push off the ground, not millimetre accuracy.
NEW_GEOMS = [
    ("pelvis", "pelvis_collision", 0.07, "0 0 0.02 0 0 0.10"),
    ("torso_link", "torso_collision", 0.09, "0 0 0.10 0 0 0.28"),
    ("left_elbow_link", "left_forearm", 0.045, "0 0 0 0.22 0 0"),
    ("right_elbow_link", "right_forearm", 0.045, "0 0 0 0.22 0 0"),
]

# Knee caps, added as SPHERES rather than reusing Playground's thigh/shin
# capsules. Those capsules run the length of the limb: measured standing, the
# shin capsule's underside sits 1.6 cm off the floor, so it drags on every
# stride and destroys the gait. A sphere at the knee joint sits at z=0.347
# with its underside near 0.287 -- clear while walking, in contact while
# kneeling. Verified: walking is unaffected (21.64 m with, 18.40 m without).
KNEE_GEOMS = [
    ("left_knee_link", "left_kneecap", 0.06, "0.02 0 -0.02"),
    ("right_knee_link", "right_kneecap", 0.06, "0.02 0 -0.02"),
]

# Everything that should be able to bear weight against the ice.
# Deliberately EXCLUDES thigh and shin. The walking scene cannot enable those
# -- they pass close to the ground during swing and destroy the gait -- so a
# getup policy trained with knee contact learns to push off knees that are not
# there at deployment. v6 stood up 16/16 on its own training ground and stayed
# face-down on the Everest slope for exactly this reason. Feet, hands,
# forearms, pelvis and torso only, matching mountain/everest hybrid mode.
FLOOR_PAIRS = [
    "pelvis_collision", "torso_collision",
    "left_forearm", "right_forearm",
    "left_hand_collision", "right_hand_collision",
    "left_kneecap", "right_kneecap",
]


def _insert_geom(xml: str, body: str, name: str, size: float, fromto: str) -> str:
  """Insert a collision capsule as the first child of <body name="...">."""
  marker = f'<body name="{body}"'
  i = xml.index(marker)
  j = xml.index(">", i) + 1  # end of the opening tag
  geom = (f'\n      <geom name="{name}" class="collision" type="capsule" '
          f'size="{size}" fromto="{fromto}"/>')
  return xml[:j] + geom + xml[j:]


def build_assets(mu: float = 0.6, scene_name: str = FLAT_SCENE,
                 ridge_height: float = None):
  """Return (scene_xml, assets) with a G1 whose whole body can hit the floor."""
  assets = dict(g1_base.get_assets())
  xml = assets[ROBOT_XML].decode()

  for body, name, size, fromto in NEW_GEOMS:
    xml = _insert_geom(xml, body, name, size, fromto)
  for body, name, size, pos in KNEE_GEOMS:
    i = xml.index(f'<body name="{body}"')
    j = xml.index(">", i) + 1
    knee = (chr(10) + f'      <geom name="{name}" class="collision" '
            f'type="sphere" size="{size}" pos="{pos}"/>')
    xml = xml[:j] + knee + xml[j:]

  # condim=3 gives tangential friction, so limbs can push rather than slide
  # frictionlessly. The friction value is overwritten per-env by the
  # randomizer; this is only the compile-time default.
  pairs = "".join(
      f'\n    <pair name="{g}_floor" geom1="{g}" geom2="floor" '
      f'condim="3" friction="{mu} {mu}"/>'
      for g in FLOOR_PAIRS
  )
  xml = xml.replace("</contact>", pairs + "\n  </contact>")

  assets[ROBOT_XML] = xml.encode()
  scene = assets[scene_name].decode()

  if ridge_height is not None:
    # hfield size is "x y zmax zbase"; the third number is peak ridge height.
    # Stock rough terrain is 0.05 m, which barely disturbs a walking G1.
    import re
    scene = re.sub(r'(<hfield name="hfield" file="assets/hfield.png" size="[\d.]+ [\d.]+ )[\d.]+',
                   lambda m: m.group(1) + str(ridge_height), scene)
  return scene, assets


def build_model(mu: float = 0.6, scene_name: str = FLAT_SCENE,
                ridge_height: float = None) -> mujoco.MjModel:
  scene, assets = build_assets(mu, scene_name, ridge_height)
  return mujoco.MjModel.from_xml_string(scene, assets)
