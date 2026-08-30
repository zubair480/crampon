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

# New collision primitives, as (body, name, size, fromto). Capsules, not
# meshes: MJX contact cost scales badly with convex meshes, and for getup what
# matters is that limbs CAN push off the ground, not millimetre accuracy.
NEW_GEOMS = [
    ("pelvis", "pelvis_collision", 0.09, "0 0 -0.02 0 0 0.06"),
    ("torso_link", "torso_collision", 0.10, "0 0 0.06 0 0 0.26"),
    ("left_elbow_link", "left_forearm", 0.045, "0 0 0 0.22 0 0"),
    ("right_elbow_link", "right_forearm", 0.045, "0 0 0 0.22 0 0"),
]

# Everything that should be able to bear weight against the ice.
FLOOR_PAIRS = [
    "pelvis_collision", "torso_collision",
    "left_forearm", "right_forearm",
    "left_hand_collision", "right_hand_collision",
    "left_thigh", "right_thigh",
    "left_shin", "right_shin",
]


def _insert_geom(xml: str, body: str, name: str, size: float, fromto: str) -> str:
  """Insert a collision capsule as the first child of <body name="...">."""
  marker = f'<body name="{body}"'
  i = xml.index(marker)
  j = xml.index(">", i) + 1  # end of the opening tag
  geom = (f'\n      <geom name="{name}" class="collision" type="capsule" '
          f'size="{size}" fromto="{fromto}"/>')
  return xml[:j] + geom + xml[j:]


def build_assets(mu: float = 0.6):
  """Return (scene_xml, assets) with a G1 whose whole body can hit the floor."""
  assets = dict(g1_base.get_assets())
  xml = assets[ROBOT_XML].decode()

  for body, name, size, fromto in NEW_GEOMS:
    xml = _insert_geom(xml, body, name, size, fromto)

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
  scene = assets["scene_mjx_feetonly_flat_terrain.xml"].decode()
  return scene, assets


def build_model(mu: float = 0.6) -> mujoco.MjModel:
  scene, assets = build_assets(mu)
  return mujoco.MjModel.from_xml_string(scene, assets)
