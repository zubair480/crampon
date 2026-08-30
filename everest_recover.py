"""Hike the moraine, slip on the ice, stand back up, keep walking.

Two policies under a supervisor. Neither can do this alone: the walking policy
has no idea how to stand up from its back, and the getup policy has no idea how
to walk. The switch is on the torso up-vector, which is the same quantity the
walking env uses to decide an episode has failed, so the trigger is not
invented for the demo.

    WALK  --(up < FALL_ENTER)-->                      GETUP
    GETUP --(up > STAND_EXIT and height clear, held)--> WALK

Two thresholds rather than one, because a single threshold flaps every time the
robot wobbles across it; and recovery must persist for HOLD steps so control is
not handed back mid-stagger.

The scene uses hybrid collision -- feet plus the torso/pelvis/forearm capsules,
with bitmasks so those capsules meet the terrain but never each other. Feet-only
would let the fallen torso sink through the snow; full-body kills the gait
because thighs and shins catch during swing.
"""

import argparse

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download

import everest
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

REPO = "Zubair480/crampon-g1-ice"

FALL_ENTER = 0.30   # below this uprightness we have fallen
STAND_EXIT = 0.88   # above this, and tall enough, we are back up
CLEAR_H = 0.55      # metres of torso above local ground
HOLD = 30           # steps recovery must persist before walking resumes


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--walk", default="policy-ice-s1.pkl")
  ap.add_argument("--getup", default="policy-getup-v5.pkl")
  ap.add_argument("--mu", type=float, default=0.5)
  ap.add_argument("--steps", type=int, default=1400)
  ap.add_argument("--start-x", type=float, default=-120.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default="everest_recover.mp4")
  ap.add_argument("--no-video", action="store_true")
  a = ap.parse_args()

  cfg = default_config()
  cfg.noise_config.level = 0.0
  env = G1Ice(config=cfg)

  walk_fn = jax.jit(build_inference_fn(
      env, hf_hub_download(repo_id=REPO, filename=a.walk)))
  if a.getup == 'zero':
    import jax.numpy as jp
    getup_fn = lambda obs, k: (jp.zeros(env.action_size), {})   # plumbing stub
  else:
    getup_fn = jax.jit(build_inference_fn(
        env, hf_hub_download(repo_id=REPO, filename=a.getup)))

  model = everest.build_model(mu=a.mu, mode="hybrid", start_x=a.start_x)
  r = NativePolicyRunner(env, walk_fn, mu=a.mu, kp_scale=0.8,
                         cold_scale=(2.25, 1.6), model=model)

  grid = everest._grid()
  res = grid.shape[0]
  mpp = everest.SIZE_M / res

  def ground_z(x, y):
    cj = int(np.clip(round(x / mpp + (res - 1) / 2.0), 0, res - 1))
    ci = int(np.clip(round(y / mpp + (res - 1) / 2.0), 0, res - 1))
    return float(grid[ci, cj]) * everest.RELIEF_M

  renderer = None
  if not a.no_video:
    renderer = mujoco.Renderer(model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 12.0, -6, 110

  cmd = np.array([0.8, 0.0, 0.0], np.float32)
  key = jax.random.PRNGKey(a.seed)
  mode, held, frames = "WALK", 0, []
  falls = recoveries = 0
  events = []

  for i in range(a.steps):
    up = r.uprightness
    z = float(r.data.qpos[2])
    clear = z - ground_z(float(r.data.qpos[0]), float(r.data.qpos[1]))

    if mode == "WALK" and up < FALL_ENTER:
      mode, held, falls = "GETUP", 0, falls + 1
      events.append((i, "SLIPPED", round(up, 2)))
    elif mode == "GETUP":
      if up > STAND_EXIT and clear > CLEAR_H:
        held += 1
        if held >= HOLD:
          mode, recoveries = "WALK", recoveries + 1
          events.append((i, "RECOVERED", round(clear, 2)))
      else:
        held = 0

    key, k = jax.random.split(key)
    if mode == "WALK":
      act, _ = walk_fn(r.observe(cmd), k)
    else:
      act, _ = getup_fn(r.observe_getup(), k)
    r.step(np.asarray(act))

    if renderer is not None:
      cam.lookat[:] = [float(r.data.qpos[0]) + 2.0,
                       float(r.data.qpos[1]) + 5.0,
                       float(r.data.qpos[2]) + 1.6]
      renderer.update_scene(r.data, camera=cam)
      frames.append(renderer.render())

  if renderer is not None:
    renderer.close()
    imageio.mimsave(a.out, frames, fps=50, macro_block_size=1)
    print("wrote", a.out)

  print(f"mu={a.mu}  {a.steps} steps")
  print(f"slips: {falls}   recoveries: {recoveries}")
  for step, what, val in events[:24]:
    print(f"  step {step:4d}  {what:10s} {val}")


if __name__ == "__main__":
  main()
