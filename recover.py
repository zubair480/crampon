"""Walk on ridged snow, trip, get back up, keep walking.

Neither policy can do this alone: the walking policy has no idea how to stand
up from its back, and the getup policy has no idea how to walk. A supervisor
switches between them on the torso's up-vector -- which is exactly how the
walking env decides an episode has failed, so the trigger is not invented.

    WALK  --(uprightness < FALL_ENTER)-->  GETUP
    GETUP --(uprightness > STAND_EXIT and height > STAND_H, held)-->  WALK

The two thresholds differ deliberately. A single threshold would flap back and
forth every time the robot wobbled across it; requiring recovery to be held
for a number of steps stops the supervisor handing control back mid-stagger.

The two policies also read different observations -- 103-dim for walking with
velocity command and gait phase, 93-dim for getup without them -- so both are
built from the same MjData each step and only the relevant one is used.
"""

import argparse

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download

from crampon import getup_model, scene
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

REPO = "Zubair480/crampon-g1-ice"

FALL_ENTER = 0.3   # below this uprightness, we have fallen
STAND_EXIT = 0.85  # above this, and tall enough, we are back up
STAND_H = 0.55     # metres
HOLD = 25          # steps recovery must persist before handing back to walking


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--walk", default="policy-ice-s1.pkl")
  ap.add_argument("--getup", default="policy-getup-v2.pkl")
  ap.add_argument("--mu", type=float, default=0.15)
  ap.add_argument("--wind", type=float, default=8.0)
  ap.add_argument("--ridge", type=float, default=0.12)
  ap.add_argument("--steps", type=int, default=1200)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", default="recover.mp4")
  ap.add_argument("--no-video", action="store_true")
  a = ap.parse_args()

  cfg = default_config()
  cfg.noise_config.level = 0.0
  env = G1Ice(config=cfg)

  # Ridged snow, with a body that can actually touch it.
  model = getup_model.build_model(
      a.mu, getup_model.ROUGH_SCENE, ridge_height=a.ridge)

  walk_fn = jax.jit(build_inference_fn(
      env, hf_hub_download(repo_id=REPO, filename=a.walk)))
  if a.getup == "zero":
    # Debug stub: holds the current pose. Lets the supervisor and the terrain
    # be tested before a getup policy exists.
    import jax.numpy as jp
    getup_fn = lambda obs, k: (jp.zeros(env.action_size), {})
  else:
    getup_fn = jax.jit(build_inference_fn(
        env, hf_hub_download(repo_id=REPO, filename=a.getup)))

  r = NativePolicyRunner(env, walk_fn, mu=a.mu, kp_scale=0.8,
                         cold_scale=(2.25, 1.6), model=model)
  scene.apply(r.model, ground=scene.SNOW)

  renderer = None
  if not a.no_video:
    renderer = mujoco.Renderer(r.model, height=480, width=854)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 4.2, -10, 130

  cmd = np.array([0.8, 0.0, 0.0], np.float32)
  key = jax.random.PRNGKey(a.seed)
  rng = np.random.default_rng(a.seed)
  v, th = a.wind * 0.5, 0.0
  dt = env.dt

  mode, recovered_for, frames = "WALK", 0, []
  falls, recoveries, events = 0, 0, []

  for i in range(a.steps):
    v += (a.wind * 0.5 - v) * (dt / 2.0) + 6.0 * np.sqrt(dt) * rng.normal()
    v = float(np.clip(v, 0.0, a.wind))
    th += 0.3 * np.sqrt(dt) * rng.normal()
    mag = 0.5 * 0.45 * 0.5 * v * v
    force = np.array([mag * np.cos(th), mag * np.sin(th), 0.0])

    up, h = r.uprightness, float(r.data.qpos[2])

    if mode == "WALK" and up < FALL_ENTER:
      mode, recovered_for, falls = "GETUP", 0, falls + 1
      events.append((i, "FELL", round(up, 2)))
    elif mode == "GETUP":
      if up > STAND_EXIT and h > STAND_H:
        recovered_for += 1
        if recovered_for >= HOLD:
          mode, recoveries = "WALK", recoveries + 1
          events.append((i, "RECOVERED", round(h, 2)))
      else:
        recovered_for = 0

    key, k = jax.random.split(key)
    if mode == "WALK":
      act, _ = walk_fn(r.observe(cmd), k)
    else:
      act, _ = getup_fn(r.observe_getup(), k)
    r.step(np.asarray(act), wind_force=force)

    if renderer is not None:
      cam.lookat[:] = r.data.qpos[:3]
      renderer.update_scene(r.data, camera=cam)
      img = renderer.render()
      if mode == "GETUP":  # tint the recovery phase so it is unmistakable
        img = img.copy()
        img[..., 0] = np.minimum(255, img[..., 0].astype(int) + 30)
      frames.append(img)

  if renderer is not None:
    renderer.close()
    imageio.mimsave(a.out, frames, fps=50, macro_block_size=1)
    print("wrote", a.out)

  print(f"mu={a.mu} ridge={a.ridge} m wind={a.wind} m/s over {a.steps} steps")
  print(f"falls: {falls}   recoveries: {recoveries}")
  for step, what, val in events[:20]:
    print(f"  step {step:4d}  {what:10s} {val}")


if __name__ == "__main__":
  main()
