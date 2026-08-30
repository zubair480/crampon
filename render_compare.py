"""Side-by-side demo: baseline vs ice policy, identical ice, wind and seed.

The seed is chosen as the one whose outcome is closest to BOTH policies'
median over the 24 evaluated seeds (see fair_eval.json), so the clip is
representative rather than flattering. At mu=0.05 that is seed 23: baseline
survives 50 steps against a median of 50, ours 129 against a median of 128.
"""

import argparse

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download

from crampon import scene
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.05)
ap.add_argument("--wind", type=float, default=8.0)
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--steps", type=int, default=260)
ap.add_argument("--out", default="compare.mp4")
a = ap.parse_args()

W, H = 640, 480
COLD, KP = (2.25, 1.6), 0.8

cfg = default_config()
cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)


def label(img, text, colour):
  """Burn a caption into the top-left of a frame, without extra deps."""
  import mujoco  # noqa: F401
  band = img.copy()
  band[:34, :, :] = (band[:34, :, :] * 0.25).astype(np.uint8)
  return band


def roll(fname):
  p = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename=fname)
  fn = jax.jit(build_inference_fn(env, p))
  r = NativePolicyRunner(env, fn, mu=a.mu, kp_scale=KP, cold_scale=COLD)
  scene.apply(r.model, ground=scene.ICE)

  rend = mujoco.Renderer(r.model, height=H, width=W)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.distance, cam.elevation, cam.azimuth = 4.0, -10, 130

  # Identical wind realisation for both policies.
  rng = np.random.default_rng(a.seed)
  v, th = a.wind * 0.5, 0.0
  key = jax.random.PRNGKey(a.seed)
  cmd = np.array([0.8, 0.0, 0.0], np.float32)
  dt = env.dt
  frames, fell_at = [], None

  for i in range(a.steps):
    v += (a.wind * 0.5 - v) * (dt / 2.0) + 6.0 * np.sqrt(dt) * rng.normal()
    v = float(np.clip(v, 0.0, a.wind))
    th += 0.3 * np.sqrt(dt) * rng.normal()
    mag = 0.5 * 0.45 * 0.5 * v * v
    f = np.array([mag * np.cos(th), mag * np.sin(th), 0.0])

    o = r.observe(cmd)
    key, k = jax.random.split(key)
    act, _ = fn(o, k)
    r.step(np.asarray(act), wind_force=f)
    if r.fallen and fell_at is None:
      fell_at = i

    cam.lookat[:] = r.data.qpos[:3]
    rend.update_scene(r.data, camera=cam)
    img = rend.render()
    if fell_at is not None:
      img = (img * 0.55).astype(np.uint8)  # dim once it has fallen
    frames.append(img)
  rend.close()
  return frames, fell_at


fb, fell_b = roll("policy-dry-200000k.pkl")
fo, fell_o = roll("policy-ice-s1.pkl")
print(f"seed {a.seed} at mu={a.mu}: baseline fell at {fell_b}, ours at {fell_o}")

n = min(len(fb), len(fo))
sep = np.full((H, 4, 3), 255, np.uint8)
out = [np.hstack([fb[i], sep, fo[i]]) for i in range(n)]
imageio.mimsave(a.out, out, fps=50, macro_block_size=1)
print(f"wrote {a.out}  ({n} frames, left = baseline, right = ours)")
