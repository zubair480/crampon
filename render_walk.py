"""Render the trained policy walking, in the Himalaya scene."""

import argparse, time
import jax, mujoco, numpy as np
import imageio.v2 as imageio
from huggingface_hub import hf_hub_download

from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from crampon import scene
from play import build_inference_fn

ap = argparse.ArgumentParser()
ap.add_argument("--filename", default="policy-dry-200000k.pkl")
ap.add_argument("--mu", type=float, default=0.6)
ap.add_argument("--wind", type=float, default=8.0)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--vx", type=float, default=1.0)
ap.add_argument("--out", default="walk.mp4")
a = ap.parse_args()

path = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename=a.filename)
cfg = default_config(); cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)
fn = jax.jit(build_inference_fn(env, path))
runner = NativePolicyRunner(env, fn, mu=a.mu, kp_scale=0.8)
scene.apply(runner.model, ground=scene.SNOW if a.mu > 0.3 else scene.ICE)

cmd = np.array([a.vx, 0.0, 0.0], dtype=np.float32)
key = jax.random.PRNGKey(0)
rho, cd, area = 0.45, 1.0, 0.5
v, th = a.wind * 0.5, 0.0
rng = np.random.default_rng(0)
dt = env.dt

renderer = mujoco.Renderer(runner.model, height=480, width=854)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance, cam.elevation, cam.azimuth = 4.0, -10, 130

frames, dist0 = [], runner.data.qpos[0]
t0 = time.time()
for i in range(a.steps):
    v += (a.wind * 0.5 - v) * (dt / 2.0) + 6.0 * np.sqrt(dt) * rng.normal()
    v = float(np.clip(v, 0.0, a.wind)); th += 0.3 * np.sqrt(dt) * rng.normal()
    mag = 0.5 * rho * cd * area * v * v
    force = np.array([mag * np.cos(th), mag * np.sin(th), 0.0])

    obs = runner.observe(cmd)
    key, k = jax.random.split(key)
    act, _ = fn(obs, k)
    runner.step(np.asarray(act), wind_force=force)
    if runner.fallen:
        print(f"  fell at step {i}"); break

    cam.lookat[:] = runner.data.qpos[:3]
    renderer.update_scene(runner.data, camera=cam)
    frames.append(renderer.render())
renderer.close()

dist = float(runner.data.qpos[0] - dist0)
print(f"{len(frames)} frames | walked {dist:.2f} m | {time.time()-t0:.0f}s | mu={a.mu}")
imageio.mimsave(a.out, frames, fps=50, macro_block_size=1)
print("wrote", a.out)
