"""Side-by-side: baseline vs ice policy, same ice, same wind, same seed."""
import argparse, time
import jax, mujoco, numpy as np, imageio.v2 as imageio
from huggingface_hub import hf_hub_download
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from crampon import scene
from play import build_inference_fn

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=0.05)
ap.add_argument("--wind", type=float, default=12.0)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--out", default="compare.mp4")
a = ap.parse_args()

W, H = 640, 480
cfg = default_config(); cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)

def roll(fname):
    p = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename=fname)
    fn = jax.jit(build_inference_fn(env, p))
    r = NativePolicyRunner(env, fn, mu=a.mu, kp_scale=0.8)
    scene.apply(r.model, ground=scene.ICE)
    rend = mujoco.Renderer(r.model, height=H, width=W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 4.0, -10, 130
    # identical wind realisation for both policies
    rng = np.random.default_rng(0); v, th = a.wind*0.5, 0.0
    key = jax.random.PRNGKey(0); cmd = np.array([0.8,0,0], np.float32)
    dt = env.dt; frames = []; fell_at = None
    for i in range(a.steps):
        v += (a.wind*0.5 - v)*(dt/2.0) + 6.0*np.sqrt(dt)*rng.normal()
        v = float(np.clip(v, 0, a.wind)); th += 0.3*np.sqrt(dt)*rng.normal()
        mag = 0.5*0.45*1.0*0.5*v*v
        f = np.array([mag*np.cos(th), mag*np.sin(th), 0.0])
        obs = r.observe(cmd); key,k = jax.random.split(key)
        act,_ = fn(obs,k); r.step(np.asarray(act), wind_force=f)
        if r.fallen and fell_at is None: fell_at = i
        cam.lookat[:] = r.data.qpos[:3]
        rend.update_scene(r.data, camera=cam); frames.append(rend.render())
        if fell_at is not None and i > fell_at + 40: break
    rend.close()
    return frames, fell_at

t0 = time.time()
fb, fell_b = roll("policy-dry-200000k.pkl")
fo, fell_o = roll("policy-ice-s1.pkl")
print(f"baseline fell at {fell_b}, ours fell at {fell_o}  ({time.time()-t0:.0f}s)")

n = max(len(fb), len(fo))
pad = lambda F: F + [F[-1]]*(n-len(F))
fb, fo = pad(fb), pad(fo)
out = [np.hstack([b, o]) for b, o in zip(fb, fo)]
imageio.mimsave(a.out, out, fps=50, macro_block_size=1)
print("wrote", a.out, f"({n} frames, left=baseline right=ours)")
