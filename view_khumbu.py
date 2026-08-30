"""Interactive viewer: the G1 walking the real Khumbu DEM, live.

Khumbu terrain: 512x512 float32 heightfield at 0.234 m/cell over a 120 m
arena, 36.3 m of relief, with an 8 cm compliant snow layer over a rigid base.
Air density 0.68 kg/m3, roughly half sea level at ~5400 m.

Collision is feet-only, which is what the policy trained against. With every
collision geom enabled the robot's shin capsules catch on the terrain and it
walks 0.81 m in a crouch instead of 7.30 m upright.

    python view_khumbu.py                    # walks on its own
    python view_khumbu.py --mu 0.15 --vx 0.5 # icier, slower
    python view_khumbu.py --vx 0             # stand still, drive with keys

Keys: W/S forward-back, A/D strafe, Q/E turn, X stop, R reset,
      ctrl+drag to shove the robot, space to pause, scroll to zoom.
"""

import argparse
import time

import jax
import mujoco
import mujoco.viewer
import numpy as np
from huggingface_hub import hf_hub_download

import khumbu
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", default="policy-ice-s1.pkl")
  ap.add_argument("--mu", type=float, default=0.5, help="snow/ice friction")
  ap.add_argument("--wind", type=float, default=8.0, help="max gust, m/s")
  ap.add_argument("--vx", type=float, default=0.8, help="0 to stand still")
  ap.add_argument("--fps", type=float, default=60.0,
                  help="render cap; 60 costs nothing here")
  args = ap.parse_args()

  cfg = default_config()
  cfg.noise_config.level = 0.0
  env = G1Ice(config=cfg)

  path = args.policy
  if not path.endswith(".pkl") or "/" not in path and "\\" not in path:
    try:
      path = hf_hub_download(repo_id="Zubair480/crampon-g1-ice",
                             filename=args.policy)
    except Exception:
      path = args.policy

  print("loading policy ...", flush=True)
  fn = jax.jit(build_inference_fn(env, path))

  x, y, z, slope = khumbu.flattest_spawn()
  print(f"spawn ({x:.1f}, {y:.1f})  local slope {slope:.1f} deg", flush=True)

  model = khumbu.build_model(mode="feet", mu=args.mu)
  runner = NativePolicyRunner(env, fn, mu=args.mu, kp_scale=0.8,
                              cold_scale=(2.25, 1.6), model=model)

  cmd = np.array([args.vx, 0.0, 0.0], dtype=np.float32)
  flags = {"reset": False}

  def key_callback(keycode):
    c = chr(keycode) if 0 < keycode < 0x110000 else ""
    s = 0.2
    if c in "Ww": cmd[0] = min(cmd[0] + s, 1.0)
    elif c in "Ss": cmd[0] = max(cmd[0] - s, -1.0)
    elif c in "Aa": cmd[1] = min(cmd[1] + s, 0.5)
    elif c in "Dd": cmd[1] = max(cmd[1] - s, -0.5)
    elif c in "Qq": cmd[2] = min(cmd[2] + s, 1.0)
    elif c in "Ee": cmd[2] = max(cmd[2] - s, -1.0)
    elif c in "Xx": cmd[:] = 0.0
    elif c in "Rr": flags["reset"] = True
    else: return
    print(f"  cmd vx={cmd[0]:+.1f} vy={cmd[1]:+.1f} wz={cmd[2]:+.1f}",
          flush=True)

  key = jax.random.PRNGKey(0)
  rng = np.random.default_rng(0)
  v, th = args.wind * 0.5, 0.0
  dt = env.dt

  # Warm up before the clock starts: the first inference JIT-compiles for
  # about a second, and entering the realtime loop with that pending makes it
  # stutter until it catches up.
  print("compiling ...", flush=True)
  for _ in range(20):
    o = runner.observe(cmd)
    key, k = jax.random.split(key)
    a, _ = fn(o, k)
    runner.step(np.asarray(a))
  runner.reset()

  print(f"mu={args.mu}  wind<={args.wind} m/s   W/S A/D Q/E drive, X stop, "
        f"R reset", flush=True)

  with mujoco.viewer.launch_passive(
      model, runner.data, key_callback=key_callback) as viewer:
    viewer.opt.geomgroup[:] = 0
    for g in (0, 1, 2):
      viewer.opt.geomgroup[g] = 1
    viewer.cam.distance = 6.0
    viewer.cam.elevation = -12
    viewer.cam.azimuth = 130

    sim_t = 0.0
    wall0 = time.perf_counter()
    render_dt = 1.0 / args.fps
    next_render = 0.0
    n_steps = n_frames = 0
    next_report = 3.0

    while viewer.is_running():
      if flags["reset"]:
        runner.reset()
        wall0 = time.perf_counter()
        sim_t = 0.0
        flags["reset"] = False

      elapsed = time.perf_counter() - wall0
      n = 0
      while sim_t < elapsed and n < 6:
        if args.wind > 0:
          v += (args.wind * 0.5 - v) * (dt / 2.0) + 6.0 * np.sqrt(dt) * rng.normal()
          v = float(np.clip(v, 0.0, args.wind))
          th += 0.3 * np.sqrt(dt) * rng.normal()
          mag = 0.5 * 0.68 * 0.5 * v * v   # rho at 5400 m
          force = np.array([mag * np.cos(th), mag * np.sin(th), 0.0])
        else:
          force = None

        obs = runner.observe(cmd)
        key, ak = jax.random.split(key)
        act, _ = fn(obs, ak)
        runner.step(np.asarray(act), wind_force=force)
        if runner.fallen:
          runner.reset()
        sim_t += dt
        n += 1
        n_steps += 1

      if sim_t < elapsed - 0.5:
        sim_t = elapsed

      if elapsed >= next_render:
        viewer.cam.lookat[:] = runner.data.qpos[:3]
        viewer.sync()
        n_frames += 1
        next_render = elapsed + render_dt
      else:
        time.sleep(0.0005)

      if elapsed >= next_report:
        print(f"  physics {n_steps/elapsed:5.1f} Hz | render "
              f"{n_frames/elapsed:5.1f} FPS | realtime "
              f"{(n_steps*dt)/elapsed:.3f}", flush=True)
        next_report = elapsed + 3.0


if __name__ == "__main__":
  main()
