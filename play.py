"""Drive a trained G1 policy live in the MuJoCo viewer, with the keyboard.

Why you cannot "just make it walk" in the plain viewer: the G1 has 29 position
servos and no built-in gait. Something has to emit 29 joint targets 50x a
second, coordinated to keep the centre of mass over a moving support polygon.
That controller is the policy -- it is the thing we train, not something
MuJoCo provides.

This loads a trained checkpoint and puts you in the loop: you send velocity
commands, the policy figures out the joint angles.

    python play.py                      # pulls latest from the Hub
    python play.py --policy path.pkl    # or a local checkpoint
    python play.py --mu 0.05 --wind 20  # ice, 20 m/s gusts

Keys:  W/S forward-back   A/D strafe   Q/E turn   X stop   R reset
       space pause (viewer)   ctrl+drag shove the robot
"""

import argparse
import time

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params

from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from crampon import scene

REPO_ID = "Zubair480/crampon-g1-ice"


def build_inference_fn(env, params_path: str):
  """Rebuild the policy network exactly as training built it, then load params."""
  cfg = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")
  net_cfg = dict(cfg.network_factory)

  networks = ppo_networks.make_ppo_networks(
      observation_size=env.observation_size,
      action_size=env.action_size,
      # Training ran with normalize_observations=True, so the params tuple
      # carries a normalizer state as its first element. Rebuild with the same
      # preprocessor or the loaded params will not line up.
      preprocess_observations_fn=running_statistics.normalize,
      **net_cfg,
  )
  params = model.load_params(params_path)
  return ppo_networks.make_inference_fn(networks)(params, deterministic=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", default=None, help="local .pkl checkpoint")
  ap.add_argument("--repo", default=REPO_ID)
  ap.add_argument("--filename", default="policy-dry-200000k.pkl")
  ap.add_argument("--mu", type=float, default=0.6)
  ap.add_argument("--wind", type=float, default=12.0)
  ap.add_argument("--kp-scale", type=float, default=0.8, help="cold derate")
  args = ap.parse_args()

  path = args.policy
  if path is None:
    from huggingface_hub import hf_hub_download
    print(f"downloading {args.filename} from {args.repo} ...")
    path = hf_hub_download(repo_id=args.repo, filename=args.filename)

  cfg = default_config()
  cfg.noise_config.level = 0.0  # real sensors bring their own noise
  env = G1Ice(config=cfg)

  inference_fn = jax.jit(build_inference_fn(env, path))
  runner = NativePolicyRunner(
      env, inference_fn, mu=args.mu, kp_scale=args.kp_scale
  )
  scene.apply(runner.model,
              ground=scene.SNOW if args.mu > 0.3 else scene.ICE)

  cmd = np.zeros(3, dtype=np.float32)
  flags = {"reset": False}

  def key_callback(keycode):
    c = chr(keycode) if 0 < keycode < 0x110000 else ""
    s_ = 0.2
    if c in "Ww": cmd[0] = min(cmd[0] + s_, 1.0)
    elif c in "Ss": cmd[0] = max(cmd[0] - s_, -1.0)
    elif c in "Aa": cmd[1] = min(cmd[1] + s_, 0.5)
    elif c in "Dd": cmd[1] = max(cmd[1] - s_, -0.5)
    elif c in "Qq": cmd[2] = min(cmd[2] + s_, 1.0)
    elif c in "Ee": cmd[2] = max(cmd[2] - s_, -1.0)
    elif c in "Xx": cmd[:] = 0.0
    elif c in "Rr": flags["reset"] = True
    else: return
    print(f"  cmd vx={cmd[0]:+.1f} vy={cmd[1]:+.1f} wz={cmd[2]:+.1f}", flush=True)

  # Wind: Ornstein-Uhlenbeck gusts, same process the env trains with.
  rho, cd, area = 0.45, 1.0, 0.5
  wind_v, wind_th = args.wind * 0.5, 0.0
  rng = np.random.default_rng(0)

  key = jax.random.PRNGKey(0)
  dt = env.dt
  print(f"mu={args.mu}  wind<={args.wind} m/s  kp x{args.kp_scale}")
  print("W/S fwd-back  A/D strafe  Q/E turn  X stop  R reset")

  with mujoco.viewer.launch_passive(
      runner.model, runner.data, key_callback=key_callback
  ) as viewer:
    viewer.opt.geomgroup[:] = 1
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -12

    while viewer.is_running():
      t0 = time.time()

      if flags["reset"]:
        runner.reset()
        flags["reset"] = False

      if args.wind > 0:
        wind_v += (args.wind * 0.5 - wind_v) * (dt / 2.0) + 8.0 * np.sqrt(dt) * rng.normal()
        wind_v = float(np.clip(wind_v, 0.0, args.wind))
        wind_th += 0.3 * np.sqrt(dt) * rng.normal()
        mag = 0.5 * rho * cd * area * wind_v ** 2
        force = np.array([mag * np.cos(wind_th), mag * np.sin(wind_th), 0.0])
      else:
        force = None

      obs = runner.observe(cmd)
      key, act_key = jax.random.split(key)
      action, _ = inference_fn(obs, act_key)
      runner.step(np.asarray(action), wind_force=force)

      if runner.fallen:
        runner.reset()

      viewer.cam.lookat[:] = runner.data.qpos[:3]
      viewer.sync()

      # Native runs ~9x faster than realtime, so pace it to wall clock.
      sleep = dt - (time.time() - t0)
      if sleep > 0:
        time.sleep(sleep)


if __name__ == "__main__":
  main()
