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
  ap.add_argument("--filename", default="policy-ice-200000k.pkl")
  ap.add_argument("--mu", type=float, default=0.05)
  ap.add_argument("--wind", type=float, default=12.0)
  args = ap.parse_args()

  path = args.policy
  if path is None:
    from huggingface_hub import hf_hub_download
    print(f"downloading {args.filename} from {args.repo} ...")
    path = hf_hub_download(repo_id=args.repo, filename=args.filename)

  cfg = default_config()
  cfg.wind_config.speed_range = [0.0, args.wind]
  cfg.wind_config.enable = args.wind > 0
  env = G1Ice(config=cfg)
  env._mjx_model = env.mjx_model.tree_replace(
      {"pair_friction": env.mjx_model.pair_friction.at[0:2, 0:2].set(args.mu)}
  )

  inference_fn = jax.jit(build_inference_fn(env, path))
  reset_fn = jax.jit(env.reset)
  step_fn = jax.jit(env.step)

  rng = jax.random.PRNGKey(0)
  state = reset_fn(rng)

  cmd = np.zeros(3, dtype=np.float32)  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
  reset_flag = {"do": False}

  def key_callback(keycode):
    c = chr(keycode) if 0 < keycode < 0x110000 else ""
    step = 0.2
    if c in "Ww": cmd[0] = min(cmd[0] + step, 1.0)
    elif c in "Ss": cmd[0] = max(cmd[0] - step, -1.0)
    elif c in "Aa": cmd[1] = min(cmd[1] + step, 0.5)
    elif c in "Dd": cmd[1] = max(cmd[1] - step, -0.5)
    elif c in "Qq": cmd[2] = min(cmd[2] + step, 1.0)
    elif c in "Ee": cmd[2] = max(cmd[2] - step, -1.0)
    elif c in "Xx": cmd[:] = 0.0
    elif c in "Rr": reset_flag["do"] = True
    print(f"  command: vx={cmd[0]:+.1f} vy={cmd[1]:+.1f} wz={cmd[2]:+.1f}", flush=True)

  mj_model = env.mj_model
  mj_data = mujoco.MjData(mj_model)

  print(f"mu={args.mu}  wind<={args.wind} m/s   W/S A/D Q/E to drive, X stop, R reset")

  with mujoco.viewer.launch_passive(
      mj_model, mj_data, key_callback=key_callback
  ) as viewer:
    viewer.opt.geomgroup[:] = 1
    viewer.cam.distance = 3.5
    viewer.cam.elevation = -12

    while viewer.is_running():
      if reset_flag["do"]:
        rng, k = jax.random.split(rng)
        state = reset_fn(k)
        reset_flag["do"] = False

      state.info["command"] = jp.array(cmd)
      rng, act_key = jax.random.split(rng)
      action, _ = inference_fn(state.obs, act_key)
      state = step_fn(state, action)

      mj_data.qpos[:] = np.array(state.data.qpos)
      mj_data.qvel[:] = np.array(state.data.qvel)
      mujoco.mj_forward(mj_model, mj_data)
      viewer.cam.lookat[:] = mj_data.qpos[:3]
      viewer.sync()

      if float(state.done) > 0:
        rng, k = jax.random.split(rng)
        state = reset_fn(k)


if __name__ == "__main__":
  main()
