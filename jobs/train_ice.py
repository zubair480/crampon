# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "playground",
#   "jax[cuda12]<0.10",
#   "jaxlib<0.10",
#   "brax>=0.14.2",
#   "huggingface_hub",
#   "crampon @ git+https://github.com/zubair480/crampon.git",
# ]
# ///
"""Train G1 locomotion on ice, then sweep friction and push results to the Hub.

One script serves both arms of the comparison, so the baseline and the ice
policy differ only in their training distribution and nothing else:

    MODE=ice   train on mu in [0.02, 0.35] with cold actuators   (ours)
    MODE=dry   train on mu in [0.40, 1.00], no cold              (baseline)

Both are evaluated on the same friction sweep. The difference between the two
curves is the entire result.

Knobs are env vars so the same file runs as a cheap smoke test and as the
full run:

    TIMESTEPS      default 5_000_000
    NUM_ENVS       default 2048
    EPISODE_LENGTH default 1000
    EVAL_ENVS      default 256
    MODE           default "ice"
    REPO_ID        default "Zubair480/crampon-g1-ice"
    SKIP_UPLOAD    set to "1" to skip pushing to the Hub

Launch:
    hf jobs uv run --flavor l4x1 --timeout 30m --namespace iteratehack \
        --secrets HF_TOKEN -e TIMESTEPS=2000000 jobs/train_ice.py
"""

import functools
import json
import os
import time

import jax

# ----------------------------- config -----------------------------
TIMESTEPS = int(os.environ.get("TIMESTEPS", 200_000_000))  # Playground tuned value
NUM_ENVS = int(os.environ.get("NUM_ENVS", 0))  # 0 = use tuned default (8192)
EPISODE_LENGTH = int(os.environ.get("EPISODE_LENGTH", 1000))
EVAL_ENVS = int(os.environ.get("EVAL_ENVS", 256))
MODE = os.environ.get("MODE", "ice").lower()
# Wind is a curriculum knob, not a constant. A policy that cannot yet stand
# will never learn under 348 N gusts, so cap it during training and only
# evaluate at full 200 km/h. WIND is the max gust speed in m/s; 0 disables.
WIND = float(os.environ.get("WIND", "0" if os.environ.get("MODE","ice")=="dry" else "12"))
REPO_ID = os.environ.get("REPO_ID", "Zubair480/crampon-g1-ice")
SKIP_UPLOAD = os.environ.get("SKIP_UPLOAD", "") == "1"
# Warm start: filename of a checkpoint in REPO_ID to resume from. Training ice
# from scratch diverges -- an exploring policy on near-frictionless ground
# generates enormous velocities, enormous observations, and the network goes
# NaN. Starting from a policy that already walks keeps states in range.
RESTORE = os.environ.get("RESTORE", "")
# Friction band for this stage of the curriculum. Anneal across runs:
#   0.20-0.60  ->  0.05-0.35  ->  0.02-0.20
MU_LO = float(os.environ.get("MU_LO", 0)) or None
MU_HI = float(os.environ.get("MU_HI", 0)) or None
# Clip normalized observations. A single wild state on ice can otherwise blow
# up the gradient and poison the weights permanently.
OBS_CLIP = float(os.environ.get("OBS_CLIP", 10.0))
# Each mu forces a recompile of the wrapped env, so a full 7-point sweep costs
# real minutes. Keep it short for smoke runs.
SWEEP_MUS = [float(x) for x in os.environ["SWEEP_MUS"].split(",")]     if os.environ.get("SWEEP_MUS") else None

RUN_NAME = os.environ.get("RUN_NAME") or f"{MODE}-{TIMESTEPS//1000}k"


def main() -> None:
  print("=" * 64, flush=True)
  print(f"crampon train | mode={MODE} timesteps={TIMESTEPS:,} envs={NUM_ENVS}",
        flush=True)
  print("jax backend:", jax.default_backend(), jax.devices(), flush=True)
  assert jax.default_backend() == "gpu", "no GPU -- wrong flavor"

  from brax.io import model
  from brax.training.acme import running_statistics
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.agents.ppo import train as ppo
  from mujoco_playground import wrapper

  from crampon.ice_env import G1Ice
  from crampon.randomize_ice import DRY, MIXED, make_randomizer
  from crampon.eval import sweep

  # The only difference between the two arms.
  if MU_LO is not None and MU_HI is not None:
    band, cold = (MU_LO, MU_HI), MODE != "dry"
  elif MODE == "dry":
    band, cold = DRY, False
  else:
    band, cold = MIXED, True
  print(f"friction band {band}  cold={cold}", flush=True)
  randomize = make_randomizer(band, cold=cold)

  from crampon.ice_env import default_config as ice_default_config
  cfg = ice_default_config()
  cfg.wind_config.speed_range = [0.0, WIND]
  cfg.wind_config.enable = WIND > 0.0
  print(f"wind: max {WIND:.1f} m/s "
        f"({0.5*0.45*1.0*0.5*WIND**2:.0f} N peak drag)", flush=True)

  restore_params = None
  if RESTORE:
    from huggingface_hub import hf_hub_download
    rp = hf_hub_download(repo_id=REPO_ID, filename=RESTORE)
    restore_params = model.load_params(rp)
    print(f"warm starting from {RESTORE}", flush=True)

  env = G1Ice(config=cfg)
  eval_env = G1Ice(config=cfg)
  print(f"env ready | action_size={env.action_size}", flush=True)

  t0 = time.time()
  progress = []

  def progress_fn(step, metrics):
    r = float(metrics.get("eval/episode_reward", float("nan")))
    el = float(metrics.get("eval/avg_episode_length", float("nan")))
    print(f"  step={step:>10,}  reward={r:9.3f}  ep_len={el:7.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    progress.append({"step": int(step), "reward": r, "episode_length": el})

  # Start from Playground's TUNED config for this robot rather than
  # hand-picked numbers. Our first attempt used 15M steps, 2048 envs, and no
  # value_obs_key -- 8% of the required training, a quarter of the envs, and a
  # critic that could not see the privileged observation. It plateaued at 61
  # steps of survival and 0% success at every friction level.
  from mujoco_playground.config import locomotion_params
  ppo_cfg = locomotion_params.brax_ppo_config("G1JoystickFlatTerrain")

  net_cfg = dict(ppo_cfg.network_factory)
  del ppo_cfg["network_factory"]
  train_kwargs = dict(ppo_cfg)
  train_kwargs["num_timesteps"] = TIMESTEPS
  if NUM_ENVS:
    train_kwargs["num_envs"] = NUM_ENVS
  train_kwargs["episode_length"] = EPISODE_LENGTH

  print("ppo config:", json.dumps(
      {k: v for k, v in train_kwargs.items() if isinstance(v, (int, float, str))},
      indent=2), flush=True)
  print("network:", net_cfg, flush=True)

  make_inference_fn, params, _ = ppo.train(
      environment=env,
      eval_env=eval_env,
      network_factory=functools.partial(
          ppo_networks.make_ppo_networks,
          preprocess_observations_fn=functools.partial(
              running_statistics.normalize, max_abs_value=OBS_CLIP
          ),
          **net_cfg
      ),
      restore_params=restore_params,
      randomization_fn=randomize,
      # Playground envs are MjxEnv, not brax envs -- brax's default wrapper
      # reaches for env.sys and dies.
      wrap_env_fn=wrapper.wrap_for_brax_training,
      progress_fn=progress_fn,
      seed=0,
      **train_kwargs,
  )
  train_secs = time.time() - t0
  print(f"training done in {train_secs:.0f}s", flush=True)

  params_path = f"policy-{RUN_NAME}.pkl"
  model.save_params(params_path, params)
  print(f"saved {params_path}", flush=True)

  print("friction sweep:", flush=True)
  inference_fn = make_inference_fn(params, deterministic=True)
  sweep_kwargs = {"num_envs": EVAL_ENVS, "episode_length": 500}
  if SWEEP_MUS is not None:
    sweep_kwargs["mus"] = SWEEP_MUS
  results = sweep(eval_env, inference_fn, **sweep_kwargs)

  report = {
      "mode": MODE,
      "timesteps": TIMESTEPS,
      "num_envs": NUM_ENVS,
      "train_seconds": train_secs,
      "progress": progress,
      "sweep": results,
  }
  report_path = f"report-{RUN_NAME}.json"
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

  if SKIP_UPLOAD:
    print("SKIP_UPLOAD set, not pushing to Hub", flush=True)
  else:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    for path in (params_path, report_path):
      api.upload_file(
          path_or_fileobj=path,
          path_in_repo=path,
          repo_id=REPO_ID,
          repo_type="model",
      )
    print(f"pushed to https://huggingface.co/{REPO_ID}", flush=True)

  print("=" * 64, flush=True)
  print("TRAIN + SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
  main()
