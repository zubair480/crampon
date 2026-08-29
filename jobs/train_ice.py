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
TIMESTEPS = int(os.environ.get("TIMESTEPS", 5_000_000))
NUM_ENVS = int(os.environ.get("NUM_ENVS", 2048))
EPISODE_LENGTH = int(os.environ.get("EPISODE_LENGTH", 1000))
EVAL_ENVS = int(os.environ.get("EVAL_ENVS", 256))
MODE = os.environ.get("MODE", "ice").lower()
REPO_ID = os.environ.get("REPO_ID", "Zubair480/crampon-g1-ice")
SKIP_UPLOAD = os.environ.get("SKIP_UPLOAD", "") == "1"
# Each mu forces a recompile of the wrapped env, so a full 7-point sweep costs
# real minutes. Keep it short for smoke runs.
SWEEP_MUS = [float(x) for x in os.environ["SWEEP_MUS"].split(",")]     if os.environ.get("SWEEP_MUS") else None

RUN_NAME = f"{MODE}-{TIMESTEPS//1000}k"


def main() -> None:
  print("=" * 64, flush=True)
  print(f"crampon train | mode={MODE} timesteps={TIMESTEPS:,} envs={NUM_ENVS}",
        flush=True)
  print("jax backend:", jax.default_backend(), jax.devices(), flush=True)
  assert jax.default_backend() == "gpu", "no GPU -- wrong flavor"

  from brax.io import model
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.agents.ppo import train as ppo
  from mujoco_playground import wrapper

  from crampon.ice_env import G1Ice
  from crampon.randomize_ice import DRY, MIXED, make_randomizer
  from crampon.eval import sweep

  # The only difference between the two arms.
  if MODE == "dry":
    randomize = make_randomizer(DRY, cold=False)
  else:
    randomize = make_randomizer(MIXED, cold=True)

  env = G1Ice()
  eval_env = G1Ice()
  print(f"env ready | action_size={env.action_size}", flush=True)

  t0 = time.time()
  progress = []

  def progress_fn(step, metrics):
    r = float(metrics.get("eval/episode_reward", float("nan")))
    el = float(metrics.get("eval/avg_episode_length", float("nan")))
    print(f"  step={step:>10,}  reward={r:9.3f}  ep_len={el:7.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    progress.append({"step": int(step), "reward": r, "episode_length": el})

  make_inference_fn, params, _ = ppo.train(
      environment=env,
      eval_env=eval_env,
      num_timesteps=TIMESTEPS,
      num_evals=10,
      episode_length=EPISODE_LENGTH,
      num_envs=NUM_ENVS,
      batch_size=256,
      num_minibatches=32,
      unroll_length=20,
      num_updates_per_batch=4,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      discounting=0.97,
      reward_scaling=1.0,
      normalize_observations=True,
      network_factory=functools.partial(
          ppo_networks.make_ppo_networks,
          policy_hidden_layer_sizes=(512, 256, 128),
          value_hidden_layer_sizes=(512, 256, 128),
      ),
      randomization_fn=randomize,
      # Playground envs are MjxEnv, not brax envs -- brax's default wrapper
      # reaches for env.sys and dies.
      wrap_env_fn=wrapper.wrap_for_brax_training,
      progress_fn=progress_fn,
      seed=0,
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
