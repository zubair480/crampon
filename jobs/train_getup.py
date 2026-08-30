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
"""Train the G1 to stand back up after falling from any orientation.

Playground ships no G1 getup task -- getup exists only for quadrupeds. This
trains the port in crampon/getup_env.py, on the full-body-contact model from
crampon/getup_model.py without which a fallen humanoid sinks through the floor.

    TIMESTEPS   default 80_000_000
    NUM_ENVS    default 8192
    MU          floor friction (0.6 normal, 0.05 ice)
    REPO_ID     where to push the checkpoint
"""

import functools
import json
import os
import time

import jax

TIMESTEPS = int(os.environ.get("TIMESTEPS", 80_000_000))
NUM_ENVS = int(os.environ.get("NUM_ENVS", 8192))
MU = float(os.environ.get("MU", 0.6))
REPO_ID = os.environ.get("REPO_ID", "Zubair480/crampon-g1-ice")
RUN_NAME = os.environ.get("RUN_NAME", f"getup-mu{MU}")


def main() -> None:
  print("=" * 64, flush=True)
  print(f"crampon getup | timesteps={TIMESTEPS:,} envs={NUM_ENVS} mu={MU}",
        flush=True)
  print("jax:", jax.default_backend(), jax.devices(), flush=True)
  assert jax.default_backend() == "gpu", "no GPU -- wrong flavor"

  from brax.io import model
  from brax.training.acme import running_statistics
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.agents.ppo import train as ppo
  from mujoco_playground import wrapper

  from crampon.getup_env import G1Getup

  env = G1Getup(mu=MU)
  eval_env = G1Getup(mu=MU)
  print(f"env ready | act={env.action_size} obs={env.observation_size}",
        flush=True)

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
      num_evals=15,
      episode_length=300,
      num_envs=NUM_ENVS,
      batch_size=256,
      num_minibatches=32,
      num_updates_per_batch=4,
      unroll_length=20,
      learning_rate=3e-4,
      entropy_cost=0.005,
      clipping_epsilon=0.2,
      max_grad_norm=1.0,
      discounting=0.97,
      reward_scaling=1.0,
      normalize_observations=True,
      num_resets_per_eval=1,
      network_factory=functools.partial(
          ppo_networks.make_ppo_networks,
          preprocess_observations_fn=functools.partial(
              running_statistics.normalize, max_abs_value=10.0),
          policy_hidden_layer_sizes=(512, 256, 128),
          value_hidden_layer_sizes=(512, 256, 128),
          policy_obs_key="state",
          value_obs_key="privileged_state",
      ),
      wrap_env_fn=wrapper.wrap_for_brax_training,
      progress_fn=progress_fn,
      seed=0,
  )
  train_secs = time.time() - t0
  print(f"training done in {train_secs:.0f}s", flush=True)

  params_path = f"policy-{RUN_NAME}.pkl"
  model.save_params(params_path, params)

  # Success = ended the episode upright and near standing height.
  import jax.numpy as jp
  import numpy as np
  inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
  reset = jax.jit(eval_env.reset)
  step = jax.jit(eval_env.step)
  ups, heights = [], []
  for s in range(32):
    st = reset(jax.random.PRNGKey(1000 + s))
    key = jax.random.PRNGKey(s)
    for _ in range(300):
      key, k = jax.random.split(key)
      act, _ = inference_fn(st.obs, k)
      st = step(st, act)
    ups.append(float(eval_env.get_gravity(st.data, "torso")[-1]))
    heights.append(float(st.data.qpos[2]))
  ups, heights = np.array(ups), np.array(heights)
  success = float(((ups > 0.8) & (heights > 0.6)).mean())
  print(f"getup success rate: {success:.3f} over 32 random fallen starts",
        flush=True)
  print(f"  mean uprightness {ups.mean():.3f}  mean height {heights.mean():.3f}",
        flush=True)

  report = {"run": RUN_NAME, "mu": MU, "timesteps": TIMESTEPS,
            "train_seconds": train_secs, "progress": progress,
            "success_rate": success,
            "final_uprightness": ups.tolist(),
            "final_height": heights.tolist()}
  report_path = f"report-{RUN_NAME}.json"
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

  from huggingface_hub import HfApi
  api = HfApi(token=os.environ.get("HF_TOKEN"))
  api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
  for p in (params_path, report_path):
    api.upload_file(path_or_fileobj=p, path_in_repo=p, repo_id=REPO_ID,
                    repo_type="model")
  print(f"pushed to https://huggingface.co/{REPO_ID}", flush=True)
  print("GETUP TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
  main()
