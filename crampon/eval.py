"""Evaluate a policy across a friction sweep.

This is the measurement that decides whether any of the training mattered:
success rate as a function of ground friction. Train on the wide MIXED band,
then evaluate at pinned mu values -- the RL equivalent of a held-out test set.
Include values outside the training band to probe generalization.

"Success" here is surviving the episode without termination. On ice the stock
policy skates, trips, and terminates early; a policy that has learned to keep
its ground reaction forces inside a narrow friction cone stays up.
"""

import functools
from typing import Callable, Dict, List, Sequence

import jax
import jax.numpy as jp

from mujoco_playground import wrapper

from crampon.randomize_ice import make_randomizer

# Values below 0.35 are inside the MIXED training band; 0.5 and 1.0 are
# outside it, and test whether the policy generalizes back to normal ground.
DEFAULT_MUS: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00)


def evaluate_at_mu(
    env,
    inference_fn: Callable,
    mu: float,
    num_envs: int = 256,
    episode_length: int = 500,
    seed: int = 0,
    cold: bool = True,
) -> Dict[str, float]:
  """Roll out `num_envs` episodes at a single pinned friction value."""
  randomize = make_randomizer((mu, mu), cold=cold)

  key = jax.random.PRNGKey(seed)
  key, rand_key, reset_key = jax.random.split(key, 3)

  # brax binds rng into the randomizer via partial; we do the same by hand.
  randomization_fn = functools.partial(
      randomize, rng=jax.random.split(rand_key, num_envs)
  )
  wrapped = wrapper.wrap_for_brax_training(
      env, episode_length=episode_length, randomization_fn=randomization_fn
  )

  reset_fn = jax.jit(wrapped.reset)
  step_fn = jax.jit(wrapped.step)
  act_fn = jax.jit(inference_fn)

  state = reset_fn(jax.random.split(reset_key, num_envs))

  fell = jp.zeros(num_envs, dtype=bool)
  steps_alive = jp.zeros(num_envs)
  total_reward = jp.zeros(num_envs)

  for _ in range(episode_length):
    key, act_key = jax.random.split(key)
    action, _ = act_fn(state.obs, act_key)
    state = step_fn(state, action)
    alive = ~fell
    # Count reward and steps only up to the moment an episode terminates --
    # the AutoReset wrapper restarts it, and post-fall reward is meaningless.
    steps_alive = steps_alive + alive
    total_reward = total_reward + state.reward * alive
    fell = fell | (state.done > 0)

  survived = ~fell
  return {
      "mu": float(mu),
      "success_rate": float(survived.mean()),
      "mean_steps_alive": float(steps_alive.mean()),
      "mean_reward": float(total_reward.mean()),
      "num_envs": int(num_envs),
      "episode_length": int(episode_length),
  }


def sweep(
    env,
    inference_fn: Callable,
    mus: Sequence[float] = DEFAULT_MUS,
    **kwargs,
) -> List[Dict[str, float]]:
  """Full friction sweep. Returns one record per mu, ready to plot or dump."""
  results = []
  for mu in mus:
    r = evaluate_at_mu(env, inference_fn, mu, **kwargs)
    print(
        f"  mu={r['mu']:.2f}  success={r['success_rate']:.3f}  "
        f"steps={r['mean_steps_alive']:.1f}  reward={r['mean_reward']:.2f}",
        flush=True,
    )
    results.append(r)
  return results
