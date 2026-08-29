# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "playground",
#   "jax[cuda12]<0.10",
#   "jaxlib<0.10",
#   "brax>=0.14.2",
# ]
# ///
"""HF Jobs GPU smoke test for the Himalaya G1 ice stack.

Validates on real CUDA hardware, before we commit budget to a long run:
  1. jax sees the GPU
  2. the jax<0.10 pin still satisfies brax (device_put_replicated exists)
  3. mujoco_playground + the G1 env load
  4. our ice/cold randomizer produces sane batched models
  5. PPO actually steps on the G1 ice env

Run:  hf jobs uv run --flavor l4x1 --timeout 20m hf_smoke.py
"""

import jax
from mujoco import mjx

TORSO_BODY_ID = 16
NU = 29  # G1 actuated DoF

# Friction bands (tangential mu).
ICE = (0.02, 0.15)
MIXED = (0.02, 0.35)  # ice -> packed snow, the training distribution
DRY = (0.4, 1.0)  # stock Playground, kept for baseline runs


def make_randomizer(mu_range=MIXED, cold=True):
  """Build a domain randomizer over a given friction band.

  Pass a degenerate range like (0.05, 0.05) to pin mu exactly -- that is how
  eval sweeps a success-rate-vs-mu curve.
  """
  mu_lo, mu_hi = mu_range

  def domain_randomize(model: mjx.Model, rng: jax.Array):

    @jax.vmap
    def rand_dynamics(rng):
      # --- ICE: floor/foot tangential friction ---
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=mu_lo, maxval=mu_hi)
      pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)

      # --- COLD: grease thickens, so joint dry friction and damping rise ---
      lo, hi = (1.5, 3.0) if cold else (0.5, 2.0)
      rng, key = jax.random.split(rng)
      frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
          key, shape=(NU,), minval=lo, maxval=hi
      )
      dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

      rng, key = jax.random.split(rng)
      dmp_lo, dmp_hi = (1.2, 2.0) if cold else (1.0, 1.0)
      damping = model.dof_damping[6:] * jax.random.uniform(
          key, shape=(NU,), minval=dmp_lo, maxval=dmp_hi
      )
      dof_damping = model.dof_damping.at[6:].set(damping)

      # --- COLD: batteries sag, so the servos push less hard ---
      # NOTE: G1 actuators are UNLIMITED-force position servos
      # (actuator_forcerange is all zeros, actuator_forcelimited is False), so
      # scaling forcerange is a silent no-op. The knob that actually bites is
      # the position gain kp, held in gainprm[:,0] and mirrored as -kp in
      # biasprm[:,1]. Both must be scaled together or the servo goes unstable.
      rng, key = jax.random.split(rng)
      trq_lo, trq_hi = (0.7, 0.9) if cold else (1.0, 1.0)
      kp_scale = jax.random.uniform(key, shape=(NU,), minval=trq_lo, maxval=trq_hi)
      actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
          model.actuator_gainprm[:, 0] * kp_scale
      )
      actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
          model.actuator_biasprm[:, 1] * kp_scale
      )

      # --- stock Playground randomization, unchanged ---
      rng, key = jax.random.split(rng)
      armature = model.dof_armature[6:] * jax.random.uniform(
          key, shape=(NU,), minval=1.0, maxval=1.05
      )
      dof_armature = model.dof_armature.at[6:].set(armature)

      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
      body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
      body_mass = body_mass.at[TORSO_BODY_ID].set(
          body_mass[TORSO_BODY_ID] + dmass
      )

      rng, key = jax.random.split(rng)
      qpos0 = model.qpos0
      qpos0 = qpos0.at[7:].set(
          qpos0[7:]
          + jax.random.uniform(key, shape=(NU,), minval=-0.05, maxval=0.05)
      )

      return (pair_friction, dof_frictionloss, dof_damping,
              actuator_gainprm, actuator_biasprm, dof_armature, body_mass, qpos0)

    (pair_friction, frictionloss, damping, gainprm, biasprm,
     armature, body_mass, qpos0) = rand_dynamics(rng)

    fields = {
        "pair_friction": pair_friction,
        "dof_frictionloss": frictionloss,
        "dof_damping": damping,
        "actuator_gainprm": gainprm,
        "actuator_biasprm": biasprm,
        "dof_armature": armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
    }

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    model = model.tree_replace(fields)
    return model, in_axes

  return domain_randomize


# Ready-made randomizers.
randomize_ice = make_randomizer(ICE)
randomize_mixed = make_randomizer(MIXED)
randomize_baseline = make_randomizer(DRY, cold=False)


# ----------------------------- smoke test -----------------------------
if __name__ == "__main__":
  import time
  import jax.numpy as jp

  print("=" * 60, flush=True)
  print("[1] jax backend:", jax.default_backend(), flush=True)
  print("    devices:", jax.devices(), flush=True)
  assert jax.default_backend() == "gpu", "NO GPU -- wrong flavor or bad jax build"

  print("[2] device_put_replicated present:",
        hasattr(jax, "device_put_replicated"), flush=True)
  assert hasattr(jax, "device_put_replicated"), "jax too new for brax"

  from mujoco_playground import locomotion
  t0 = time.time()
  env = locomotion.load("G1JoystickFlatTerrain")
  print(f"[3] G1 env loaded in {time.time()-t0:.1f}s | act={env.action_size}", flush=True)

  m0 = env.mjx_model
  rng = jax.random.split(jax.random.PRNGKey(0), 4096)
  t0 = time.time()
  m, in_axes = randomize_mixed(m0, rng)
  mu = m.pair_friction[:, 0, 0]
  kp = m.actuator_gainprm[:, 0, 0] / m0.actuator_gainprm[0, 0]
  print(f"[4] randomized 4096 models in {time.time()-t0:.1f}s", flush=True)
  print(f"    mu   min={float(mu.min()):.3f} max={float(mu.max()):.3f}", flush=True)
  print(f"    kp   min={float(kp.min()):.3f} max={float(kp.max()):.3f}", flush=True)
  assert bool(jp.isfinite(m.actuator_gainprm).all()), "NaN in gainprm"
  assert float(mu.min()) >= MIXED[0] - 1e-6 and float(mu.max()) <= MIXED[1] + 1e-6

  print("[5] running short PPO on G1 ice ...", flush=True)
  from brax.training.agents.ppo import train as ppo
  from mujoco_playground import wrapper
  t0 = time.time()
  ppo.train(
      environment=env,
      num_timesteps=200_000,
      num_envs=1024,
      batch_size=256,
      unroll_length=10,
      num_minibatches=8,
      num_updates_per_batch=2,
      num_evals=2,
      episode_length=200,
      learning_rate=3e-4,
      randomization_fn=randomize_mixed,
      # Playground envs are MjxEnv, not brax envs -- brax's default
      # wrap_for_training reaches for env.sys and dies with
      # AttributeError: 'Joystick' object has no attribute 'sys'.
      wrap_env_fn=wrapper.wrap_for_brax_training,
      progress_fn=lambda s, met: print(
          f"    step={s} reward={float(met.get('eval/episode_reward', float('nan'))):.2f}",
          flush=True),
  )
  print(f"[5] PPO ok in {time.time()-t0:.1f}s", flush=True)
  print("=" * 60, flush=True)
  print("SMOKE TEST PASSED -- stack is good, safe to burn budget", flush=True)
