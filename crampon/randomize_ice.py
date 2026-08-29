"""Ice / wind / cold domain randomization for the Unitree G1.

Drop-in replacement for mujoco_playground's g1.randomize.domain_randomize.
Stock Playground samples floor friction from U(0.4, 1.0) -- that is a dry gym
floor. Himalaya ice is mu ~ 0.02-0.15, packed snow ~ 0.2-0.35, so we sample the
whole slick band and additionally model what -20C does to the actuators.

Wind is NOT here: it is a force on mjx.Data (xfrc_applied), not a model field,
so it has to be applied in the env's step(). See wind.py.
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
