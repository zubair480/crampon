"""G1 joystick locomotion on ice, in wind.

Subclasses Playground's G1 Joystick env and adds two things it lacks:

  1. Sustained wind. The stock env has `push_config`, but a push is an
     instantaneous kick to qvel[:2] -- an impulse. Wind is a *sustained force*,
     and a policy that survives impulses is not the same as one that leans into
     a 40 m/s headwind for a hundred steps. Wind is applied through
     `xfrc_applied` on the torso body.

  2. A bigger constraint buffer. Stock njmax is 29*2 + 8*4 = 90. On ice the
     solver needs more (an L4 run warned "increase njmax to 94"), and dropped
     constraints corrupt contact physics silently.

Wind force comes from the drag equation rather than a made-up number:

    F = 0.5 * rho * Cd * A * v^2

with rho = 0.45 kg/m^3 (~8000 m, roughly a third of sea level -- thin air is
why a Himalayan gale pushes less hard than the same speed at sea level) and
A = 0.5 m^2 for a humanoid torso. At 55.6 m/s (200 km/h) that is 348 N against
a G1 whose total model mass is 33.3 kg (327 N of weight) -- a sideways push
1.06x the robot's own weight. The wind shoves harder than gravity holds.

Gust speed follows an Ornstein-Uhlenbeck process so gusts are temporally
correlated the way real wind is; direction does a slow random walk.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick


def default_config() -> config_dict.ConfigDict:
  cfg = g1_joystick.default_config()

  # Ice generates more constraints than a dry floor. Stock is 90.
  cfg.njmax = 220
  cfg.naconmax = 12 * 8192

  cfg.wind_config = config_dict.create(
      enable=True,
      speed_range=[0.0, 55.6],  # m/s; 55.6 == 200 km/h
      air_density=0.45,  # kg/m^3 at ~8000 m (sea level is 1.225)
      drag_coeff=1.0,  # bluff body
      frontal_area=0.5,  # m^2, humanoid torso
      gust_tau=2.0,  # s, OU relaxation time
      gust_sigma=8.0,  # m/s per sqrt(s)
      dir_sigma=0.3,  # rad per sqrt(s)
  )
  return cfg


class G1Ice(g1_joystick.Joystick):
  """G1 joystick tracking, on ice, in wind."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        task=task, config=config, config_overrides=config_overrides
    )

  def _wind_force(self, v: jax.Array, theta: jax.Array) -> jax.Array:
    wc = self._config.wind_config
    mag = 0.5 * wc.air_density * wc.drag_coeff * wc.frontal_area * v * v
    mag = mag * float(wc.enable)
    return jp.array([mag * jp.cos(theta), mag * jp.sin(theta), 0.0])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    wc = self._config.wind_config

    rng, k_speed, k_dir = jax.random.split(state.info["rng"], 3)
    v = jax.random.uniform(
        k_speed, (), minval=wc.speed_range[0], maxval=wc.speed_range[1]
    )
    theta = jax.random.uniform(k_dir, (), maxval=2 * jp.pi)

    state.info["rng"] = rng
    state.info["wind_speed"] = v
    state.info["wind_dir"] = theta
    state.info["wind_force"] = self._wind_force(v, theta)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    wc = self._config.wind_config
    rng, k_speed, k_dir = jax.random.split(state.info["rng"], 3)

    # Ornstein-Uhlenbeck gust: mean-reverting, so gusts persist over ~gust_tau
    # seconds instead of being white noise the policy can average away.
    dt = self.dt
    v_mean = 0.5 * (wc.speed_range[0] + wc.speed_range[1])
    v = state.info["wind_speed"]
    v = v + (v_mean - v) * (dt / wc.gust_tau)
    v = v + wc.gust_sigma * jp.sqrt(dt) * jax.random.normal(k_speed)
    v = jp.clip(v, wc.speed_range[0], wc.speed_range[1])

    theta = state.info["wind_dir"]
    theta = theta + wc.dir_sigma * jp.sqrt(dt) * jax.random.normal(k_dir)

    force = self._wind_force(v, theta)
    xfrc = state.data.xfrc_applied.at[self._torso_body_id, :3].set(force)
    state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

    state.info["rng"] = rng
    state.info["wind_speed"] = v
    state.info["wind_dir"] = theta
    state.info["wind_force"] = force

    return super().step(state, action)
