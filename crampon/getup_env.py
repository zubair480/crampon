"""G1 stands back up after falling, from any orientation, on ice.

Ported from Playground's Go1Getup (a quadruped) onto the G1 humanoid. The
reward structure is robot-agnostic -- torso upright, torso at standing height,
joints near neutral, stop moving once you are up -- so the port is mostly
constants plus the collision model that makes ground contact exist at all.

Key differences from the quadruped original:
  - 29 actuated joints instead of 12
  - standing torso height ~0.78 m instead of 0.275 m
  - the G1 must be given full-body floor contacts first (getup_model.py);
    stock Playground collides the feet only, so a fallen humanoid sinks
    through the ice with nothing to push against.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts

from crampon import getup_model


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=300,
      action_repeat=1,
      action_scale=0.5,
      drop_from_height_prob=1.0,  # always start fallen; that is the task
      settle_time=0.5,
      soft_joint_pos_limit_factor=0.95,
      restricted_joint_range=False,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=1.0,
              torso_height=1.0,
              posture=1.0,
              stand_still=1.0,
              action_rate=-0.001,
              dof_pos_limits=-0.1,
              torques=-1e-5,
              dof_acc=-2.5e-7,
              dof_vel=-1e-3,
          ),
      ),
      impl="jax",
      naconmax=30 * 8192,
      njmax=400,  # a fallen humanoid makes far more contacts than a walking one
  )


class G1Getup(g1_base.G1Env):
  """Recover from a fall to standing, from any initial orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      mu: float = 0.6,
  ):
    mjx_env.ensure_menagerie_exists()
    super().__init__(
        xml_path=consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    # Swap in the full-body-contact model. Stock G1 collides feet only, so a
    # fallen robot passes through the floor and getup is meaningless.
    self._mj_model = getup_model.build_model(mu)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self._lowers, self._uppers = self._mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    self._z_des = 0.78  # G1 standing torso height
    self._nu = self._mjx_model.nu
    self._imu_site = self._mj_model.site("imu_in_pelvis").id

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    """Drop the robot from 0.6 m at a uniformly random orientation.

    Uniform over the sphere, not a perturbation of upright -- "from any angle"
    means face down, on its back and on either side must all appear.
    """
    rng, ori_rng, joint_rng = jax.random.split(rng, 3)
    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[2].set(0.6)
    quat = jax.random.normal(ori_rng, (4,))
    quat = quat / (jp.linalg.norm(quat) + 1e-6)
    qpos = qpos.at[3:7].set(quat)
    qpos = qpos.at[7:].set(
        jax.random.uniform(joint_rng, (self._nu,),
                           minval=self._lowers, maxval=self._uppers))
    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    qpos = jp.where(
        jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        self._get_random_qpos(key2), self._init_q)
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key3, (6,), minval=-0.5, maxval=0.5))

    data = mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax, njmax=self._config.njmax)
    data = mjx.forward(self.mjx_model, data)
    # Settle so the episode starts from a genuine resting fallen pose, not
    # mid-air.
    data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
    data = data.replace(time=0.0)

    info = {"rng": rng,
            "last_act": jp.zeros(self._nu),
            "last_last_act": jp.zeros(self._nu)}
    metrics = {f"reward/{k}": jp.zeros(())
               for k in self._config.reward_config.scales.keys()}
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Act relative to the CURRENT pose, not the nominal one: from a fallen
    # start the nominal pose is unreachable. Go1Getup makes the same choice.
    motor_targets = state.data.qpos[7:] + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps)

    obs = self._get_obs(data, state.info)
    rewards = self._get_reward(data, action, state.info)
    rewards = {k: v * self._config.reward_config.scales[k]
               for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, -10.0, 10.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    return state.replace(data=data, obs=obs, reward=reward,
                         done=done.astype(reward.dtype))

  def _get_obs(self, data: mjx.Data, info: dict) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data, "pelvis")
    gravity = data.site_xmat[self._imu_site].reshape(3, 3).T @ jp.array(
        [0.0, 0.0, -1.0])
    joint_angles = data.qpos[7:] - self._default_pose
    joint_vel = data.qvel[6:]
    state = jp.hstack([gyro, gravity, joint_angles, joint_vel,
                       info["last_act"]])
    privileged = jp.hstack([state, data.qpos[2:3], data.qvel[0:6]])
    return {"state": state, "privileged_state": privileged}

  # --- rewards -------------------------------------------------------------
  def _get_reward(self, data, action, info) -> Dict[str, jax.Array]:
    # get_gravity returns the torso up-vector; its z is -1 when upright and
    # +1 when upside down, so negate to get "uprightness".
    up = -self.get_gravity(data, "torso")[-1]
    height = data.qpos[2]
    upright = jp.clip(up, 0.0, 1.0)
    at_height = jp.exp(-10.0 * jp.square(height - self._z_des))
    gate = upright * at_height  # posture only matters once actually standing
    return {
        "orientation": upright,
        "torso_height": jp.exp(-5.0 * jp.square(height - self._z_des)),
        "posture": gate * jp.exp(
            -0.5 * jp.sum(jp.square(data.qpos[7:] - self._default_pose))),
        "stand_still": gate * jp.exp(-jp.sum(jp.square(action))),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "dof_pos_limits": jp.sum(
            (data.qpos[7:] < self._soft_lowers)
            | (data.qpos[7:] > self._soft_uppers)),
        "torques": jp.sum(jp.abs(data.actuator_force)),
        "dof_acc": jp.sum(jp.square(data.qacc[6:])),
        "dof_vel": jp.sum(jp.square(data.qvel[6:])),
    }
