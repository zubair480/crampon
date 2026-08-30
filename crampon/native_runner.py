"""Run a trained policy on NATIVE MuJoCo instead of MJX.

MJX is built to step thousands of robots on a GPU. Stepping exactly one on a
CPU pays all of its batching overhead for none of the benefit -- measured at
664 ms per control step (1.5 Hz) versus 0.5 ms for native mj_step (1928 Hz).
That is 1280x, and it is why the interactive viewer crawls.

This is also the deployment path: a Jetson Thor on the real G1 runs native
MuJoCo or the robot itself, never MJX. So the same code that makes the demo
smooth is the code that would ship.

The only real work is rebuilding the 103-dim observation from a native MjData
exactly as joystick.Joystick._get_obs builds it from mjx.Data:

    linvel(3) gyro(3) gravity(3) command(3)
    joint_angles - default_pose(29)  joint_vel(29)  last_act(29)  phase(4)

Observation noise is deliberately omitted -- it exists to harden training, and
real hardware supplies its own.
"""

import mujoco
import numpy as np


def _sensor_slice(model: mujoco.MjModel, name: str):
  sid = model.sensor(name).id
  adr = model.sensor_adr[sid]
  return slice(adr, adr + model.sensor_dim[sid])


class NativePolicyRunner:
  """Steps native MuJoCo with a trained policy in the loop."""

  def __init__(self, env, inference_fn, mu: float = 0.05, kp_scale: float = 1.0,
               gait_freq: float = 1.4, cold_scale=None):
    self.model = env.mj_model
    self.data = mujoco.MjData(self.model)
    self.inference_fn = inference_fn

    self._default_pose = np.array(env._default_pose)
    self._action_scale = float(env._config.action_scale)
    self._ctrl_dt = float(env.dt)
    self._n_substeps = int(env.n_substeps)

    # Ice + cold, applied to the native model. cold_scale replicates what
    # randomize_ice does during training: thickened grease raises joint dry
    # friction and damping, battery sag lowers servo gain. Omitting these made
    # native evaluation disagree with the MJX sweep.
    self.model.pair_friction[0:2, 0:2] = mu
    if kp_scale != 1.0:
      self.model.actuator_gainprm[:, 0] *= kp_scale
      self.model.actuator_biasprm[:, 1] *= kp_scale
    if cold_scale is not None:
      fl, dmp = cold_scale
      self.model.dof_frictionloss[6:] *= fl
      self.model.dof_damping[6:] *= dmp

    self._gyro = _sensor_slice(self.model, "gyro_pelvis")
    self._linvel = _sensor_slice(self.model, "local_linvel_pelvis")
    # Termination sensors, matching joystick._get_termination exactly.
    self._up_torso = _sensor_slice(self.model, "upvector_torso")
    self._selfcontact = [
        _sensor_slice(self.model, n) for n in (
            "right_foot_left_foot_found",
            "left_foot_right_shin_found",
            "right_foot_left_shin_found",
        )
    ]
    self._imu_site = env._pelvis_imu_site_id
    self._torso_body = env._torso_body_id

    self._phase_dt = 2 * np.pi * self._ctrl_dt * gait_freq
    self._priv_size = int(np.prod(env.observation_size["privileged_state"]))
    self.reset()

  def reset(self) -> None:
    if self.model.nkey > 0:
      mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
    else:
      mujoco.mj_resetData(self.model, self.data)
    self.data.ctrl[:] = self.data.qpos[7:]
    mujoco.mj_forward(self.model, self.data)
    self.last_act = np.zeros(self.model.nu)
    self.phase = np.array([0.0, np.pi])

  def observe(self, command) -> dict:
    d = self.data
    gravity = d.site_xmat[self._imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    state = np.hstack([
        d.sensordata[self._linvel],            # 3
        d.sensordata[self._gyro],              # 3
        gravity,                               # 3
        np.asarray(command, dtype=np.float64), # 3
        d.qpos[7:] - self._default_pose,       # 29
        d.qvel[6:],                            # 29
        self.last_act,                         # 29
        np.concatenate([np.cos(self.phase), np.sin(self.phase)]),  # 4
    ])
    # The policy reads only "state"; privileged_state is fed to the critic
    # during training and is unused at inference. Zeros keep the pytree shape
    # the observation normalizer expects.
    return {
        "state": state.astype(np.float32),
        "privileged_state": np.zeros(self._priv_size, dtype=np.float32),
    }

  def step(self, action: np.ndarray, wind_force=None) -> None:
    """Apply one control step: policy action -> joint targets -> physics."""
    action = np.asarray(action, dtype=np.float64)
    self.data.ctrl[:] = self._default_pose + action * self._action_scale

    self.data.xfrc_applied[:] = 0.0
    if wind_force is not None:
      self.data.xfrc_applied[self._torso_body, :3] = wind_force

    for _ in range(self._n_substeps):
      mujoco.mj_step(self.model, self.data)

    self.last_act = action
    self.phase = np.fmod(self.phase + self._phase_dt + np.pi, 2 * np.pi) - np.pi

  @property
  def fallen(self) -> bool:
    """Exactly joystick._get_termination: torso up-vector flipped, or a
    self-collision between feet and shins, or NaN.

    Note there is NO height threshold. An earlier version of this used
    qpos[2] < 0.4 as a proxy for "fallen", which was invented rather than
    taken from the env -- and it scored the ice policy's low, wide crouch as
    a fall. That single wrong line made a policy that survives 62% LONGER by
    the real metric look like it lost on 12 seeds out of 12.
    """
    d = self.data
    if d.sensordata[self._up_torso][-1] < 0.0:
      return True
    for s in self._selfcontact:
      if d.sensordata[s][0] > 0:
        return True
    return bool(not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all())
