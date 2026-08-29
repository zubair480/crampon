"""Render a trained policy to video, and plot the friction sweep.

Demo is 20 of the 100 judging points, and a 2-minute demo video is a hard
submission requirement. This turns a checkpoint into both deliverables.

The side-by-side is the shot that wins: same friction, same wind, baseline
policy skating and falling next to ours staying upright. Judges do not read
reward curves, they watch the robot.
"""

import json
from typing import Callable, List, Optional, Sequence

import jax
import jax.numpy as jp
import mujoco
import numpy as np


def pin_conditions(env, mu: float, kp_scale: float = 0.8):
  """Return a single (unbatched) mjx model with friction and cold pinned.

  The training randomizer vmaps over a batch of models. For rendering we want
  exactly one deterministic model, so set the same fields directly rather than
  randomizing and then trying to un-batch the result.
  """
  m = env.mjx_model
  return m.tree_replace({
      "pair_friction": m.pair_friction.at[0:2, 0:2].set(mu),
      "actuator_gainprm": m.actuator_gainprm.at[:, 0].set(
          m.actuator_gainprm[:, 0] * kp_scale
      ),
      "actuator_biasprm": m.actuator_biasprm.at[:, 1].set(
          m.actuator_biasprm[:, 1] * kp_scale
      ),
  })


def rollout_qpos(
    env,
    inference_fn: Callable,
    mu: float,
    steps: int = 500,
    seed: int = 0,
    kp_scale: float = 0.8,
) -> List[np.ndarray]:
  """Roll out one unbatched episode at a pinned mu, returning qpos per step."""
  env._mjx_model = pin_conditions(env, mu, kp_scale)

  reset_fn = jax.jit(env.reset)
  step_fn = jax.jit(env.step)
  act_fn = jax.jit(inference_fn)

  key = jax.random.PRNGKey(seed)
  state = reset_fn(key)

  frames = [np.array(state.data.qpos)]
  for _ in range(steps):
    key, act_key = jax.random.split(key)
    action, _ = act_fn(state.obs, act_key)
    state = step_fn(state, action)
    frames.append(np.array(state.data.qpos))
    if float(state.done) > 0:
      break
  return frames


def write_video(
    mj_model: mujoco.MjModel,
    qpos_frames: Sequence[np.ndarray],
    path: str,
    fps: int = 50,
    width: int = 960,
    height: int = 540,
    camera: Optional[str] = None,
) -> str:
  """Render qpos frames with MuJoCo's offscreen renderer and write an mp4."""
  import imageio.v2 as imageio

  data = mujoco.MjData(mj_model)
  renderer = mujoco.Renderer(mj_model, height=height, width=width)

  images = []
  for qpos in qpos_frames:
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)
    if camera is None:
      renderer.update_scene(data)
    else:
      renderer.update_scene(data, camera=camera)
    images.append(renderer.render())

  imageio.mimsave(path, images, fps=fps, macro_block_size=1)
  renderer.close()
  return path


def plot_sweep(reports: dict, path: str = "sweep.png") -> str:
  """The money plot: success rate vs friction, one line per policy.

  `reports` maps a label ("ours (ice)", "baseline (dry)") to the `sweep` list
  from a training report JSON.
  """
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)

  for label, sweep in reports.items():
    mus = [r["mu"] for r in sweep]
    sr = [r["success_rate"] for r in sweep]
    ax.plot(mus, sr, marker="o", linewidth=2, label=label)

  # Mark the band the ice policy was trained on.
  ax.axvspan(0.02, 0.35, alpha=0.10, color="tab:blue")
  ax.text(0.06, 0.04, "ice / packed snow", fontsize=8, color="tab:blue")

  ax.set_xlabel(r"ground friction coefficient  $\mu$")
  ax.set_ylabel("success rate  (episode survived)")
  ax.set_title("G1 locomotion: survival vs ground friction")
  ax.set_ylim(-0.02, 1.02)
  ax.set_xscale("log")
  ax.grid(alpha=0.3)
  ax.legend()
  fig.tight_layout()
  fig.savefig(path)
  return path


def plot_from_files(paths: dict, out: str = "sweep.png") -> str:
  """plot_sweep, reading report-*.json files straight off disk."""
  reports = {}
  for label, p in paths.items():
    with open(p) as f:
      reports[label] = json.load(f)["sweep"]
  return plot_sweep(reports, out)
