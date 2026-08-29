"""Measure where wall time goes in the live loop, including viewer.sync()."""
import time, statistics as st
import jax, numpy as np, mujoco, mujoco.viewer
from huggingface_hub import hf_hub_download
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from crampon import scene
from play import build_inference_fn

p = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename="policy-dry-200000k.pkl")
cfg = default_config(); cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)
fn = jax.jit(build_inference_fn(env, p))
runner = NativePolicyRunner(env, fn, mu=0.6, kp_scale=0.8)
scene.apply(runner.model, ground=scene.SNOW)
scene.set_realtime_quality(runner.model)

cmd = np.array([1.0, 0.0, 0.0], np.float32)
key = jax.random.PRNGKey(0)
dt = env.dt
t_obs, t_inf, t_step, t_sync, t_loop = [], [], [], [], []

with mujoco.viewer.launch_passive(runner.model, runner.data) as v:
    scene.set_visual_groups(v.opt)
    for i in range(400):
        top = time.perf_counter()
        a = time.perf_counter(); obs = runner.observe(cmd); b = time.perf_counter()
        key, k = jax.random.split(key)
        act, _ = fn(obs, k)
        act = np.asarray(act)          # forces JAX to finish
        c = time.perf_counter()
        runner.step(act); d = time.perf_counter()
        v.cam.lookat[:] = runner.data.qpos[:3]
        v.sync(); e = time.perf_counter()
        if runner.fallen: runner.reset()
        if i > 50:
            t_obs.append((b-a)*1000); t_inf.append((c-b)*1000)
            t_step.append((d-c)*1000); t_sync.append((e-d)*1000)
            t_loop.append((e-top)*1000)
        sl = dt - (time.perf_counter() - top)
        if sl > 0: time.sleep(sl)

def r(n, x):
    print("%-10s mean %6.2f ms  p50 %6.2f  p95 %6.2f  max %7.2f" %
          (n, st.mean(x), st.median(x), sorted(x)[int(len(x)*.95)], max(x)))
print()
r("observe", t_obs); r("inference", t_inf); r("mj_step", t_step); r("viewer.sync", t_sync)
r("WORK TOTAL", t_loop)
print("\nbudget per frame at 50 Hz: 20.00 ms")
over = sum(1 for x in t_loop if x > 20)
print("frames where work alone exceeded budget: %d / %d (%.1f%%)" % (over, len(t_loop), 100*over/len(t_loop)))
