"""Does the DECOUPLED loop actually hold real time? Measures achieved Hz."""
import time
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
scene.apply(runner.model, ground=scene.SNOW); scene.set_realtime_quality(runner.model)

cmd = np.array([1.0, 0.0, 0.0], np.float32); key = jax.random.PRNGKey(0)
dt = env.dt; DURATION = 10.0

# Warm up: first inference triggers a JIT compile worth ~1s. Timing that as if
# it were steady state is how you get a benchmark that lies to you.
for _ in range(20):
    o = runner.observe(cmd); key, k = jax.random.split(key)
    a, _ = fn(o, k); runner.step(np.asarray(a))
runner.reset()

with mujoco.viewer.launch_passive(runner.model, runner.data) as v:
    scene.set_visual_groups(v.opt)
    sim_t, frames, steps, behind = 0.0, 0, 0, 0
    wall0 = time.perf_counter()
    while time.perf_counter() - wall0 < DURATION and v.is_running():
        elapsed = time.perf_counter() - wall0
        n = 0
        while sim_t < elapsed and n < 6:
            obs = runner.observe(cmd)
            key, k = jax.random.split(key)
            act, _ = fn(obs, k)
            runner.step(np.asarray(act))
            if runner.fallen: runner.reset()
            sim_t += dt; n += 1; steps += 1
        if sim_t < elapsed - 0.5:
            behind += 1; sim_t = elapsed
        v.cam.lookat[:] = runner.data.qpos[:3]
        v.sync(); frames += 1
    wall = time.perf_counter() - wall0

print()
print("wall clock      : %.2f s" % wall)
print("simulated time  : %.2f s   (ratio %.3f -- 1.000 is exact realtime)" % (sim_t, sim_t/wall))
print("physics steps   : %d  -> %.1f Hz  (target %.0f Hz)" % (steps, steps/wall, 1/dt))
print("rendered frames : %d  -> %.1f FPS" % (frames, frames/wall))
print("clock resyncs   : %d  (0 means it never fell behind)" % behind)
print("true realtime   : %.3f" % ((steps*dt)/wall))
