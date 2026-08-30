"""How variable is a single rollout? Scan seeds, no rendering."""
import jax, numpy as np, statistics as st
from huggingface_hub import hf_hub_download
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

MU, WIND, STEPS, SEEDS = 0.05, 12.0, 500, list(range(12))
cfg = default_config(); cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)

def survive(fname, seed):
    p = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename=fname)
    fn = jax.jit(build_inference_fn(env, p))
    r = NativePolicyRunner(env, fn, mu=MU, kp_scale=0.8)
    rng = np.random.default_rng(seed); v, th = WIND*0.5, 0.0
    key = jax.random.PRNGKey(seed); cmd = np.array([0.8,0,0], np.float32)
    dt = env.dt
    for i in range(STEPS):
        v += (WIND*0.5 - v)*(dt/2.0) + 6.0*np.sqrt(dt)*rng.normal()
        v = float(np.clip(v,0,WIND)); th += 0.3*np.sqrt(dt)*rng.normal()
        mag = 0.5*0.45*0.5*v*v
        f = np.array([mag*np.cos(th), mag*np.sin(th), 0.0])
        o = r.observe(cmd); key,k = jax.random.split(key)
        act,_ = fn(o,k); r.step(np.asarray(act), wind_force=f)
        if r.fallen: return i
    return STEPS

res = {}
for name, f in [("baseline","policy-dry-200000k.pkl"), ("ours","policy-ice-s1.pkl")]:
    res[name] = [survive(f, s) for s in SEEDS]
    v = res[name]
    print(f"{name:9s} median {st.median(v):5.0f}  mean {st.mean(v):5.1f}  min {min(v):3d}  max {max(v):3d}")
print()
print("seed:      " + " ".join(f"{s:4d}" for s in SEEDS))
for n in res: print(f"{n:9s}  " + " ".join(f"{x:4d}" for x in res[n]))
wins = sum(1 for a,b in zip(res['ours'],res['baseline']) if a > b)
print(f"\nours survives longer on {wins}/{len(SEEDS)} seeds")
