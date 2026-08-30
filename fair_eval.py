"""Evaluate every policy under IDENTICAL conditions. No exceptions.

The training-job sweeps cannot be compared: each ran inside its own job with
that job's wind setting, so the baseline was measured at WIND=0 and the ice
policy at WIND=8. Different conditions, presented as one plot. This replaces
it -- same mu, same wind, same cold, same seeds, every policy.
"""
import json, statistics as st, sys
import jax, numpy as np
from huggingface_hub import hf_hub_download
from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

MUS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
SEEDS = list(range(10))
WIND, STEPS = 8.0, 500
COLD = (2.25, 1.6)   # midpoint of the training ranges
KP = 0.8
POLICIES = [("baseline", "policy-dry-200000k.pkl"), ("ours", "policy-ice-s1.pkl")]

cfg = default_config(); cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)

def run(fname, mu, seed):
    p = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename=fname)
    fn = jax.jit(build_inference_fn(env, p))
    r = NativePolicyRunner(env, fn, mu=mu, kp_scale=KP, cold_scale=COLD)
    rng = np.random.default_rng(seed); v, th = WIND*0.5, 0.0
    key = jax.random.PRNGKey(seed); cmd = np.array([0.8,0,0], np.float32); dt = env.dt
    for i in range(STEPS):
        v += (WIND*0.5-v)*(dt/2.0) + 6.0*np.sqrt(dt)*rng.normal()
        v = float(np.clip(v,0,WIND)); th += 0.3*np.sqrt(dt)*rng.normal()
        mag = 0.5*0.45*0.5*v*v
        f = np.array([mag*np.cos(th), mag*np.sin(th), 0.0])
        o = r.observe(cmd); key,k = jax.random.split(key)
        a,_ = fn(o,k); r.step(np.asarray(a), wind_force=f)
        if r.fallen: return i
    return STEPS

out = {}
print(f"identical conditions: wind {WIND} m/s, kp x{KP}, cold {COLD}, {len(SEEDS)} seeds\n")
print("%-6s %12s %12s %9s" % ("mu", "baseline", "ours", "delta"))
for mu in MUS:
    row = {}
    for name, f in POLICIES:
        vals = [run(f, mu, s) for s in SEEDS]
        row[name] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
    b, o = row["baseline"][0], row["ours"][0]
    print("%-6.2f %6.1f +- %-4.1f %6.1f +- %-4.1f %+8.0f%%" %
          (mu, row["baseline"][0], row["baseline"][1], row["ours"][0], row["ours"][1],
           (o/b-1)*100 if b else 0))
    out[mu] = row
json.dump({str(k): v for k, v in out.items()}, open("fair_eval.json","w"), indent=2)
print("\nwrote fair_eval.json")
