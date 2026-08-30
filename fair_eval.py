"""Evaluate every policy under IDENTICAL conditions. No exceptions.

The training-job sweeps cannot be compared against each other: each ran inside
its own job with that job's wind setting, so the baseline was measured at
WIND=0 and the ice policy at WIND=8. Different conditions, one plot. This
replaces that -- same mu, same wind, same cold, same seeds, every policy.

Weighted toward low friction because that is where the policies actually
differ; above mu=0.05 both survive every episode and there is nothing to claim.
"""

import json
import statistics as st

import jax
import numpy as np
from huggingface_hub import hf_hub_download

from crampon.ice_env import G1Ice, default_config
from crampon.native_runner import NativePolicyRunner
from play import build_inference_fn

MUS = [0.010, 0.015, 0.020, 0.030, 0.050, 0.100, 0.350]
SEEDS = list(range(24))
WIND, STEPS = 8.0, 500
COLD = (2.25, 1.6)  # midpoint of the training ranges
KP = 0.8
POLICIES = [("baseline", "policy-dry-200000k.pkl"),
            ("ours", "policy-ice-s1.pkl")]

cfg = default_config()
cfg.noise_config.level = 0.0
env = G1Ice(config=cfg)


def rollout(fn, mu, seed):
  r = NativePolicyRunner(env, fn, mu=mu, kp_scale=KP, cold_scale=COLD)
  rng = np.random.default_rng(seed)
  v, th = WIND * 0.5, 0.0
  key = jax.random.PRNGKey(seed)
  cmd = np.array([0.8, 0.0, 0.0], np.float32)
  dt = env.dt
  for i in range(STEPS):
    v += (WIND * 0.5 - v) * (dt / 2.0) + 6.0 * np.sqrt(dt) * rng.normal()
    v = float(np.clip(v, 0.0, WIND))
    th += 0.3 * np.sqrt(dt) * rng.normal()
    mag = 0.5 * 0.45 * 0.5 * v * v
    f = np.array([mag * np.cos(th), mag * np.sin(th), 0.0])
    o = r.observe(cmd)
    key, k = jax.random.split(key)
    a, _ = fn(o, k)
    r.step(np.asarray(a), wind_force=f)
    if r.fallen:
      return i
  return STEPS


def main():
  fns = {}
  for name, f in POLICIES:
    p = hf_hub_download(repo_id="Zubair480/crampon-g1-ice", filename=f)
    fns[name] = jax.jit(build_inference_fn(env, p))

  print("identical conditions | wind %.0f m/s | kp x%.1f | cold %s | %d seeds"
        % (WIND, KP, COLD, len(SEEDS)))
  print()
  print("%-7s %17s %17s %9s" % ("mu", "baseline", "ours", "delta"))
  out = {}
  for mu in MUS:
    row = {}
    for name, _ in POLICIES:
      vals = [rollout(fns[name], mu, s) for s in SEEDS]
      row[name] = {"mean": st.mean(vals),
                   "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
                   "vals": vals}
    b, o = row["baseline"]["mean"], row["ours"]["mean"]
    wins = sum(1 for x, y in zip(row["ours"]["vals"], row["baseline"]["vals"])
               if x > y)
    print("%-7.3f %9.1f +- %-5.1f %9.1f +- %-5.1f %+8.0f%%   ours ahead %d/%d"
          % (mu, row["baseline"]["mean"], row["baseline"]["sd"],
             row["ours"]["mean"], row["ours"]["sd"],
             (o / b - 1) * 100 if b else 0.0, wins, len(SEEDS)))
    out["%g" % mu] = row

  with open("fair_eval.json", "w") as f:
    json.dump(out, f, indent=2)
  print()
  print("wrote fair_eval.json")


if __name__ == "__main__":
  main()
