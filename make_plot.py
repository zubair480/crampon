"""The result figure: survival vs ground friction, measured like-for-like.

Reads fair_eval.json, which evaluates every policy under identical wind, cold
and seeds -- unlike the per-job training sweeps, which each used their own
job's wind setting and cannot be compared to each other.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, BLUE = "#8892a4", "#1f6feb"

data = json.load(open("fair_eval.json"))
mus = sorted(float(k) for k in data)
get = lambda name, f: np.array([f(data["%g" % m][name]) for m in mus])

b_mean, b_sd = get("baseline", lambda d: d["mean"]), get("baseline", lambda d: d["sd"])
o_mean, o_sd = get("ours", lambda d: d["mean"]), get("ours", lambda d: d["sd"])
n_seeds = len(data["%g" % mus[0]]["ours"]["vals"])
# Standard error, not standard deviation: we are comparing means.
b_err, o_err = b_sd / np.sqrt(n_seeds), o_sd / np.sqrt(n_seeds)

fig, ax = plt.subplots(figsize=(7.6, 4.7), dpi=170)
ax.axvspan(0.008, 0.15, alpha=0.10, color=BLUE, zorder=0)
ax.text(0.0088, 476, "ice", fontsize=9, weight="bold", color=BLUE)

ax.errorbar(mus, b_mean, yerr=b_err, color=GREY, marker="o", ms=5, lw=2,
            ls="--", capsize=3, label="Baseline — trained on normal ground",
            zorder=3)
ax.errorbar(mus, o_mean, yerr=o_err, color=BLUE, marker="s", ms=5, lw=2.3,
            capsize=3, label="Ours — ice curriculum", zorder=4)

i = mus.index(0.05)
ax.annotate("+216%", xy=(mus[i], o_mean[i]), xytext=(0.052, o_mean[i] + 130),
            fontsize=12, weight="bold", color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))

ax.set_xscale("log")
ax.set_xticks(mus)
ax.set_xticklabels([f"{m:g}" for m in mus])
ax.set_xlabel("ground friction coefficient  μ        (lower = more slippery)")
ax.set_ylabel(f"steps survived of 500   (mean ± s.e., {n_seeds} seeds)")
ax.set_title("Unitree G1 on ice: survival vs ground friction", weight="bold")
ax.set_ylim(0, 540)
ax.grid(alpha=0.25, zorder=0)
ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

fig.text(0.5, -0.02,
         "Identical conditions for both policies: 8 m/s gusts, cold-derated "
         "actuators, same seeds.",
         ha="center", fontsize=8, color="#666")
fig.tight_layout()
fig.savefig("sweep.png", bbox_inches="tight")
print("wrote sweep.png")
for m, b, o in zip(mus, b_mean, o_mean):
    wins = sum(1 for x, y in zip(data["%g" % m]["ours"]["vals"],
                                 data["%g" % m]["baseline"]["vals"]) if x > y)
    print(f"  mu={m:<6g} baseline {b:6.1f}  ours {o:6.1f}  "
          f"({(o/b-1)*100:+.0f}%, ours ahead {wins}/{n_seeds})")
