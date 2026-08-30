"""The money plot: survival vs ground friction, baseline against ice-trained."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

R = "Zubair480/crampon-g1-ice"
runs = [("Baseline (trained on normal ground)", "report-dry-200000k.json", "#8892a4", "o", "--"),
        ("Ours (ice curriculum)",               "report-ice-s1.json",      "#1f6feb", "s", "-")]

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=170)
data = {}
for label, f, c, mk, ls in runs:
    sweep = json.load(open(hf_hub_download(repo_id=R, filename=f)))["sweep"]
    mus = [r["mu"] for r in sweep]
    steps = [r["mean_steps_alive"] for r in sweep]
    data[label] = (mus, steps)
    ax.plot(mus, steps, marker=mk, color=c, linewidth=2.2, linestyle=ls,
            markersize=6, label=label, zorder=3)

ax.axvspan(0.02, 0.15, alpha=0.13, color="#1f6feb", zorder=0)
ax.text(0.022, 505, "ice", fontsize=9, color="#1f6feb", weight="bold")
ax.axvspan(0.15, 0.4, alpha=0.06, color="#1f6feb", zorder=0)
ax.text(0.17, 505, "packed snow", fontsize=9, color="#5a7fa8")

# Call out the headline number.
b, o = data[runs[0][0]][1][2], data[runs[1][0]][1][2]
ax.annotate(f"+{(o/b-1)*100:.0f}%", xy=(0.10, o), xytext=(0.055, o + 95),
            fontsize=12, weight="bold", color="#1f6feb",
            arrowprops=dict(arrowstyle="->", color="#1f6feb", lw=1.6))

ax.set_xscale("log")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0])
ax.set_xticklabels(["0.02", "0.05", "0.10", "0.20", "0.35", "0.50", "1.0"])
ax.set_xlabel("ground friction coefficient  μ        (lower = more slippery)")
ax.set_ylabel("mean steps survived  (of 500)")
ax.set_title("Unitree G1 locomotion: survival vs ground friction", weight="bold")
ax.set_ylim(0, 560)
ax.grid(alpha=0.25, zorder=0)
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig("sweep.png")
print("wrote sweep.png")
for label, (mus, steps) in data.items():
    print(f"  {label}: {[round(s) for s in steps]}")
