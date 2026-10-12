"""Figures for the restructured, diagnostic-led paper.

fig_saturation_gap.png
    The centrepiece. Every method on two axes: F1 on the generated benchmark
    against false-positive rate on correct third-party CDC code. The benchmark
    axis collapses to a point at 1.000 while the real axis spans 0.68 to 1.00,
    which is the paper's argument in one panel.

fig_feature_shift.png
    Why it happens. Node-feature means for generated safe, generated unsafe,
    and found safe crossings. Real CDC sits outside both generated classes on
    the cell types it is built from.

Run:  python journal/scripts/j20_paper_figs.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

D = json.loads((J.JROOT / "results" / "j19_diagnostic.json").read_text())
C_LEARN, C_RULE, INK = "#b91c1c", "#0e7490", "#1f2937"

# ------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(6.0, 3.7), dpi=200)
for name, v in D["transfer"].items():
    if name == "always-safe":
        continue
    learned = "matcher" not in name
    ax.scatter(v["test_id_f1"], v["conf_fpr"], s=64,
               color=C_LEARN if learned else C_RULE,
               marker="o" if learned else "^", zorder=3,
               edgecolor="white", linewidth=0.8)
    dx, ha = -0.006, "right"
    if name == "LogisticReg":
        dx, ha = 0.006, "left"
    ax.annotate(name, (v["test_id_f1"], v["conf_fpr"]),
                xytext=(dx, 0.022), textcoords="offset fontsize",
                fontsize=7.2, ha=ha, color=INK)

ax.axhspan(0.85, 1.02, color="#b91c1c", alpha=0.06, zorder=0)
ax.text(0.845, 0.955, "flags almost every correct\nsynchronizer",
        fontsize=7, color=C_LEARN, va="center")
ax.set_xlabel("$F_1$ on the generated benchmark (test\\_id)", fontsize=9)
ax.set_ylabel("false positives on correct\nthird-party CDC code", fontsize=9)
ax.set_xlim(0.83, 1.03)
ax.set_ylim(0.60, 1.04)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.25)
h = [plt.Line2D([], [], marker="o", ls="", color=C_LEARN, label="learned"),
     plt.Line2D([], [], marker="^", ls="", color=C_RULE, label="rule engine")]
ax.legend(handles=h, fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout()
p1 = J.FIGURES / "fig_saturation_gap.png"
fig.savefig(p1, bbox_inches="tight")
print(f"  -> {p1.relative_to(J.ROOT)}")

# ------------------------------------------------------------- figure 2
sh = D["feature_shifts"][:6]
labels = [s["feature"].replace("$_", "").replace("_", "") for s in sh]
x = np.arange(len(sh))
w = 0.27
fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=200)
ax.bar(x - w, [s["gen_safe"] for s in sh], w, label="generated safe",
       color="#0e7490")
ax.bar(x, [s["gen_unsafe"] for s in sh], w, label="generated unsafe",
       color="#d97706")
ax.bar(x + w, [s["found_safe"] for s in sh], w, label="found safe (real)",
       color="#b91c1c")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("mean fraction of nodes", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.16))
fig.tight_layout()
p2 = J.FIGURES / "fig_feature_shift.png"
fig.savefig(p2, bbox_inches="tight")
print(f"  -> {p2.relative_to(J.ROOT)}")

for p in (p1, p2):
    shutil.copy(p, J.ROOT / "results" / "figures" / p.name)
print("  copied into results/figures/")
