"""Replacement figures for the reframed Section 5.2 and 5.3.

fig_per_template_v2.png
    Adds the strengthened matcher to the per-template panel. The original
    figure contrasts a textbook rule set against CrossExplain, which reads as
    "rules cannot do this". With the strengthened matcher in the same panel it
    reads correctly: the textbook rule set has blind spots, a domain-aware rule
    set does not, and both rule sets and the learned model are saturated on the
    generated split.

fig_qualhold_v2.png
    The synthesis-robustness result that actually has a mechanism. The safe
    qualifier-hold idiom is recognizable at O0, where the recirculation mux is
    an explicit $_MUX_ cell, and disappears at O1 and beyond, where the mux is
    absorbed into an enable flip-flop. A rule set that does not special-case
    the optimized shape scores 1.000 at O0 and 0.000 afterwards.

Writes into journal/figures/ and copies into paper/ for the Overleaf build.

Run:  python journal/scripts/j06_paper_figures.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from strong_matcher import strong_predict, StrongPatternMatcher   # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import pattern_matcher_predict       # noqa: E402

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

C_GNN, C_PM, C_STRONG, INK = "#0e7490", "#b91c1c", "#d97706", "#1f2937"
RATIO = 0.70          # fitted on train in j04

models = J.load_seed_models()

# ------------------------------------------------------------- figure 1
test_id = J.load("test_id")
y = J.y_of(test_id)
tpls = sorted({s["template"] for s in test_id if s["template"]})

pm = np.array([p == 0 for p in pattern_matcher_predict(test_id)])
st = np.array([p == 0 for p in strong_predict(test_id, gray_xor_ratio=RATIO)])
gn = J.seed_probs(models, test_id).mean(0) > 0.5

acc = {"textbook rules": [], "domain-aware rules": [], "CrossExplain": []}
for t in tpls:
    m = np.array([s["template"] == t for s in test_id])
    acc["textbook rules"].append(J.accuracy(pm[m], y[m]))
    acc["domain-aware rules"].append(J.accuracy(st[m], y[m]))
    acc["CrossExplain"].append(J.accuracy(gn[m], y[m]))

x = np.arange(len(tpls))
w = 0.27
fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
ax.bar(x - w, acc["textbook rules"], w, label="textbook rules", color=C_PM)
ax.bar(x, acc["domain-aware rules"], w, label="domain-aware rules", color=C_STRONG)
ax.bar(x + w, acc["CrossExplain"], w, label="CrossExplain", color=C_GNN)
ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", "\n") for t in tpls], fontsize=6.6)
ax.set_ylabel("detection accuracy", fontsize=8.5)
ax.set_ylim(0, 1.06)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=7.5, ncol=3, loc="lower center", frameon=False,
          bbox_to_anchor=(0.5, 1.0))
for xi, v in zip(x - w, acc["textbook rules"]):
    if v < 0.02:
        ax.text(xi, 0.03, "0.000", ha="center", fontsize=6, color=C_PM,
                rotation=90)
fig.tight_layout()
p1 = J.FIGURES / "fig_per_template_v2.png"
fig.savefig(p1, bbox_inches="tight")
print(f"  -> {p1.relative_to(J.ROOT)}")

# ------------------------------------------------------------- figure 2
# the qualified-hold idiom, with and without the optimized-shape rule
mux = [s for s in test_id if s["template"] == "MUX_HOLD"]
y_m = J.y_of(mux)
lv_m = J.levels(mux)


class NoShapeB(StrongPatternMatcher):
    """The same rule set, but recognizing the qualified hold only in its
    unoptimized form (explicit mux). This is what a rule engine written
    against RTL-like netlists would do."""

    def _qualified_hold(self, sample, feats, pred, succ, cap):
        drivers = pred[cap]
        from strong_matcher import _ctype
        if not any(_ctype(feats[d]) == "$_MUX_" for d in drivers):
            return False
        return super()._qualified_hold(sample, feats, pred, succ, cap)


full = StrongPatternMatcher(gray_xor_ratio=RATIO)
partial = NoShapeB(gray_xor_ratio=RATIO)
levels = ["O0", "O1", "O2", "O3"]
a_full, a_part, a_gnn = [], [], []
gn_m = J.seed_probs(models, mux).mean(0) > 0.5
for l in levels:
    m = lv_m == l
    sub = [s for s, k in zip(mux, m) if k]
    a_full.append(J.accuracy(np.array([full.predict(s) == 0 for s in sub]), y_m[m]))
    a_part.append(J.accuracy(np.array([partial.predict(s) == 0 for s in sub]), y_m[m]))
    a_gnn.append(J.accuracy(gn_m[m], y_m[m]))

fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
ax.plot(levels, a_part, "o--", color=C_PM, lw=1.8,
        label="rules, unoptimized shape only")
ax.plot(levels, a_full, "^-", color=C_STRONG, lw=1.8,
        label="rules, both shapes")
ax.plot(levels, a_gnn, "s-", color=C_GNN, lw=1.8, label="CrossExplain")
ax.set_xlabel("synthesis optimization level", fontsize=8.5)
ax.set_ylabel("accuracy on MUX\\_HOLD".replace("\\_", "_"), fontsize=8.5)
ax.set_ylim(-0.05, 1.08)
ax.tick_params(labelsize=8)
ax.grid(alpha=0.25)
ax.legend(fontsize=7, frameon=False, loc="center left")
fig.tight_layout()
p2 = J.FIGURES / "fig_qualhold_v2.png"
fig.savefig(p2, bbox_inches="tight")
print(f"  -> {p2.relative_to(J.ROOT)}")

for p in (p1, p2):
    shutil.copy(p, J.ROOT / "paper" / p.name)
print(f"  copied both into {(J.ROOT / 'paper').relative_to(J.ROOT.parent)}")

print("\n  per-template accuracy used in fig 1:")
for i, t in enumerate(tpls):
    print(f"    {t:18s} textbook={acc['textbook rules'][i]:.3f}  "
          f"domain-aware={acc['domain-aware rules'][i]:.3f}  "
          f"CrossExplain={acc['CrossExplain'][i]:.3f}")
print("\n  MUX_HOLD accuracy by level (fig 2):")
for i, l in enumerate(levels):
    print(f"    {l}: unoptimized-shape-only={a_part[i]:.3f}  "
          f"both-shapes={a_full[i]:.3f}  CrossExplain={a_gnn[i]:.3f}")

J.save("j06_paper_figures", {
    "per_template_test_id": {t: {k: acc[k][i] for k in acc}
                             for i, t in enumerate(tpls)},
    "mux_hold_by_level": {l: {"rules_unoptimized_shape_only": a_part[i],
                              "rules_both_shapes": a_full[i],
                              "crossexplain": a_gnn[i]}
                          for i, l in enumerate(levels)},
})
