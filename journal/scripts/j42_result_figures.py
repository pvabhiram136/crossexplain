"""j42 - results figures for Sections VII and VIII.

Both sections carried tables and no figure. The saturation result in
particular is a shape rather than a set of numbers -- one axis pinned, the
other spread -- and a table hides that.

  fig_saturation.png   18 ablation configurations, generated vs hand-written
  fig_transfer.png     9 methods, benchmark F1 against real-code FPR

Both are generated from result JSON (j33, j32) so they cannot drift from the
prose. This replaces the stale fig_saturation_gap.png, which predated the
corrected numbers and was never referenced.

Layout notes, because the first version of both figures was unreadable:
  - the saturation legend sat on top of the four extraction-cone rows, so it
    is now placed above the axes and the figure is tall enough for 18 rows;
  - in the transfer plot the released and ensemble CrossExplain checkpoints
    have IDENTICAL coordinates, so their labels printed on top of each other.
    Coincident points are now merged into one annotation, and labels are
    placed left or right depending on which side has room.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

GRP = {"depth": ("encoder depth", "#2166ac"),
       "features": ("node features", "#b2182b"),
       "cone": ("extraction cone", "#1b7837")}

SHORT = {
    "CrossExplain (5-seed ens.)": "CE ensemble",
    "CrossExplain (released)": "CE released",
    "GCN ablation": "GCN",
    "GraphSAGE ablation": "GraphSAGE",
    "single-task ablation": "single-task",
    "Random Forest": "Random Forest",
    "Logistic regression": "Logistic reg.",
    "matcher, textbook": "matcher (textbook)",
    "matcher, domain-aware": "matcher (domain-aware)",
}


def saturation():
    d = json.loads((J.RESULTS / "j33_saturation_table.json").read_text())
    rows = d["rows"]
    # 18 rows need vertical room; the first version was 3.6in and the labels
    # collided with each other as well as with the legend.
    fig, ax = plt.subplots(figsize=(5.6, 4.8))

    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        col = GRP[r["group"]][1]
        ax.plot([r["real"], r["gen"]], [yi, yi], color="0.8", lw=1.1, zorder=1)
        ax.scatter(r["real"], yi, s=32, color=col, zorder=3,
                   edgecolor="white", lw=0.6)
        ax.scatter(r["gen"], yi, s=30, marker="D", color="0.2", zorder=3,
                   edgecolor="white", lw=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([r["tag"].replace("_", " ") for r in rows], fontsize=7.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("$F_1$")
    ax.set_xlim(0.50, 1.04)
    ax.grid(alpha=0.22, ls=":", axis="x")
    ax.axvline(1.0, color="0.55", ls=":", lw=1)

    handles = [Line2D([], [], marker="D", ls="", color="0.2", ms=5,
                      label="generated split")]
    handles += [Line2D([], [], marker="o", ls="", color=c, ms=5,
                       label=f"hand-written: {n}")
                for n, c in GRP.values()]
    # Legend above the axes: inside, it covered the extraction-cone rows.
    ax.legend(handles=handles, fontsize=7, ncol=2, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out = J.FIGURES / "fig_saturation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(J.ROOT)}  ({len(rows)} rows)")
    return d


def transfer():
    d = json.loads((J.RESULTS / "j32_transfer_table.json").read_text())
    rows = d["rows"]

    # Merge points that share coordinates. The released and 5-seed-ensemble
    # checkpoints agree exactly on both axes, so two labels were being drawn
    # on the same spot and neither was readable.
    groups = defaultdict(list)
    for r in rows:
        groups[(round(r["test_id_f1"], 6), round(r["conf_fpr"], 6))].append(r)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for (x, yv), members in groups.items():
        learned = all("matcher" not in m["method"] for m in members)
        col = "#2166ac" if learned else "#b2182b"
        ax.scatter(x, yv, s=62, marker="o" if learned else "s", color=col,
                   alpha=0.9, zorder=3, edgecolor="white", lw=0.9)
        label = " / ".join(SHORT[m["method"]] for m in members)
        # Points on the saturated line have no room to their right, so their
        # labels go left; everything else goes right.
        right = x < 0.995
        ax.annotate(label, (x, yv), fontsize=6.8, color="0.2",
                    xytext=(7 if right else -7, 0),
                    textcoords="offset points",
                    ha="left" if right else "right", va="center")

    ax.set_xlabel("$F_1$ on the generated benchmark")
    ax.set_ylabel("false-positive rate, real CDC code")
    ax.set_xlim(0.80, 1.035)
    ax.set_ylim(-0.08, 1.12)
    ax.axvline(1.0, color="0.55", ls=":", lw=1)
    ax.grid(alpha=0.22, ls=":")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#2166ac", ms=6,
               label="learned"),
        Line2D([], [], marker="s", ls="", color="#b2182b", ms=6,
               label="rule engine")],
        fontsize=7, loc="center left", framealpha=0.95)
    fig.tight_layout()
    out = J.FIGURES / "fig_transfer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    n_merged = sum(len(m) - 1 for m in groups.values())
    print(f"  -> {out.relative_to(J.ROOT)}  "
          f"({len(rows)} methods, {len(groups)} distinct points, "
          f"{n_merged} label collision(s) merged)")
    return d


def main():
    J.banner("j42  results figures for Sections VII and VIII")
    s = saturation()
    t = transfer()

    stale = J.FIGURES / "fig_saturation_gap.png"
    if stale.exists():
        stale.unlink()
        print(f"  removed stale {stale.name}")

    J.save("j42_result_figures", dict(
        saturation=dict(n_configs=s["n_configs"],
                        gen_range=s["gen_range"], real_range=s["real_range"]),
        transfer=dict(n_methods=len(t["rows"]),
                      n_at_ceiling=t["n_at_ceiling"],
                      real_range=t["real_range"])))


if __name__ == "__main__":
    main()
