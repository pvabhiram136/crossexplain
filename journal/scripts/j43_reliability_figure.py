"""j43 - reliability diagram for Section VI-C.

Section VI-C reports that confidence is well calibrated in distribution
(ECE 0.013 to 0.022) and not on real designs (0.267), and Section X leans on
that result when it argues that confidence cannot tell you which draw you have.
Neither section had a figure, and calibration is a shape a table cannot show.
The real-design curve does not sit tidily below the diagonal: it wanders
non-monotonically (0.00 accuracy at confidence 0.52, then 0.80, then 0.29),
which is a sharper illustration of the 0.483 self-correctness AUC than any
single ECE number.

`figures/fig_reliability.png` already existed, produced by j02, but nothing
tied it to the current result file, so this rebuilds it from
`j02_calibration.json` under the project's rule that every figure in the paper
comes from result JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

SHOW = [("test_id", "generated (in distribution)", "#2166ac", "o"),
        ("test_real", "hand-written designs", "#b2182b", "s")]


def main():
    J.banner("j43  reliability diagram for Section VI-C")
    cal = json.loads((J.RESULTS / "j02_calibration.json").read_text())
    err = json.loads(
        (J.RESULTS / "j02_calibration.json").read_text())["error_detection"]

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.plot([0.5, 1], [0.5, 1], ls="--", lw=1, color="0.55",
            label="perfect calibration", zorder=1)

    out = {}
    for key, label, col, mk in SHOW:
        c = cal["calibration"][key]
        bins = [b for b in c["reliability"] if b["count"] > 0]
        x = [b["mean_conf"] for b in bins]
        y = [b["accuracy"] for b in bins]
        # Marker area scales with how many samples fall in the bin, so the
        # reader can see which points carry weight.
        n = np.array([b["count"] for b in bins], dtype=float)
        ax.plot(x, y, ls="-", lw=1.2, color=col, alpha=0.7, zorder=2)
        ax.scatter(x, y, s=18 + 90 * n / n.max(), color=col, marker=mk,
                   alpha=0.85, zorder=3, edgecolor="white", lw=0.7,
                   label=f"{label} (ECE {c['ece']:.3f})")
        out[key] = dict(ece=c["ece"], n=c["n"], n_bins=len(bins))
        print(f"  {key:10s} ECE {c['ece']:.4f}  n={c['n']}  "
              f"{len(bins)} occupied bins")

    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.22, ls=":")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    dst = J.FIGURES / "fig_reliability.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {dst.relative_to(J.ROOT)}")
    print(f"  self-correctness AUC on real designs: {err['auc']:.4f}")

    J.save("j43_reliability", dict(splits=out, auc=err["auc"]))


if __name__ == "__main__":
    main()
