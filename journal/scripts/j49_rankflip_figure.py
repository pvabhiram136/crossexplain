"""j49 - the ordering collapse, as a fan-out chart (Section VIII).

fig_transfer.png already shows *that* the benchmark axis is pinned. It does not
show what that costs you, because a scatter invites the reader to look for a
trend line and there is no trend to find. The claim in Section VIII is
specifically about ORDERING: six of the nine methods are tied to four decimal
places on the benchmark, and those same six span nearly the whole range of
real-code false-positive rate.

This figure draws that as a fan. On the left, methods sit at their benchmark
score, so the six tied methods collapse onto one point -- which is the honest
picture of what the benchmark knows. On the right they sit at their real-code
false-positive rate, and the six lines leaving the tie point arrive spread
across the axis.

Legibility notes, since an earlier figure in this paper had to be redrawn:
  - the six tied methods share one left-hand coordinate exactly, so they are
    drawn as a single bracketed cluster rather than six coincident dots with
    six overlapping labels;
  - every label is on the right, where the points are distinct, and the right
    column is sorted so labels never collide;
  - lines are coloured by direction of travel, which needs no legend entry
    because the two ends are already labelled with their values.
"""
from __future__ import annotations

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


def main() -> int:
    J.banner("j49  ordering-collapse fan chart")

    d = json.loads((J.RESULTS / "j32_transfer_table.json").read_text())
    rows = d["rows"]
    rho, p = d["spearman_rho"], d["spearman_p"]

    gen = np.array([r["test_id_f1"] for r in rows])
    real = np.array([r["conf_fpr"] for r in rows])
    name = [SHORT[r["method"]] for r in rows]

    ceiling = 1.0
    tied = [i for i in range(len(rows)) if abs(gen[i] - ceiling) < 5e-5]
    print(f"  {len(tied)} of {len(rows)} methods tied at F1 = {ceiling:.4f}")
    print(f"  their real-code FPR spans {real[tied].min():.3f}"
          f" to {real[tied].max():.3f}")
    print(f"  Spearman rho = {rho:.3f} (p = {p:.3f})")

    fig, ax = plt.subplots(figsize=(6.4, 4.3))

    # right-hand positions: sorted by FPR so labels cannot collide
    order = np.argsort(real)
    ypos = {int(i): k for k, i in enumerate(order)}
    n = len(rows)

    XL, XR = 0.0, 1.0
    # left-hand position: the benchmark score itself, rescaled onto the same
    # vertical span so the two columns are comparable by eye. Ties coincide,
    # which is the point of the figure.
    lo, hi = gen.min(), gen.max()
    def left_y(v):
        return (v - lo) / (hi - lo) * (n - 1)

    for i in range(n):
        y0, y1 = left_y(gen[i]), ypos[i]
        worse = real[i] > 0.5
        ax.plot([XL, XR], [y0, y1], "-", lw=1.6, alpha=0.75,
                color="#b2182b" if worse else "#2166ac",
                zorder=1, solid_capstyle="round")
        ax.plot([XR], [y1], "o", ms=7, zorder=3,
                color="#b2182b" if worse else "#2166ac")
        ax.annotate(f"  {name[i]}  ({real[i]:.3f})", (XR, y1),
                    va="center", ha="left", fontsize=9, zorder=4)

    # the tie cluster, drawn once
    ty = left_y(ceiling)
    ax.plot([XL], [ty], "o", ms=10, color="#333333", zorder=5)
    ax.annotate(f"{len(tied)} methods tied\nat $F_1$ = {ceiling:.4f}",
                (XL, ty), xytext=(-16, 0), textcoords="offset points",
                va="center", ha="right", fontsize=9.5, zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0",
                          ec="#999999", lw=0.8))
    # The Random Forest sits at 0.9963, which on this axis is 0.2 of a row
    # below the tie point -- close enough that its label printed underneath the
    # tie box. Left labels are therefore repelled to a legible spacing and
    # joined to their true position by a leader line, so the label moves but
    # the datum does not.
    untied = sorted((i for i in range(n) if i not in tied),
                    key=lambda i: -gen[i])
    MINGAP, taken = 0.78, [ty]
    for i in untied:
        y_true = left_y(gen[i])
        y_lab = y_true
        while any(abs(y_lab - t) < MINGAP for t in taken):
            y_lab -= 0.16
        taken.append(y_lab)
        if abs(y_lab - y_true) > 1e-9:
            ax.plot([XL - 0.013, XL - 0.055], [y_true, y_lab], "-",
                    lw=0.7, color="#999999", zorder=2)
        ax.plot([XL], [y_true], "o", ms=5, color="#777777", zorder=3)
        ax.annotate(f"{gen[i]:.4f}  ", (XL - 0.055, y_lab),
                    va="center", ha="right", fontsize=8.5, color="#555555")

    ax.set_xlim(-0.60, 1.62)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xticks([XL, XR])
    ax.set_xticklabels(["$F_1$ on the\ngenerated benchmark",
                        "false-positive rate\non real CDC code"], fontsize=10)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_title(f"Spearman $\\rho$ = {rho:.3f} "
                 f"($p$ = {p:.2f}) between the two columns",
                 fontsize=10, pad=12)

    out = J.FIGURES / "fig_rankflip.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")

    J.save("j49_rankflip_figure", dict(
        n_methods=n, n_tied=len(tied),
        tied_real_min=float(real[tied].min()),
        tied_real_max=float(real[tied].max()),
        tied_real_span=float(real[tied].max() - real[tied].min()),
        spearman_rho=rho, spearman_p=p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
