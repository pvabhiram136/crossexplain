"""j41 - the feature-shift figure for Section V, regenerated from j19.

`figures/fig_feature_shift.png` already existed, produced by an
earlier script and nothing tied it to a result file that is still current. The
project's rule is that every figure in the paper is generated from result JSON,
so this derives the shifts from the CURRENT corpus and writes its own summary so
j34 can check the prose against it.

The result involves no training, so it is unaffected by the dispersion of
Sections IX and X.
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

NICE = {"$_DFF_P_": "D flip-flop", "is_ff": "is flip-flop",
        "comb": "combinational", "$_MUX_": "multiplexer",
        "src_dom": "source domain", "FF_OTHER": "other flip-flop",
        "$_XOR_": "XOR", "PORT_IN": "input port"}


def feature_shifts(top_n=8):
    """Recompute the shift from the CURRENT corpus.

    This used to read `feature_shifts` out of j19_diagnostic.json, which was
    computed against the 31-crossing corpus. When the corpus grew those numbers
    went stale silently -- the figure still generated, the audit still passed,
    and Section V-F would have quoted the old corpus. Deriving them here ties
    the figure to whatever the corpus currently is.
    """
    sys.path.insert(0, str(J.ROOT))
    from crossexplain.netlist_graph import CELL_VOCAB                # noqa: E402

    names = list(CELL_VOCAB) + ["is_ff", "dst_dom", "src_dom", "comb",
                                "log_fanin", "log_fanout", "seed", "dist"]

    def mean(samples):
        return np.stack([np.asarray(s["node_feats"], dtype=np.float32).mean(0)
                         for s in samples]).mean(0)

    train = J.load("train")
    gs = mean([s for s in train if s["label_safe"] == 1])
    gu = mean([s for s in train if s["label_safe"] == 0])
    fr = mean(J.confirmation_set())
    shift = fr - gs

    order = np.argsort(-np.abs(shift))[:top_n]
    rows = [dict(feature=names[i], gen_safe=float(gs[i]),
                 gen_unsafe=float(gu[i]), found_safe=float(fr[i]),
                 shift=float(shift[i])) for i in order]
    # "toward the unsafe mean" = the found value moved past gen_safe in the
    # direction of gen_unsafe
    toward = sum(1 for r in rows
                 if (r["found_safe"] - r["gen_safe"]) *
                    (r["gen_unsafe"] - r["gen_safe"]) > 0)
    return rows, toward


def main():
    J.banner("j41  feature-shift figure for Section V")
    rows, toward = feature_shifts()
    print(f"  {len(rows)} features, {toward} of {len(rows)} shift toward the "
          f"generated UNSAFE mean")

    mux = next(r for r in rows if r["feature"] == "$_MUX_")
    print(f"  multiplexer share: generated safe {mux['gen_safe']:.3f}, "
          f"generated unsafe {mux['gen_unsafe']:.3f}, "
          f"found real {mux['found_safe']:.3f}")

    labels = [NICE.get(r["feature"], r["feature"]) for r in rows]
    y = np.arange(len(rows))[::-1]
    gs = [r["gen_safe"] for r in rows]
    gu = [r["gen_unsafe"] for r in rows]
    fr = [r["found_safe"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    h = 0.26
    ax.barh(y + h, gs, height=h, color="#92c5de", label="generated, safe")
    ax.barh(y, gu, height=h, color="#f4a582", label="generated, unsafe")
    ax.barh(y - h, fr, height=h, color="#1a1a1a", label="third-party, safe")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean share of nodes in a crossing subgraph")
    ax.grid(alpha=0.25, ls=":", axis="x")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    # No axes title: the LaTeX caption states this, and a title here would
    # collide with the legend placed above the axes.
    fig.tight_layout()
    out = J.FIGURES / "fig_feature_shift.png"
    fig.savefig(out, dpi=200)
    print(f"  -> {out.relative_to(J.ROOT)}")

    J.save("j41_feature_shift", dict(
        n_features=len(rows),
        n_toward_unsafe=toward,
        mux_gen_safe=mux["gen_safe"], mux_gen_unsafe=mux["gen_unsafe"],
        mux_found=mux["found_safe"],
        rows=rows))


if __name__ == "__main__":
    main()
