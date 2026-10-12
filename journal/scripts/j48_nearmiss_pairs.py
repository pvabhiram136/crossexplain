"""j48 - subgraph-size statistics for the three near-miss template pairs.

Section III claimed that MUX_UNSYNC_QUAL and MUX_HOLD span "33 to 101 nodes
against 34 to 100, with means of 60.7 and 59.2". j47 found that no result file
derives those numbers, and they turn out to be the *train split alone*. The
sentence reads as a property of the templates. On test_id the two means invert
(51.2 against 53.8), so the split-specific form is both unscoped and, read as
written, wrong about the direction.

This script derives the statistic the sentence actually wants to make -- over
the whole corpus, for each of the three pairs marked with a dagger in
Table I -- and writes it out so j34 can check the prose against it.

Overlap is reported as the Jaccard overlap of the two [min, max] intervals,
which is the quantity the claim "no size threshold separates them" is about: a
threshold on node count can only separate two templates if their ranges are
close to disjoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

# (unsafe member, safe member) -- the dagger in Table I marks the member whose
# counterpart carries the opposite label.
PAIRS = [
    ("MUX_UNSYNC_QUAL", "MUX_HOLD"),
    ("SYNC2_LEAK", "SYNC2"),
    ("COMB_BEFORE_SYNC", "SYNC2_DEEPSRC"),
]


def sizes(samples, template):
    return np.array([len(s["node_feats"]) for s in samples
                     if s["template"] == template])


def interval_overlap(a, b):
    """Jaccard overlap of the two closed intervals spanned by a and b."""
    lo = max(a.min(), b.min())
    hi = min(a.max(), b.max())
    inter = max(0, hi - lo)
    union = max(a.max(), b.max()) - min(a.min(), b.min())
    return float(inter / union) if union else 1.0


def main() -> int:
    J.banner("j48  near-miss pair subgraph sizes")

    allS = [s for sp in J.SPLITS for s in J.load(sp)]
    print(f"  corpus: {len(allS)} labeled samples over {len(J.SPLITS)} splits\n")

    out = {"n_samples": len(allS), "pairs": {}}
    for unsafe, safe in PAIRS:
        u, s = sizes(allS, unsafe), sizes(allS, safe)
        ov = interval_overlap(u, s)
        rec = dict(
            unsafe=unsafe, safe=safe,
            unsafe_n=int(len(u)), safe_n=int(len(s)),
            unsafe_mean=round(float(u.mean()), 1),
            safe_mean=round(float(s.mean()), 1),
            unsafe_min=int(u.min()), unsafe_max=int(u.max()),
            safe_min=int(s.min()), safe_max=int(s.max()),
            mean_gap=round(abs(float(u.mean() - s.mean())), 1),
            range_overlap=round(ov, 3),
        )
        out["pairs"][f"{unsafe}|{safe}"] = rec
        print(f"  {unsafe} (unsafe) vs {safe} (safe)")
        print(f"    {unsafe:18s} n={len(u):4d}  {u.min():3d}-{u.max():3d}"
              f"  mean {u.mean():6.1f}")
        print(f"    {safe:18s} n={len(s):4d}  {s.min():3d}-{s.max():3d}"
              f"  mean {s.mean():6.1f}")
        print(f"    mean gap {rec['mean_gap']:.1f} nodes,"
              f" range overlap {ov:.3f}\n")

    # The per-split breakdown is what exposed the original error, so it is
    # kept in the record rather than discarded.
    out["by_split"] = {}
    for sp in J.SPLITS:
        S = J.load(sp)
        d = {}
        for unsafe, safe in PAIRS:
            u, s = sizes(S, unsafe), sizes(S, safe)
            if len(u) and len(s):
                d[f"{unsafe}|{safe}"] = dict(
                    unsafe_mean=round(float(u.mean()), 1),
                    safe_mean=round(float(s.mean()), 1))
        out["by_split"][sp] = d

    worst = max(out["pairs"].values(), key=lambda r: r["mean_gap"])
    print(f"  largest mean gap of any pair: {worst['mean_gap']} nodes"
          f" ({worst['unsafe']} vs {worst['safe']})")
    J.save("j48_nearmiss_pairs", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
