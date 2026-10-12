"""P0.5 - does the headline finding survive a fair structural baseline?

The paper's centrepiece is that a rule-based matcher scores 0.000 on MUX_HOLD
(systematic false positives) and 0.000 on BUS_NOSYNC (systematic false
negatives), while CrossExplain scores 1.000 on both. Part of that gap is real
and part of it is a weak rule set (see journal/strong_matcher.py).

This script runs the original matcher, the strengthened matcher, the flat
Random Forest, and the GNN ensemble side by side, with per-template breakdowns
and paired design-clustered tests. If the finding survives, it is worth much
more than it was. If it does not, we need to know now.

Threshold fitting for the strengthened matcher happens on TRAIN only.

Run:  python journal/scripts/j04_strong_baseline.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from strong_matcher import strong_predict, fit_thresholds     # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import pattern_matcher_predict, FlatML   # noqa: E402

OUT = {}

# ------------------------------------------------------- fit on train only
J.banner("P0.5a  Fitting the strengthened matcher on TRAIN only")
train = J.load("train")
ratio, tr_acc = fit_thresholds(train)
print(f"  gray/carry XOR ratio = {ratio:.2f}   train accuracy = {tr_acc:.4f}")
orig_tr = np.array([p == 0 for p in pattern_matcher_predict(train)])
print(f"  original matcher      train accuracy = "
      f"{J.accuracy(orig_tr, J.y_of(train)):.4f}")
OUT["fit"] = dict(gray_xor_ratio=ratio, strong_train_acc=tr_acc,
                  original_train_acc=J.accuracy(orig_tr, J.y_of(train)))

models = J.load_seed_models()
flat = FlatML(seed=0).fit(train)


def all_preds(samples):
    return {
        "matcher_original": np.array([p == 0 for p in pattern_matcher_predict(samples)]),
        "matcher_strong": np.array([p == 0 for p in strong_predict(samples, gray_xor_ratio=ratio)]),
        "random_forest": np.array([d == 0 for d in flat.predict(samples)[0]]),
        "crossexplain_ens": J.seed_probs(models, samples).mean(0) > 0.5,
    }


# ----------------------------------------------------------- headline table
J.banner("P0.5b  Detection, all methods, all splits (design-clustered CIs)")
tables = {}
for sp in ["test_id", "test_big", "test_real"]:
    s = J.load(sp)
    y, u = J.y_of(s), J.units(s)
    preds = all_preds(s)
    tables[sp] = {}
    print(f"\n  --- {sp} (n={len(s)}, {J.effective_n(s)['designs']} designs) ---")
    for m, p in preds.items():
        mt = J.metrics(p, y)
        ci = J.clustered_ci(p, y, u["design"], "f1", B=2000)
        tables[sp][m] = dict(**mt, ci_design=ci)
        print(f"    {m:18s} F1={mt['f1']:.3f} [{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}]  "
              f"acc={mt['accuracy']:.3f}  FPR={mt['false_positive_rate']:.3f}")
    # paired tests against the strengthened matcher, the fair opponent
    for base in ["matcher_original", "matcher_strong", "random_forest"]:
        r = J.paired_clustered(preds["crossexplain_ens"], preds[base], y,
                               u["design"], "f1", B=4000)
        tables[sp][f"paired_ens_vs_{base}"] = r
        flag = "SIGNIFICANT" if r["significant"] else "not significant"
        print(f"    dF1 ens - {base:18s} = {r['delta']:+.3f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_value']:.3f}  {flag}")
OUT["detection"] = tables

# --------------------------------------------------- per-template autopsy
J.banner("P0.5c  Per-template accuracy: does the MUX_HOLD finding survive?")
for sp in ["test_id", "test_real"]:
    s = J.load(sp)
    y = J.y_of(s)
    preds = all_preds(s)
    tpls = sorted({x["template"] for x in s if x["template"]})
    print(f"\n  --- {sp} ---")
    print(f"    {'template':20s} {'n':>4s} {'label':>6s} " +
          "".join(f"{m[:14]:>16s}" for m in preds))
    rows = {}
    for t in tpls:
        m_ = np.array([x["template"] == t for x in s])
        lab = "safe" if not y[m_][0] else "unsafe"
        rows[t] = {"n": int(m_.sum()), "label": lab}
        line = f"    {t:20s} {m_.sum():4d} {lab:>6s} "
        for mn, p in preds.items():
            a = J.accuracy(p[m_], y[m_])
            rows[t][mn] = a
            line += f"{a:16.3f}"
        print(line)
    OUT.setdefault("per_template", {})[sp] = rows

# ------------------------------------------------- what changed, and why
J.banner("P0.5d  Verdict")
for sp in ["test_id", "test_real"]:
    r = OUT["per_template"][sp]
    for t in ["MUX_HOLD", "BUS_NOSYNC"]:
        if t in r:
            print(f"  {sp:9s} {t:12s} original={r[t]['matcher_original']:.3f}  "
                  f"strengthened={r[t]['matcher_strong']:.3f}  "
                  f"CrossExplain={r[t]['crossexplain_ens']:.3f}")

J.save("j04_strong_baseline", OUT)
