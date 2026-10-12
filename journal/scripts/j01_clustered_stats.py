"""P0.1 + P0.2 - clustered inference and effective sample size.

Replaces the sample-level bootstrap of scripts/07_error_analysis.py with one
that respects how the data was collected, and adds the paired difference tests
that the conference paper never ran.

Three things are established here:

  1. Effective n. Every split contains far fewer independent units than rows,
     because each crossing is synthesized at up to four optimization levels.
  2. Clustered intervals. Resampling designs rather than rows roughly doubles
     every confidence interval on test_real.
  3. Paired tests. Comparing two overlapping marginal intervals is not a
     significance test. Bootstrapping the difference on shared clusters is.

Run:  python journal/scripts/j01_clustered_stats.py
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import pattern_matcher_predict, FlatML   # noqa: E402

OUT = {}

# ------------------------------------------------------------------ 1. n_eff
J.banner("P0.2  Effective sample size (rows are not independent observations)")
graph_dupes = {}
for sp in J.SPLITS:
    s = J.load(sp)
    eff = J.effective_n(s)

    # how many rows are structurally distinct graphs?
    seen, by_cross = set(), defaultdict(dict)
    for x in s:
        h = hashlib.md5(repr((x["node_feats"], sorted(x["edges"]),
                              sorted(x["seeds"]))).encode()).hexdigest()
        seen.add(h)
        by_cross[(x["design"], x["cid"])][x["level"]] = h
    eff["distinct_graphs"] = len(seen)
    eff["rows_per_crossing"] = round(eff["samples"] / max(1, eff["crossings"]), 2)

    # which optimization levels collapse onto each other
    pairs = defaultdict(lambda: [0, 0])
    lv = ["O0", "O1", "O2", "O3"]
    for d in by_cross.values():
        for i in range(len(lv)):
            for k in range(i + 1, len(lv)):
                a, b = lv[i], lv[k]
                if a in d and b in d:
                    pairs[f"{a}=={b}"][1] += 1
                    pairs[f"{a}=={b}"][0] += int(d[a] == d[b])
    eff["identical_level_pairs"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(pairs.items())}
    graph_dupes[sp] = eff
    print(f"  {sp:9s} rows={eff['samples']:5d}  crossings={eff['crossings']:4d}  "
          f"designs={eff['designs']:4d}  distinct graphs={eff['distinct_graphs']:4d}  "
          f"rows/crossing={eff['rows_per_crossing']}")
    if eff["identical_level_pairs"]:
        print(f"            identical across levels: {eff['identical_level_pairs']}")
OUT["effective_n"] = graph_dupes

# ---------------------------------------------------------------- 2. models
J.banner("Loading the five cached seeds and both baselines")
models = J.load_seed_models()
train = J.load("train")
flat = FlatML(seed=0).fit(train)
print(f"  {len(models)} GNN seeds, RandomForest fitted on {len(train)} train rows")

real = J.load("test_real")
y = J.y_of(real)
u = J.units(real)
lv = J.levels(real)

P = J.seed_probs(models, real)          # [5, 68]
ens = P.mean(0)
preds = {
    "crossexplain_ens": ens > 0.5,
    "crossexplain_single": P[3] > 0.5,          # median-by-generated-val seed
    "pattern_matcher": np.array([p == 0 for p in pattern_matcher_predict(real)]),
    "random_forest": np.array([d == 0 for d in flat.predict(real)[0]]),
}

# --------------------------------------------------- 3. clustered intervals
J.banner("P0.1a  How much does the resampling unit matter? (test_real)")
ci = {}
for mname, pr in preds.items():
    ci[mname] = {}
    for uname in ["sample", "crossing", "design"]:
        r = J.clustered_ci(pr, y, u[uname], "f1", B=5000)
        ci[mname][uname] = r
    row = ci[mname]
    print(f"  {mname:22s} F1={row['sample']['point']:.3f}   "
          f"sample [{row['sample']['ci_lo']:.3f},{row['sample']['ci_hi']:.3f}] w={row['sample']['width']:.3f}   "
          f"design [{row['design']['ci_lo']:.3f},{row['design']['ci_hi']:.3f}] w={row['design']['width']:.3f}")
OUT["ci_by_unit_test_real"] = ci

print("\n  per-seed, sample-level (as published) vs design-clustered:")
seed_ci = {}
for i in range(len(models)):
    a = J.clustered_ci(P[i] > 0.5, y, u["sample"], "f1", B=5000)
    b = J.clustered_ci(P[i] > 0.5, y, u["design"], "f1", B=5000)
    seed_ci[i] = {"sample": a, "design": b}
    print(f"    seed{i}  F1={a['point']:.3f}   sample w={a['width']:.3f}   "
          f"design w={b['width']:.3f}   design CI=[{b['ci_lo']:.3f},{b['ci_hi']:.3f}]")
OUT["per_seed_ci_test_real"] = seed_ci

# ------------------------------------------------------- 4. paired testing
J.banner("P0.1b  Paired design-clustered tests (the correct comparison)")
paired = {}
for base in ["pattern_matcher", "random_forest"]:
    for metric in ["f1", "accuracy"]:
        r = J.paired_clustered(preds["crossexplain_ens"], preds[base], y,
                               u["design"], metric, B=10000)
        paired[f"ens_vs_{base}_{metric}"] = r
        flag = "SIGNIFICANT" if r["significant"] else "not significant"
        print(f"  d{metric:8s} ensemble - {base:16s} = {r['delta']:+.3f}  "
              f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p_value']:.3f}   {flag}")

print("\n  deduplicated view: O0 only, 17 distinct crossings")
m0 = lv == "O0"
for base in ["pattern_matcher", "random_forest"]:
    r = J.paired_clustered(preds["crossexplain_ens"][m0], preds[base][m0],
                           y[m0], u["design"][m0], "f1", B=10000)
    paired[f"O0_ens_vs_{base}_f1"] = r
    flag = "SIGNIFICANT" if r["significant"] else "not significant"
    print(f"    dF1 ensemble - {base:16s} = {r['delta']:+.3f}  "
          f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p_value']:.3f}   {flag}")
OUT["paired_tests_test_real"] = paired

# per-level point estimates (the replication is visible here)
OUT["by_level_test_real"] = {
    l: {m: J.metrics(pr[lv == l], y[lv == l]) for m, pr in preds.items()}
    for l in ["O0", "O1", "O2", "O3"]}
print("\n  ensemble F1 by level: " + "  ".join(
    f"{l}={OUT['by_level_test_real'][l]['crossexplain_ens']['f1']:.3f}"
    for l in ["O0", "O1", "O2", "O3"]))

# ------------------------------------- 5. the generated splits, for contrast
J.banner("P0.1c  Generated splits (clustered by design)")
gen = {}
for sp in ["test_id", "test_big"]:
    s = J.load(sp)
    ys, us = J.y_of(s), J.units(s)
    pe = J.seed_probs(models, s).mean(0) > 0.5
    pm = np.array([p == 0 for p in pattern_matcher_predict(s)])
    rf = np.array([d == 0 for d in flat.predict(s)[0]])
    gen[sp] = {
        "effective_n": J.effective_n(s),
        "crossexplain_ens": J.clustered_ci(pe, ys, us["design"], "f1", B=2000),
        "pattern_matcher": J.clustered_ci(pm, ys, us["design"], "f1", B=2000),
        "random_forest": J.clustered_ci(rf, ys, us["design"], "f1", B=2000),
        "paired_vs_matcher": J.paired_clustered(pe, pm, ys, us["design"], "f1", B=2000),
        "paired_vs_rf": J.paired_clustered(pe, rf, ys, us["design"], "f1", B=2000),
    }
    g = gen[sp]
    print(f"  {sp}: ensemble F1={g['crossexplain_ens']['point']:.3f} "
          f"[{g['crossexplain_ens']['ci_lo']:.3f},{g['crossexplain_ens']['ci_hi']:.3f}]  "
          f"vs matcher d={g['paired_vs_matcher']['delta']:+.3f} "
          f"p={g['paired_vs_matcher']['p_value']:.3f}  "
          f"vs RF d={g['paired_vs_rf']['delta']:+.3f} "
          f"p={g['paired_vs_rf']['p_value']:.3f}")
OUT["generated_splits"] = gen

# --------------------------------------------------------- 6. power target
J.banner("P0.1d  How many designs would we need? (1/sqrt(k) extrapolation)")
half = paired["ens_vs_pattern_matcher_f1"]
h0 = (half["ci_hi"] - half["ci_lo"]) / 2
k0 = half["n_clusters"]
delta = abs(half["delta"])
power = []
for k in [12, 20, 30, 40, 50, 60, 80, 100]:
    h = h0 * np.sqrt(k0 / k)
    power.append(dict(designs=k, half_width=round(float(h), 3),
                      detects_observed_delta=bool(h < delta)))
    print(f"  {k:4d} designs -> half-width {h:.3f}   "
          f"{'detects' if h < delta else 'does not detect'} delta={delta:.3f}")
OUT["power_projection"] = dict(observed_delta=float(delta), observed_half_width=float(h0),
                               observed_designs=int(k0), projection=power)

J.save("j01_clustered_stats", OUT)
print("\nDone. Headline: see paired_tests_test_real in the JSON.")
