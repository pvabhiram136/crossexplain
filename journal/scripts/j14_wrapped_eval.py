"""Evaluate the trained model on the wrapped-core corpus.

This is the direct test of P0 finding (d). That finding said the sim-to-real
gap is not a size effect but a *neighbourhood composition* effect: safe
crossings embedded in wide busy logic get read as hazard evidence, which shows
up as safe-side false positives. The old real split could only support that
inferentially, since its safe crossings averaged 43 nodes against 29.1 in
training. The wrapped corpus puts safe crossings of known label inside real
peripheral logic averaging 254 nodes, so the prediction is sharp:

  if finding (d) is right, FPR on the wrapped safe crossings should be high,
  and should rise with subgraph size.

Nothing here is trained or tuned on the wrapped corpus. The five seeds are the
released checkpoints from scripts/05_experiments.py, unchanged.

Run:  python journal/scripts/j14_wrapped_eval.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from wrappers import extract_wrapped                          # noqa: E402
from strong_matcher import strong_predict                     # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import pattern_matcher_predict, FlatML  # noqa: E402

WRAP = J.JROOT / "corpus" / "wrapped"
LEVELS = ["O0", "O1", "O2", "O3"]
RATIO = 0.70          # strengthened matcher threshold, fitted on train in j04

J.banner("Re-extracting the wrapped corpus")
samples = []
for gtf in sorted(WRAP.glob("*.json")):
    if any(gtf.name.endswith(f".{l}.json") for l in LEVELS):
        continue
    gt = json.loads(gtf.read_text())
    name = gtf.stem
    for lvl in LEVELS:
        jf = WRAP / f"{name}.{lvl}.json"
        if not jf.exists():
            continue
        try:
            ss, _ = extract_wrapped(jf, gt, name, lvl)
        except Exception as e:                                 # noqa: BLE001
            print(f"  {name} @{lvl}: {e}")
            continue
        samples.extend(asdict(s) for s in ss if s.label_safe != -1)

y = J.y_of(samples)
u = J.units(samples)
nodes = np.array([len(s["node_feats"]) for s in samples])
print(f"  {len(samples)} labeled samples, "
      f"{len(set(u['crossing']))} crossings, {len(set(u['design']))} designs")
print(f"  safe {int((~y).sum())} / unsafe {int(y.sum())}, "
      f"mean {nodes.mean():.1f} nodes")

J.banner("Detection on the wrapped corpus (design-clustered CIs)")
models = J.load_seed_models()
train = J.load("train")
flat = FlatML(seed=0).fit(train)

preds = {
    "crossexplain_ens": J.seed_probs(models, samples).mean(0) > 0.5,
    "matcher_textbook": np.array([p == 0 for p in pattern_matcher_predict(samples)]),
    "matcher_domain_aware": np.array([p == 0 for p in strong_predict(samples, gray_xor_ratio=RATIO)]),
    "random_forest": np.array([d == 0 for d in flat.predict(samples)[0]]),
}

OUT = {"n_samples": len(samples),
       "n_crossings": len(set(u["crossing"])),
       "n_designs": len(set(u["design"])),
       "mean_nodes": float(nodes.mean()), "detection": {}}

for m, p in preds.items():
    mt = J.metrics(p, y)
    ci = J.clustered_ci(p, y, u["design"], "f1", B=3000)
    OUT["detection"][m] = dict(**mt, ci_design=ci)
    print(f"  {m:22s} F1={mt['f1']:.3f} [{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}]  "
          f"acc={mt['accuracy']:.3f}  recall={mt['recall']:.3f}  "
          f"FPR={mt['false_positive_rate']:.3f}")

J.banner("The finding (d) test: false positives on SAFE crossings by size")
safe_mask = ~y
ens = preds["crossexplain_ens"]
print(f"  overall FPR on wrapped safe crossings : "
      f"{ens[safe_mask].mean():.3f}  ({int(ens[safe_mask].sum())}"
      f"/{int(safe_mask.sum())} flagged)")

# reference points from the conference artifact
R = json.loads((J.ROOT / "results" / "results.json").read_text())
old = R["test_real"]["crossexplain_ens3"]["detection"]
print(f"  same model, old real split FPR        : "
      f"{old['false_positive_rate']:.3f}   (mean 43.0 nodes)")

bands = [(0, 50), (50, 150), (150, 400), (400, 10 ** 9)]
print(f"\n  {'subgraph nodes':>16s} {'n':>4s} {'FPR':>7s}  safe crossings only")
size_rows = []
for lo, hi in bands:
    m = safe_mask & (nodes >= lo) & (nodes < hi)
    if not m.sum():
        continue
    fpr = float(ens[m].mean())
    size_rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), fpr=fpr))
    label = f"{lo}-{hi if hi < 10**9 else '+'}"
    print(f"  {label:>16s} {int(m.sum()):4d} {fpr:7.3f}")
OUT["safe_fpr_by_size"] = size_rows
OUT["safe_fpr_overall"] = float(ens[safe_mask].mean())
OUT["old_real_split_fpr"] = old["false_positive_rate"]

# recall on unsafe, for contrast: does the model still catch real hazards?
print(f"\n  recall on wrapped UNSAFE crossings    : {ens[y].mean():.3f}"
       f"  ({int(ens[y].sum())}/{int(y.sum())} caught)")
OUT["unsafe_recall"] = float(ens[y].mean())

J.banner("Per-template breakdown")
tpl = np.array([s["template"] for s in samples])
rows = {}
for t in sorted(set(tpl)):
    m = tpl == t
    lab = "safe" if not y[m][0] else "unsafe"
    acc = J.accuracy(ens[m], y[m])
    rows[t] = dict(n=int(m.sum()), label=lab, ens_acc=acc,
                   mean_nodes=float(nodes[m].mean()))
    print(f"  {t:14s} {lab:6s} n={int(m.sum()):3d}  "
          f"CrossExplain acc={acc:.3f}  mean nodes={nodes[m].mean():.0f}")
OUT["per_template"] = rows

J.save("j14_wrapped_eval", OUT)
