"""P0.4 - encoder depth ablation.

The conference paper asserts, and HANDOFF.md elevates to a non-negotiable
design decision:

    "The qualifier chain of a mux-recirculation crossing sits roughly five
     hops from the capture flip-flop, so a three-layer encoder cannot see the
     evidence that separates MUX_HOLD from MUX_UNSYNC_QUAL."

No experiment in results.json supports this. It is mechanistically plausible
and probably true, but asserted. This script measures it: depths 1 to 7, three
seeds each, reporting accuracy on the near-miss pair specifically rather than
only in aggregate (aggregate accuracy is saturated and would hide the effect).

Run:  python journal/scripts/j03_depth_ablation.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.train import train_model, predict           # noqa: E402

DEPTHS = [1, 2, 3, 4, 5, 6, 7]
SEEDS = [0, 1, 2]
NEAR_MISS = ["MUX_HOLD", "MUX_UNSYNC_QUAL"]

train = J.load("train")
val = J.load("val")
test_id = J.load("test_id")
real = J.load("test_real")

templates = sorted({s["template"] for s in test_id if s["template"]})
tpl_idx = {t: np.array([s["template"] == t for s in test_id]) for t in templates}
y_id, y_real = J.y_of(test_id), J.y_of(real)
u_real = J.units(real)

J.banner("P0.4  Encoder depth ablation (3 seeds per depth)")
print(f"  depths {DEPTHS}, near-miss pair {NEAR_MISS}")
print(f"  train={len(train)}  val={len(val)}  test_id={len(test_id)}  real={len(real)}")

OUT = {"depths": DEPTHS, "seeds": SEEDS, "per_depth": {}}
t_start = time.time()

for d in DEPTHS:
    per_seed = []
    probs_id, probs_real = [], []
    for sd in SEEDS:
        t0 = time.time()
        m = train_model(train, val, arch="gat", layers=d, epochs=40,
                        verbose=False, seed=sd)
        p_id = np.array(predict(m, test_id)[0])
        p_re = np.array(predict(m, real)[0])
        probs_id.append(p_id)
        probs_real.append(p_re)
        row = {
            "seed": sd,
            "test_id_f1": J.f1(p_id > 0.5, y_id),
            "test_real_f1": J.f1(p_re > 0.5, y_real),
            "per_template_acc": {
                t: J.accuracy(p_id[tpl_idx[t]] > 0.5, y_id[tpl_idx[t]])
                for t in templates},
            "n_params": int(sum(p.numel() for p in m.parameters())),
            "train_sec": round(time.time() - t0, 1),
        }
        per_seed.append(row)
        nm = "  ".join(f"{t}={row['per_template_acc'][t]:.3f}" for t in NEAR_MISS)
        print(f"  depth {d}  seed {sd}:  test_id F1={row['test_id_f1']:.4f}  "
              f"real F1={row['test_real_f1']:.3f}   {nm}   ({row['train_sec']}s)")

    ens_id = np.mean(probs_id, axis=0)
    ens_re = np.mean(probs_real, axis=0)
    OUT["per_depth"][d] = {
        "per_seed": per_seed,
        "mean_test_id_f1": float(np.mean([r["test_id_f1"] for r in per_seed])),
        "mean_test_real_f1": float(np.mean([r["test_real_f1"] for r in per_seed])),
        "std_test_real_f1": float(np.std([r["test_real_f1"] for r in per_seed])),
        "mean_per_template_acc": {
            t: float(np.mean([r["per_template_acc"][t] for r in per_seed]))
            for t in templates},
        "ensemble_test_id_f1": J.f1(ens_id > 0.5, y_id),
        "ensemble_real_f1": J.f1(ens_re > 0.5, y_real),
        "ensemble_real_ci_design": J.clustered_ci(ens_re > 0.5, y_real,
                                                  u_real["design"], "f1", B=2000),
        "ensemble_per_template_acc": {
            t: J.accuracy(ens_id[tpl_idx[t]] > 0.5, y_id[tpl_idx[t]])
            for t in templates},
        "n_params": per_seed[0]["n_params"],
    }

J.banner("Summary: does depth matter, and where?")
print(f"  {'depth':>5s} {'params':>7s} {'test_id F1':>10s} {'real F1':>12s} "
      + "  ".join(f"{t:>16s}" for t in NEAR_MISS))
for d in DEPTHS:
    r = OUT["per_depth"][d]
    nm = "  ".join(f"{r['mean_per_template_acc'][t]:16.3f}" for t in NEAR_MISS)
    print(f"  {d:5d} {r['n_params']:7d} {r['mean_test_id_f1']:10.4f} "
          f"{r['mean_test_real_f1']:7.3f}+-{r['std_test_real_f1']:.3f} {nm}")

print("\n  full per-template mean accuracy (test_id):")
hdr = "  " + " " * 8 + "".join(f"{t[:11]:>12s}" for t in templates)
print(hdr)
for d in DEPTHS:
    r = OUT["per_depth"][d]["mean_per_template_acc"]
    print(f"  depth {d} " + "".join(f"{r[t]:12.3f}" for t in templates))

# the specific claim under test
J.banner("Verdict on the paper's depth claim")
lo = [OUT["per_depth"][d]["mean_per_template_acc"] for d in DEPTHS if d <= 3]
hi = [OUT["per_depth"][d]["mean_per_template_acc"] for d in DEPTHS if d >= 5]
for t in NEAR_MISS:
    a = float(np.mean([x[t] for x in lo]))
    b = float(np.mean([x[t] for x in hi]))
    print(f"  {t:18s} depth<=3 mean acc = {a:.3f}   depth>=5 mean acc = {b:.3f}   "
          f"delta = {b - a:+.3f}")
    OUT.setdefault("claim_check", {})[t] = dict(shallow=a, deep=b, delta=b - a)

print(f"\n  total wall time {time.time() - t_start:.0f}s")
J.save("j03_depth_ablation", OUT)
