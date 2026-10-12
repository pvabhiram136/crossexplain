"""Does changing the readout fix the idiom-by-size interaction?

j15 showed safe-side false positives require both the qualifier-held idiom and
a large neighbourhood, and proposed mean-pooling dilution as the mechanism.
This trains four readout variants from scratch on the unchanged generated
training split and evaluates each on:

  test_id            must not regress (the shipped model is at 1.000)
  test_real          the old real split
  wrapped corpus     84 samples, mean 254 nodes
  the j15 2x2        the controlled idiom-by-size cells

The prediction, if dilution is the mechanism: `seed_only` and `dist_seed`
should cut the false-positive rate on large MUX_HOLD while leaving test_id
alone. If they do not, the mechanism is wrong and the interaction has another
cause.

Nothing is trained or tuned on any wrapped or real design.

Run:  python journal/scripts/j16_readout_fix.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from readouts import READOUTS, train_readout, predict_readout  # noqa: E402
from wrappers import extract_wrapped                          # noqa: E402

SEEDS = [0, 1, 2]
LEVELS = ["O0", "O1", "O2", "O3"]
CORPUS = J.JROOT / "corpus"


def load_wrapped(folder):
    out = []
    d = CORPUS / folder
    for gtf in sorted(d.glob("*.json")):
        if any(gtf.name.endswith(f".{l}.json") for l in LEVELS):
            continue
        gt = json.loads(gtf.read_text())
        for lvl in LEVELS:
            jf = d / f"{gtf.stem}.{lvl}.json"
            if not jf.exists():
                continue
            try:
                ss, _ = extract_wrapped(jf, gt, gtf.stem, lvl)
            except Exception:                                  # noqa: BLE001
                continue
            out.extend(asdict(s) for s in ss if s.label_safe != -1)
    return out


J.banner("Loading splits")
train = J.load("train")
val = J.load("val")
test_id = J.load("test_id")
real = J.load("test_real")
wrapped = load_wrapped("wrapped")

# j15 wrote the netlists but not its ground-truth JSON, so the cell labels are
# reconstructed here from the same deterministic generator rather than read
# from disk.
from wrappers import make_wrapper, CORES                      # noqa: E402
CELL = {"pin_sync2": ("SYNC2", "small"), "sync2_deep": ("SYNC2", "large"),
        "mux_hold_narrow": ("MUX_HOLD", "small"),
        "sync2_held": ("MUX_HOLD", "large")}
grid = []
gdir = CORPUS / "sizeidiom"
for core in CORES:
    for style, cell in CELL.items():
        name = f"sz_{core}_{style}"
        _, gt = make_wrapper(core, [style], module_name=name)
        for lvl in LEVELS:
            jf = gdir / f"{name}.{lvl}.json"
            if not jf.exists():
                continue
            try:
                ss, _ = extract_wrapped(jf, gt, name, lvl)
            except Exception:                                  # noqa: BLE001
                continue
            for s in ss:
                if s.label_safe == -1:
                    continue
                d = asdict(s)
                d["cell"] = cell
                grid.append(d)
print(f"  train {len(train)}  test_id {len(test_id)}  test_real {len(real)}  "
      f"wrapped {len(wrapped)}  grid {len(grid)}")

y_id, y_re = J.y_of(test_id), J.y_of(real)
y_wr = J.y_of(wrapped)
u_re, u_wr = J.units(real), J.units(wrapped)
wr_nodes = np.array([len(s["node_feats"]) for s in wrapped])
wr_tpl = np.array([s["template"] for s in wrapped])
g_cell = np.array([f"{s['cell'][0]}/{s['cell'][1]}" for s in grid])
y_g = J.y_of(grid)

J.banner(f"Training {len(READOUTS)} readout variants x {len(SEEDS)} seeds")
OUT = {}
for ro in READOUTS:
    probs = {k: [] for k in ("id", "re", "wr", "g")}
    for sd in SEEDS:
        m = train_readout(train, val, readout=ro, seed=sd, epochs=40)
        probs["id"].append(np.array(predict_readout(m, test_id)))
        probs["re"].append(np.array(predict_readout(m, real)))
        probs["wr"].append(np.array(predict_readout(m, wrapped)))
        probs["g"].append(np.array(predict_readout(m, grid)))
    ens = {k: np.mean(v, axis=0) > 0.5 for k, v in probs.items()}

    r = dict(
        test_id_f1=J.f1(ens["id"], y_id),
        test_real=J.metrics(ens["re"], y_re),
        test_real_ci=J.clustered_ci(ens["re"], y_re, u_re["design"], "f1",
                                    B=2000),
        wrapped=J.metrics(ens["wr"], y_wr),
        wrapped_safe_fpr=float(ens["wr"][~y_wr].mean()),
        wrapped_unsafe_recall=float(ens["wr"][y_wr].mean()),
        mux_hold_large_fpr=float(ens["wr"][(wr_tpl == "MUX_HOLD")].mean()),
        grid={c: float(ens["g"][g_cell == c].mean())
              for c in sorted(set(g_cell))},
    )
    OUT[ro] = r
    print(f"\n  --- {ro} ---")
    print(f"    test_id F1        {r['test_id_f1']:.4f}")
    print(f"    test_real F1      {r['test_real']['f1']:.3f} "
          f"[{r['test_real_ci']['ci_lo']:.3f},{r['test_real_ci']['ci_hi']:.3f}]"
          f"  FPR {r['test_real']['false_positive_rate']:.3f}")
    print(f"    wrapped F1        {r['wrapped']['f1']:.3f}  "
          f"safe-FPR {r['wrapped_safe_fpr']:.3f}  "
          f"unsafe-recall {r['wrapped_unsafe_recall']:.3f}")
    print(f"    grid FPR          " +
          "  ".join(f"{c}={v:.3f}" for c, v in r["grid"].items()))

J.banner("Summary: does any readout fix the interaction?")
base = OUT["mean_seed"]
print(f"  {'readout':11s} {'test_id':>8s} {'real F1':>8s} {'real FPR':>9s} "
      f"{'wrap safe-FPR':>14s} {'MUX_HOLD/large':>15s}")
for ro, r in OUT.items():
    mark = "  <- shipped" if ro == "mean_seed" else ""
    print(f"  {ro:11s} {r['test_id_f1']:8.4f} {r['test_real']['f1']:8.3f} "
          f"{r['test_real']['false_positive_rate']:9.3f} "
          f"{r['wrapped_safe_fpr']:14.3f} "
          f"{r['grid'].get('MUX_HOLD/large', float('nan')):15.3f}{mark}")

best = min(OUT.items(), key=lambda kv: kv[1]["wrapped_safe_fpr"])
print(f"\n  lowest wrapped safe-FPR: {best[0]} at "
      f"{best[1]['wrapped_safe_fpr']:.3f} against "
      f"{base['wrapped_safe_fpr']:.3f} for the shipped readout")
print(f"  its test_id F1 is {best[1]['test_id_f1']:.4f} "
      f"(shipped {base['test_id_f1']:.4f})")

J.save("j16_readout_fix", OUT)
