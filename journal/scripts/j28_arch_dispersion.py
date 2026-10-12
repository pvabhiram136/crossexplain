"""j28 - does the GCN's advantage survive a null change?

j27 established that the released GCN is *confidently* right on real CDC code
(median P(unsafe) 0.109, no decision within 0.05 of threshold) and the released
GAT is confidently wrong (median 0.999). Neither is a threshold accident.

That is not yet an architectural claim. j23 showed that one architecture,
retrained under a mathematically-null change, spans the whole range. So the
question is whether the GCN's advantage is a property of the architecture or a
property of that particular checkpoint.

The test: train each architecture under both orderings of the readout. If GCN
lands low under both and GAT lands high under both, the architecture is doing
the work. If either flips, the released checkpoints are single draws and must be
reported as such.

Six trainings: {gat, gcn, sage} x {global_first, seed_first}.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jlib as J                                                   # noqa: E402
import readouts                                                    # noqa: E402
from readouts import ReadoutNet, predict_readout, train_readout     # noqa: E402
from j23_pooling_order import LegacyOrderNet                        # noqa: E402

ARCHS = ["gat", "gcn", "sage"]
POOL = {"global_first": ReadoutNet, "seed_first": LegacyOrderNet}
EPOCHS, SEED, THREADS = 40, 0, 1


def main():
    J.banner("j28  does the architecture effect survive a null change?")
    train, val = J.load("train"), J.load("val")
    test_id = J.load("test_id")
    y_id = J.y_of(test_id)
    conf = J.confirmation_set()
    wrapped = J.wrapped_set()
    y_w = J.y_of(wrapped)
    unit = np.array([s["design"] for s in conf])
    y_conf = np.zeros(len(conf), dtype=bool)

    rows = []
    for arch, pool in itertools.product(ARCHS, POOL):
        print(f"  training {arch:5s} / {pool} ...", flush=True)
        readouts.ReadoutNet = POOL[pool]
        m = train_readout(train, val, readout="mean_seed", arch=arch,
                          seed=SEED, epochs=EPOCHS, threads=THREADS)
        pc = np.array(predict_readout(m, conf))
        pw = np.array(predict_readout(m, wrapped))
        pi = np.array(predict_readout(m, test_id)) > 0.5
        ci = J.clustered_ci(pc > 0.5, y_conf, unit, metric="fpr")
        rows.append(dict(
            arch=arch, pooling=pool,
            test_id_f1=float(J.f1(pi, y_id)),
            conf_fpr=float((pc > 0.5).mean()),
            ci=[ci["ci_lo"], ci["ci_hi"]],
            conf_median_p=float(np.median(pc)),
            conf_min_margin=float(np.abs(pc - 0.5).min()),
            wrapped_recall=float((pw[y_w] > 0.5).mean())))
        r = rows[-1]
        print(f"           test_id {r['test_id_f1']:.4f}  "
              f"conf_FPR {r['conf_fpr']:.3f}  "
              f"median p {r['conf_median_p']:.4f}  "
              f"wrapped_recall {r['wrapped_recall']:.3f}", flush=True)
    readouts.ReadoutNet = ReadoutNet

    J.banner("Does the ordering flip the verdict?")
    print(f"  {'arch':>6s} {'global_first':>13s} {'seed_first':>11s} "
          f"{'swing':>7s}")
    summary = {}
    for arch in ARCHS:
        a = next(r for r in rows if r["arch"] == arch
                 and r["pooling"] == "global_first")["conf_fpr"]
        b = next(r for r in rows if r["arch"] == arch
                 and r["pooling"] == "seed_first")["conf_fpr"]
        summary[arch] = dict(global_first=a, seed_first=b, swing=abs(a - b))
        print(f"  {arch:>6s} {a:13.3f} {b:11.3f} {abs(a - b):7.3f}")

    swings = [v["swing"] for v in summary.values()]
    spread_within = max(swings)
    spread_across = (max(r["conf_fpr"] for r in rows)
                     - min(r["conf_fpr"] for r in rows))
    print(f"\n  largest swing from the null change alone : {spread_within:.3f}")
    print(f"  total spread across all six configs      : {spread_across:.3f}")
    verdict = ("architecture effect is not separable from the null change"
               if spread_within >= 0.5 * spread_across else
               "architecture effect survives the null change")
    print(f"  verdict: {verdict}")

    J.save("j28_arch_dispersion", dict(
        archs=ARCHS, epochs=EPOCHS, seed=SEED, rows=rows, summary=summary,
        max_null_swing=spread_within, total_spread=spread_across,
        verdict=verdict))


if __name__ == "__main__":
    main()
