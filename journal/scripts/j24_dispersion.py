"""j24 - how far apart can two models be while the benchmark calls them equal?

j23 found one pair. This measures the whole distribution.

Every model here is trained from the same initialisation, on the same data, in
the same order, with the same hyperparameters and the same seed. The
configurations differ ONLY along axes that are mathematically no-ops:

  pooling order   which of two disjoint index_add_ reductions runs first
  node labeling   the graph relabeled by a permutation; the encoder is
                  permutation-equivariant, so outputs are unchanged in exact
                  arithmetic
  thread count    how a reduction is split across cores

In exact arithmetic all of these models ARE the same model. In float32 they
are not, and the question is how much of that reaches behaviour.

Reported per configuration:
  test_id F1        the benchmark the conference paper reports
  test_real F1      the twelve hand-written designs
  wrapped recall    fraction of authored hazards caught in real cores
  conf FPR          false positives on 31 crossings from 16 real CDC libraries

The claim the paper needs is the gap between column one and column four.
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

ARCH, EPOCHS, SEED = "gcn", 40, 0
POOL = {"global_first": ReadoutNet, "seed_first": LegacyOrderNet}
PERMS = [None, 1, 2]
THREADS = [1, 4]


def relabel(sample, rng):
    """Relabel a crossing's nodes by a random permutation.

    A graph isomorphism: same nodes, same edges, same features, same seeds,
    different integer names. Message passing over sum/mean aggregation is
    permutation-equivariant, so every output is unchanged in exact arithmetic.
    """
    n = len(sample["node_feats"])
    p = rng.permutation(n)                    # p[old] = new
    inv = np.empty(n, dtype=int)
    inv[p] = np.arange(n)                     # inv[new] = old
    s = dict(sample)
    s["node_feats"] = [sample["node_feats"][inv[i]] for i in range(n)]
    s["loc_mask"] = [sample["loc_mask"][inv[i]] for i in range(n)]
    s["nodes"] = [sample["nodes"][inv[i]] for i in range(n)]
    s["edges"] = [(int(p[u]), int(p[v])) for u, v in sample["edges"]]
    s["seeds"] = [int(p[i]) for i in sample["seeds"]]
    return s


def relabel_all(samples, seed):
    if seed is None:
        return samples
    rng = np.random.default_rng(seed)
    return [relabel(s, rng) for s in samples]


def main():
    J.banner("j24  dispersion under mathematically-null changes")
    train, val = J.load("train"), J.load("val")
    evals = {
        "test_id": (J.load("test_id"), "f1"),
        "test_real": (J.load("test_real"), "f1"),
        "wrapped": (J.wrapped_set(), "recall"),
        "conf": (J.confirmation_set(), "fpr"),
    }
    for k, (s, _) in evals.items():
        print(f"  {k:10s} {len(s):4d} samples, "
              f"{len({x['design'] for x in s}):2d} designs")

    grid = list(itertools.product(POOL, PERMS, THREADS))
    print(f"\n  {len(grid)} configurations, all mathematically identical\n")

    rows = []
    for i, (pool, perm, threads) in enumerate(grid, 1):
        tag = f"{pool}/perm={perm}/threads={threads}"
        print(f"  [{i:2d}/{len(grid)}] {tag} ...", flush=True)
        readouts.ReadoutNet = POOL[pool]
        m = train_readout(relabel_all(train, perm), relabel_all(val, perm),
                          readout="mean_seed", arch=ARCH, seed=SEED,
                          epochs=EPOCHS, threads=threads)
        row = dict(pooling=pool, node_perm=perm, threads=threads)
        for name, (samples, metric) in evals.items():
            pred = np.array(predict_readout(
                m, relabel_all(samples, perm))) > 0.5
            if metric == "f1":
                row[name] = float(J.f1(pred, J.y_of(samples)))
            elif metric == "recall":
                y = J.y_of(samples)
                row[name] = float(pred[y].mean())
            else:
                row[name] = float(pred.mean())
        rows.append(row)
        print(f"           test_id {row['test_id']:.4f}  "
              f"test_real {row['test_real']:.4f}  "
              f"wrapped_recall {row['wrapped']:.4f}  "
              f"conf_FPR {row['conf']:.3f}", flush=True)
    readouts.ReadoutNet = ReadoutNet

    J.banner("Spread across configurations")
    print(f"  {'metric':>16s} {'min':>7s} {'max':>7s} {'range':>7s} "
          f"{'mean':>7s} {'sd':>7s} {'distinct':>9s}")
    summary = {}
    for name in evals:
        v = np.array([r[name] for r in rows])
        summary[name] = dict(min=float(v.min()), max=float(v.max()),
                             range=float(v.max() - v.min()),
                             mean=float(v.mean()), sd=float(v.std(ddof=1)),
                             n_distinct=int(len(np.unique(np.round(v, 6)))))
        print(f"  {name:>16s} {v.min():7.4f} {v.max():7.4f} "
              f"{v.max() - v.min():7.4f} {v.mean():7.4f} "
              f"{v.std(ddof=1):7.4f} {summary[name]['n_distinct']:9d}")

    print(f"\n  benchmark range  {summary['test_id']['range']:.4f}")
    print(f"  real-CDC range   {summary['conf']['range']:.4f}")
    J.save("j24_dispersion", dict(arch=ARCH, epochs=EPOCHS, seed=SEED,
                                  n_configs=len(grid), rows=rows,
                                  summary=summary))


if __name__ == "__main__":
    main()
