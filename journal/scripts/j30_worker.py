"""j30 worker - train ONE null-change configuration and write its result.

Invoked as:  python j30_worker.py <arch> <pooling> <perm>
  arch     gat | gcn | sage
  pooling  global_first | seed_first
  perm     none | <int>          node relabeling seed

One process per configuration, each pinned to a single thread, so the driver
can run several at once without any of them affecting another's arithmetic.
Every configuration is deterministic and reproducible on its own; re-running
this script with the same arguments gives the same model bit for bit.

Skips its own work if the output file already exists, so the driver is
resumable after an interruption.
"""
import json
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
from j24_dispersion import relabel_all                              # noqa: E402

POOL = {"global_first": ReadoutNet, "seed_first": LegacyOrderNet}
EPOCHS, SEED, THREADS = 40, 0, 1
OUTDIR = J.RESULTS / "j30"


def tag(arch, pooling, perm):
    return f"{arch}_{pooling}_perm{perm}"


def main(arch, pooling, perm_s):
    perm = None if perm_s == "none" else int(perm_s)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{tag(arch, pooling, perm_s)}.json"
    if out.exists():
        print(f"  skip {out.name} (exists)")
        return

    train, val = J.load("train"), J.load("val")
    test_id, test_real = J.load("test_id"), J.load("test_real")
    conf, wrapped = J.confirmation_set(), J.wrapped_set()
    y_w = J.y_of(wrapped)

    readouts.ReadoutNet = POOL[pooling]
    m = train_readout(relabel_all(train, perm), relabel_all(val, perm),
                      readout="mean_seed", arch=arch, seed=SEED,
                      epochs=EPOCHS, threads=THREADS)

    def p(samples):
        return np.array(predict_readout(m, relabel_all(samples, perm)))

    pc, pw = p(conf), p(wrapped)
    rec = dict(
        arch=arch, pooling=pooling, node_perm=perm,
        epochs=EPOCHS, seed=SEED, threads=THREADS,
        test_id_f1=float(J.f1(p(test_id) > 0.5, J.y_of(test_id))),
        test_real_f1=float(J.f1(p(test_real) > 0.5, J.y_of(test_real))),
        conf_fpr=float((pc > 0.5).mean()),
        conf_median_p=float(np.median(pc)),
        conf_min_margin=float(np.abs(pc - 0.5).min()),
        conf_n=len(conf),
        wrapped_recall=float((pw[y_w] > 0.5).mean()),
        wrapped_sep=float(np.median(pw[y_w]) - np.median(pw[~y_w])))
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"  {out.name}: test_id {rec['test_id_f1']:.4f}  "
          f"conf_FPR {rec['conf_fpr']:.3f}  "
          f"median p {rec['conf_median_p']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: j30_worker.py <arch> <pooling> <perm>")
    main(*sys.argv[1:])
