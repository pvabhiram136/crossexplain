"""j23 - the pooling-order result, made reproducible.

`CrossExplainNet.forward` accumulates the global mean-pool BEFORE the seed
pool. An earlier version of `journal/readouts.py` accumulated them in the
opposite order. The two are mathematically identical: both are sums over
disjoint index sets, and neither depends on the other.

They are not behaviourally identical. This script trains three models that
differ in nothing else and reports where they land.

  reference   crossexplain.train.train_model          (global pool first)
  fixed       journal readouts, current               (global pool first)
  legacy      journal readouts, pre-fix order         (seed pool first)

Expected: `fixed` reproduces `reference` exactly, `legacy` does not, and all
three score identically on the generated benchmark.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                  # noqa: E402
import readouts                                                   # noqa: E402
from readouts import ReadoutNet, predict_readout, train_readout    # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.train import train_model, predict                # noqa: E402

ARCH, EPOCHS, SEED = "gcn", 40, 0


class LegacyOrderNet(ReadoutNet):
    """The pre-fix order: seed pool accumulated first. Kept only so the
    finding stays reproducible. Never use this to train a reported model."""

    def forward(self, x, ei, batch, seed_mask, n_graphs):
        h = F.relu(self.embed(x))
        for layer in self.layers:
            h = layer(h, ei)

        seed_h = h * seed_mask.unsqueeze(-1)
        seed_cnt = torch.zeros(n_graphs, 1, device=h.device)
        seed_cnt.index_add_(0, batch, seed_mask.unsqueeze(-1).float())
        seed_pool = torch.zeros(n_graphs, h.size(1), device=h.device)
        seed_pool.index_add_(0, batch, seed_h)
        seed_pool = seed_pool / seed_cnt.clamp(min=1.0)

        glob = self._global_half(h, x, batch, n_graphs)
        if glob is None:
            glob = seed_pool
        g = torch.cat([glob, seed_pool], dim=1)
        return (self.detect_head(g).squeeze(-1), self.type_head(g),
                self.loc_head(h).squeeze(-1))


def main():
    J.banner("j23  pooling order")
    train, val = J.load("train"), J.load("val")
    test_id = J.load("test_id")
    y_id = J.y_of(test_id)
    conf = J.confirmation_set()
    print(f"  confirmation: {len(conf)} crossings, "
          f"{len({s['design'] for s in conf})} designs, all labeled SAFE")

    rows = {}

    print("  training reference (crossexplain.train.train_model) ...")
    m = train_model(train, val, arch=ARCH, epochs=EPOCHS, verbose=False)
    rows["reference"] = (np.array(predict(m, conf)[0]) > 0.5,
                         np.array(predict(m, test_id)[0]) > 0.5)

    for tag, cls in (("fixed", ReadoutNet), ("legacy", LegacyOrderNet)):
        print(f"  training {tag} ...")
        readouts.ReadoutNet = cls
        mm = train_readout(train, val, readout="mean_seed", arch=ARCH,
                           seed=SEED, epochs=EPOCHS)
        rows[tag] = (np.array(predict_readout(mm, conf)) > 0.5,
                     np.array(predict_readout(mm, test_id)) > 0.5)
    readouts.ReadoutNet = ReadoutNet

    unit = np.array([s["design"] for s in conf])
    y_conf = np.zeros(len(conf), dtype=bool)          # all SAFE
    out = {"arch": ARCH, "epochs": EPOCHS, "seed": SEED,
           "n_conf": len(conf), "n_conf_designs": int(len(set(unit))),
           "models": {}}

    print(f"\n  {'model':>10s} {'test_id F1':>11s} {'conf FPR':>9s} "
          f"{'95% CI':>16s}  {'agree w/ reference':>19s}")
    for tag, (pc, pi) in rows.items():
        f1 = J.f1(pi, y_id)
        fpr = float(pc.mean())
        ci = J.clustered_ci(pc, y_conf, unit, metric="fpr")
        agree = float((pc == rows["reference"][0]).mean())
        print(f"  {tag:>10s} {f1:11.4f} {fpr:9.3f} "
              f"  [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}] {agree:18.1%}")
        out["models"][tag] = dict(test_id_f1=float(f1), conf_fpr=fpr,
                                  ci=[ci["ci_lo"], ci["ci_hi"]],
                                  agree_with_reference=agree)

    ok = out["models"]["fixed"]["agree_with_reference"] == 1.0
    out["fix_verified"] = ok
    print(f"\n  fix reproduces reference exactly: {ok}")
    if not ok:
        print("  WARNING: readouts.py no longer matches the shipped model.")
    J.save("j23_pooling_order", out)


if __name__ == "__main__":
    main()
