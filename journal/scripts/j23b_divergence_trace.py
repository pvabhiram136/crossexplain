"""j23b - where, exactly, the two pooling orders start to disagree.

j23 shows THAT the order matters. This shows WHEN. If the divergence were a
coding mistake it would appear at initialisation or in the forward pass; if it
is floating-point reassociation it will appear only after weights start moving,
at roughly machine epsilon, and compound.

Three checks, in increasing order of the training loop:

  1. initialisation   - same parameter names, same values?
  2. forward pass     - same outputs on one fixed batch?
  3. first step       - same loss, same gradient step?
  4. epoch trace      - parameter checksum of both, epoch by epoch

The legacy order is reconstructed here from j23 so this script does not depend
on the pre-fix state of readouts.py.
"""
import random
import sys
from pathlib import Path

import torch

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jlib as J                                                   # noqa: E402
from readouts import ReadoutNet                                    # noqa: E402
from j23_pooling_order import LegacyOrderNet                       # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.models import collate                            # noqa: E402
from crossexplain.train import (deterministic_run, multitask_loss,  # noqa: E402
                                quick_f1)

ARCH, EPOCHS = "gcn", 8
KW = dict(readout="mean_seed", arch=ARCH, hidden=64, layers=5)


def build(cls):
    return cls(**KW)


def main():
    J.banner("j23b  divergence trace")
    train, val = J.load("train"), J.load("val")

    # ---------------------------------------------------------- 1. init
    with deterministic_run(0, threads=1):
        a = build(ReadoutNet)
    with deterministic_run(0, threads=1):
        b = build(LegacyOrderNet)
    sa, sb = a.state_dict(), b.state_dict()
    print(f"  same parameter names : {set(sa) == set(sb)}")
    print(f"  max init difference  : "
          f"{max((sa[k] - sb[k]).abs().max().item() for k in sa):.3e}")

    # ------------------------------------------------------- 2. forward
    batch = collate([s for s in train if s["label_safe"] != -1][:32])
    a.eval(); b.eval()
    with torch.no_grad():
        oa = a(batch["x"], batch["ei"], batch["batch"], batch["seed_mask"],
               batch["n_graphs"])
        ob = b(batch["x"], batch["ei"], batch["batch"], batch["seed_mask"],
               batch["n_graphs"])
    for name, i in (("detect", 0), ("type", 1), ("localize", 2)):
        print(f"  forward diff {name:9s}: {(oa[i] - ob[i]).abs().max().item():.3e}")

    # ---------------------------------------------------- 3 & 4. training
    def run(cls):
        with deterministic_run(0, threads=1):
            rng = random.Random(0)
            model = build(cls)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3,
                                   weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
            data = [s for s in train if s["label_safe"] != -1]
            data.sort(key=lambda s: (s["design"], s["level"], s["cid"]))
            trace, first_loss = [], None
            for _ in range(EPOCHS):
                model.train()
                rng.shuffle(data)
                for i in range(0, len(data), 32):
                    bb = collate(data[i:i + 32])
                    out = model(bb["x"], bb["ei"], bb["batch"], bb["seed_mask"],
                                bb["n_graphs"])
                    loss, _ = multitask_loss(out, bb)
                    if first_loss is None:
                        first_loss = loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
                chk = sum(p.detach().abs().sum().item()
                          for p in model.parameters())
                trace.append((chk, quick_f1(model, val)))
            return first_loss, trace

    la, ta = run(ReadoutNet)
    lb, tb = run(LegacyOrderNet)
    print(f"\n  first-batch loss fixed  : {la:.9f}")
    print(f"  first-batch loss legacy : {lb:.9f}")
    print(f"  difference              : {abs(la - lb):.3e}")

    print(f"\n  {'ep':>3s} {'checksum fixed':>16s} {'checksum legacy':>17s} "
          f"{'|delta|':>11s} {'rel':>11s} {'valF1 f':>8s} {'valF1 l':>8s}")
    rows = []
    for ep, ((ca, fa), (cb, fb)) in enumerate(zip(ta, tb)):
        d = abs(ca - cb)
        rel = d / max(abs(ca), 1e-12)
        print(f"  {ep:3d} {ca:16.6f} {cb:17.6f} {d:11.3e} {rel:11.3e} "
              f"{fa:8.4f} {fb:8.4f}")
        rows.append(dict(epoch=ep, checksum_fixed=ca, checksum_legacy=cb,
                         abs_delta=d, rel_delta=rel, val_f1_fixed=fa,
                         val_f1_legacy=fb))

    J.save("j23b_divergence_trace", dict(
        arch=ARCH, epochs=EPOCHS,
        init_identical=bool(max((sa[k] - sb[k]).abs().max().item()
                                for k in sa) == 0.0),
        forward_max_diff=float(max((oa[i] - ob[i]).abs().max().item()
                                   for i in range(3))),
        first_loss_fixed=la, first_loss_legacy=lb, trace=rows))


if __name__ == "__main__":
    main()
