"""j27 - is the released GCN's transfer real, or is it sitting on the threshold?

The GCN ablation raises false positives on 1 of 31 third-party crossings while
the released GAT raises 30. Before treating that as an architectural finding we
need to know whether the GCN is *confidently* calling them safe or merely
landing on the safe side of 0.5 by a hair.

Released checkpoints only. No training, so nothing here is exposed to the
dispersion of j23/j24.

Reported per model, on the 31 confirmation crossings:
  median / IQR of P(unsafe)      where the decisions actually sit
  margin to threshold            |p - 0.5|, the quantity that would flip
  n within 0.05 of 0.5           how many decisions are one rounding away
and, as a control, the same on the wrapped corpus where the true labels are
mixed, so a model that is simply biased toward SAFE is visible.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.models import CrossExplainNet                    # noqa: E402
from crossexplain.train import predict                             # noqa: E402

CKPT = {
    "crossexplain_gat": ("crossexplain_gat.pt", "gat"),
    "ablation_gcn": ("ablation_gcn.pt", "gcn"),
    "ablation_sage": ("ablation_sage.pt", "sage"),
    "ablation_singletask": ("ablation_singletask.pt", "gat"),
}


def load(fname, arch):
    blob = torch.load(J.ROOT / "results" / fname, map_location="cpu",
                      weights_only=False)
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    kw = dict(arch=arch, hidden=64, layers=5)
    if isinstance(blob, dict):
        for k in ("arch", "hidden", "layers"):
            if k in blob:
                kw[k] = blob[k]
    m = CrossExplainNet(**kw)
    m.load_state_dict(state)
    m.eval()
    return m


def describe(p):
    p = np.asarray(p)
    margin = np.abs(p - 0.5)
    return dict(
        median=float(np.median(p)),
        q25=float(np.percentile(p, 25)), q75=float(np.percentile(p, 75)),
        min=float(p.min()), max=float(p.max()),
        fpr=float((p > 0.5).mean()),
        median_margin=float(np.median(margin)),
        min_margin=float(margin.min()),
        n_near_threshold=int((margin < 0.05).sum()),
        n=len(p))


def main():
    J.banner("j27  why does the released GCN transfer?")
    conf = J.confirmation_set()
    wrapped = J.wrapped_set()
    y_w = J.y_of(wrapped)
    print(f"  conf    {len(conf)} crossings, all SAFE")
    print(f"  wrapped {len(wrapped)} samples, {int(y_w.sum())} unsafe\n")

    out = {}
    print(f"  {'model':>20s} {'median p':>9s} {'IQR':>17s} {'FPR':>6s} "
          f"{'min margin':>11s} {'near 0.5':>9s}")
    for name, (fname, arch) in CKPT.items():
        m = load(fname, arch)
        pc = np.array(predict(m, conf)[0])
        pw = np.array(predict(m, wrapped)[0])
        d = describe(pc)
        d["wrapped_recall"] = float((pw[y_w] > 0.5).mean())
        d["wrapped_median_unsafe"] = float(np.median(pw[y_w]))
        d["wrapped_median_safe"] = float(np.median(pw[~y_w]))
        d["separation"] = d["wrapped_median_unsafe"] - d["wrapped_median_safe"]
        out[name] = d
        print(f"  {name:>20s} {d['median']:9.4f} "
              f"[{d['q25']:.4f}, {d['q75']:.4f}] {d['fpr']:6.3f} "
              f"{d['min_margin']:11.4f} {d['n_near_threshold']:9d}")

    J.banner("Control: does it still separate hazards in real cores?")
    print(f"  {'model':>20s} {'med p unsafe':>13s} {'med p safe':>11s} "
          f"{'separation':>11s} {'recall':>7s}")
    for name, d in out.items():
        print(f"  {name:>20s} {d['wrapped_median_unsafe']:13.4f} "
              f"{d['wrapped_median_safe']:11.4f} {d['separation']:11.4f} "
              f"{d['wrapped_recall']:7.3f}")

    g = out["ablation_gcn"]
    verdict = ("confident" if g["min_margin"] > 0.10 else
               "marginal" if g["min_margin"] > 0.02 else "on the threshold")
    print(f"\n  GCN verdict: {verdict} "
          f"(closest decision is {g['min_margin']:.4f} from 0.5, "
          f"{g['n_near_threshold']} of {g['n']} within 0.05)")
    out["_verdict"] = verdict
    J.save("j27_why_gcn", out)


if __name__ == "__main__":
    main()
