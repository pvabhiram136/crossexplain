"""j32 - the Section VIII table: benchmark score beside real-code behaviour.

j21 reported confirmation false-positive rates. Section VIII needs those next to
what each method scores on the benchmark, because the point of the section is
that the first column cannot order the last one.

Released checkpoints and fitted baselines only. Nothing here is retrained, so
none of it is exposed to the dispersion of Sections IX and X.

Columns:
  test_id F1      the generated benchmark the conference paper reports
  test_real F1    the twelve hand-written designs (near-distribution)
  wrapped recall  authored hazards inside real third-party cores
  conf FPR        false positives on 31 crossings from 16 CDC libraries,
                  with design-clustered 95% intervals

Emitted as `table*` (full width). At seven columns it does not fit one column
of a two-column layout, and a tabular that overflows a float does NOT raise an
overfull-hbox warning -- it silently prints over the neighbouring text. That is
invisible to every checker and only shows up when the PDF is read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402
from strong_matcher import strong_predict                          # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import (pattern_matcher_predict,       # noqa: E402
                                    FlatML, flat_features)
from crossexplain.models import CrossExplainNet                    # noqa: E402
from crossexplain.train import predict                             # noqa: E402

SEC = JROOT / "paper" / "sections"
RES = J.ROOT / "results"
RATIO = 0.70          # domain-aware matcher threshold, fitted on train in j04


def load_ckpt(name):
    ck = torch.load(RES / name, map_location="cpu", weights_only=False)
    state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
    kw = dict(arch="gat", hidden=64, layers=5)
    if isinstance(ck, dict):
        for k in ("arch", "hidden", "layers"):
            if k in ck:
                kw[k] = ck[k]
    m = CrossExplainNet(**kw)
    m.load_state_dict(state)
    m.eval()
    return m


def main():
    J.banner("j32  transfer table for Section VIII")
    train = J.load("train")
    y_tr = J.y_of(train)
    splits = {"test_id": J.load("test_id"), "test_big": J.load("test_big"),
              "test_real": J.load("test_real"),
              "wrapped": J.wrapped_set(), "conf": J.confirmation_set()}
    for k, v in splits.items():
        print(f"  {k:10s} {len(v):4d} samples, "
              f"{len({s['design'] for s in v}):2d} designs")

    seeds = J.load_seed_models()
    flat = FlatML(seed=0).fit(train)
    from sklearn.linear_model import LogisticRegression            # noqa: E402
    from sklearn.preprocessing import StandardScaler               # noqa: E402
    from sklearn.pipeline import make_pipeline                     # noqa: E402
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000)).fit(
        np.stack([flat_features(s) for s in train]), y_tr.astype(int))

    def gnn(m):
        return lambda s: np.array(predict(m, s)[0]) > 0.5

    methods = {
        "CrossExplain (5-seed ens.)":
            lambda s: J.seed_probs(seeds, s).mean(0) > 0.5,
        "CrossExplain (released)": gnn(load_ckpt("crossexplain_gat.pt")),
        "GCN ablation": gnn(load_ckpt("ablation_gcn.pt")),
        "GraphSAGE ablation": gnn(load_ckpt("ablation_sage.pt")),
        "single-task ablation": gnn(load_ckpt("ablation_singletask.pt")),
        # FlatML.predict returns (detection, type); a 0 in the first list is
        # UNSAFE. Matches the convention used in j21.
        "Random Forest": lambda s: np.array([d == 0 for d in flat.predict(s)[0]]),
        "Logistic regression":
            lambda s: lr.predict(np.stack([flat_features(x) for x in s])) == 1,
        # Both matchers return 1 = SAFE, so UNSAFE is `p == 0`. Same
        # convention as j21; getting this backwards silently reports 1 - FPR.
        "matcher, textbook":
            lambda s: np.array([p == 0 for p in pattern_matcher_predict(s)]),
        "matcher, domain-aware":
            lambda s: np.array([p == 0 for p in
                                strong_predict(s, gray_xor_ratio=RATIO)]),
    }

    conf = splits["conf"]
    unit = np.array([s["design"] for s in conf])
    y_conf = np.zeros(len(conf), dtype=bool)
    y_w = J.y_of(splits["wrapped"])

    rows = []
    print(f"\n  {'method':>28s} {'gen F1':>7s} {'big F1':>7s} "
          f"{'real F1':>8s} {'wr.rec':>7s} {'FPR':>6s} {'95% CI':>16s}")
    for name, fn in methods.items():
        pc = fn(conf)
        ci = J.clustered_ci(pc, y_conf, unit, metric="fpr")
        r = dict(method=name,
                 test_id_f1=float(J.f1(fn(splits["test_id"]),
                                       J.y_of(splits["test_id"]))),
                 test_big_f1=float(J.f1(fn(splits["test_big"]),
                                        J.y_of(splits["test_big"]))),
                 test_real_f1=float(J.f1(fn(splits["test_real"]),
                                         J.y_of(splits["test_real"]))),
                 wrapped_recall=float(fn(splits["wrapped"])[y_w].mean()),
                 conf_fpr=float(pc.mean()),
                 conf_flagged=int(pc.sum()), conf_n=len(conf),
                 ci=[ci["ci_lo"], ci["ci_hi"]])
        rows.append(r)
        print(f"  {name:>28s} {r['test_id_f1']:7.4f} {r['test_big_f1']:7.4f} "
              f"{r['test_real_f1']:8.4f} "
              f"{r['wrapped_recall']:7.3f} {r['conf_fpr']:6.3f} "
              f"  [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]")

    gf = [r["test_id_f1"] for r in rows]
    cf = [r["conf_fpr"] for r in rows]
    order_gen = [r["method"] for r in sorted(rows, key=lambda r: -r["test_id_f1"])]
    order_real = [r["method"] for r in sorted(rows, key=lambda r: r["conf_fpr"])]
    from scipy.stats import spearmanr
    rho, p = spearmanr(gf, cf)

    J.banner("Does the benchmark order the methods?")
    print(f"  benchmark F1 spans {min(gf):.4f} to {max(gf):.4f} "
          f"(range {max(gf) - min(gf):.4f}); "
          f"{sum(1 for x in gf if x >= 0.9999)} of {len(gf)} are at 1.0000")
    print(f"  real FPR     spans {min(cf):.3f} to {max(cf):.3f} "
          f"(range {max(cf) - min(cf):.3f})")
    print(f"  Spearman rank correlation between them: "
          f"rho = {rho:+.3f}, p = {p:.3f}")
    print(f"\n  best on benchmark: {order_gen[0]}")
    print(f"  best on real code: {order_real[0]}")

    L = [
        "% GENERATED by journal/scripts/j32_transfer_table.py",
        r"\begin{table*}[t]",
        r"\caption{Released checkpoints and fitted baselines --- no retraining, "
        r"so none of this is exposed to the dispersion of "
        r"Section~\ref{sec:dispersion}. The benchmark column cannot order the "
        r"real-code column: "
        f"{sum(1 for x in gf if x >= 0.9999)} of {len(gf)} methods are tied at "
        r"$1.0000$ while their false-positive rates on correct third-party CDC "
        f"span {min(cf):.3f} to {max(cf):.3f}. Intervals are design-clustered "
        # Every other number in this caption is interpolated from the data;
        # this one was a literal, and stayed at 16 when the corpus grew to 20
        # designs. Derived now, for the same reason as all the others.
        f"over {len({s['design'] for s in conf})} designs.}}",
        r"\label{tab:transfer}",
        r"\centering\tabfont",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Method & Gen. $F_1$ & Big $F_1$ & Real $F_1$ & Wr. rec. & FPR & 95\% CI \\",
        r"\midrule",
    ]
    for r in rows:
        L.append(f"{r['method'].replace('&', r'\&')} & "
                 f"{r['test_id_f1']:.4f} & {r['test_big_f1']:.4f} & "
                 f"{r['test_real_f1']:.4f} & "
                 f"{r['wrapped_recall']:.3f} & {r['conf_fpr']:.3f} & "
                 f"$[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    SEC.mkdir(parents=True, exist_ok=True)
    (SEC / "tab_transfer.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {(SEC / 'tab_transfer.tex').relative_to(J.ROOT)}")

    J.save("j32_transfer_table", dict(
        rows=rows, spearman_rho=float(rho), spearman_p=float(p),
        n_at_ceiling=int(sum(1 for x in gf if x >= 0.9999)),
        n_at_ceiling_big=int(sum(1 for r in rows
                                 if r["test_big_f1"] >= 0.9999)),
        gen_range=float(max(gf) - min(gf)),
        real_range=float(max(cf) - min(cf)),
        best_on_benchmark=order_gen[0], best_on_real=order_real[0]))


if __name__ == "__main__":
    main()
