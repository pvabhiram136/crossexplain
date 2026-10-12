"""P0.3 - calibration study.

The conference paper lists "an attention-based explanation with a calibrated
confidence" as contribution (iv), but never measures calibration. This script
measures it, and diagnoses why it fails out of distribution.

The mechanism under test: crossexplain/train.py::calibrate_temperature picks T
by grid search minimizing NLL on the GENERATED validation split. That split is
saturated (F1 ~ 0.995, and every learned model reaches 1.000 on test_id), so
the search has no incentive to soften anything and can even sharpen. We check
this by refitting T on the real split as an oracle and comparing.

Run:  python journal/scripts/j02_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

OUT = {}
models = J.load_seed_models()
SPLITS = ["val", "test_id", "test_big", "test_real"]

data = {sp: J.load(sp) for sp in SPLITS}
probs = {sp: J.seed_probs(models, data[sp]) for sp in SPLITS}
ys = {sp: J.y_of(data[sp]) for sp in SPLITS}

# ------------------------------------------------------------- 1. the table
J.banner("P0.3a  Calibration of the detection probability")
print(f"  {'split':10s} {'n':>5s} {'ECE(ens)':>9s} {'Brier':>7s} "
      f"{'mean conf':>10s} {'accuracy':>9s} {'gap':>7s}")
tab = {}
for sp in SPLITS:
    ens = probs[sp].mean(0)
    y = ys[sp]
    conf = np.where(ens > 0.5, ens, 1 - ens)
    acc = float(((ens > 0.5) == y).mean())
    tab[sp] = dict(
        n=len(y), ece=J.ece(ens, y), brier=J.brier(ens, y),
        mean_confidence=float(conf.mean()), accuracy=acc,
        overconfidence_gap=float(conf.mean() - acc),
        per_seed_ece=[J.ece(probs[sp][i], y) for i in range(len(models))],
        per_seed_accuracy=[float(((probs[sp][i] > 0.5) == y).mean())
                           for i in range(len(models))],
        reliability=J.reliability(ens, y),
    )
    t = tab[sp]
    print(f"  {sp:10s} {t['n']:5d} {t['ece']:9.4f} {t['brier']:7.4f} "
          f"{t['mean_confidence']:10.3f} {t['accuracy']:9.3f} "
          f"{t['overconfidence_gap']:+7.3f}")
OUT["calibration"] = tab

print("\n  per-seed ECE on test_real: " + "  ".join(
    f"s{i}={v:.3f}" for i, v in enumerate(tab["test_real"]["per_seed_ece"])))
print("  ensembling changes ECE on test_real from mean "
      f"{np.mean(tab['test_real']['per_seed_ece']):.3f} (single seed) "
      f"to {tab['test_real']['ece']:.3f} (5-seed average)")

# ------------------------------------------------- 2. what T was picked, why
J.banner("P0.3b  Why temperature scaling failed: the fitting set is saturated")
temps = [float(m.temperature) for m in models]
print(f"  temperatures selected on generated val: {temps}")
print("  (T < 1 SHARPENS the distribution. Two seeds selected T = 0.5.)")

val_ens = probs["val"].mean(0)
print(f"  generated val: F1={J.f1(val_ens > 0.5, ys['val']):.4f}  "
      f"ECE={J.ece(val_ens, ys['val']):.4f}  "
      f"-> nothing for the NLL search to correct")


def refit_T(logit_probs, y, grid=None):
    """Grid-search T on a given set, operating on recovered logits."""
    p = np.clip(np.asarray(logit_probs, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    y = np.asarray(y, float)
    best, best_nll = 1.0, np.inf
    for t in (grid or np.arange(0.25, 6.01, 0.05)):
        q = 1 / (1 + np.exp(-z / t))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if nll < best_nll:
            best_nll, best = nll, float(t)
    return best, float(best_nll)


oracle = {}
for sp in SPLITS:
    ens = probs[sp].mean(0)
    t_star, nll = refit_T(ens, ys[sp])
    z = np.log(np.clip(ens, 1e-6, 1 - 1e-6) / (1 - np.clip(ens, 1e-6, 1 - 1e-6)))
    tempered = 1 / (1 + np.exp(-z / t_star))
    oracle[sp] = dict(T_oracle=t_star, ece_before=J.ece(ens, ys[sp]),
                      ece_after=J.ece(tempered, ys[sp]),
                      f1_before=J.f1(ens > 0.5, ys[sp]),
                      f1_after=J.f1(tempered > 0.5, ys[sp]))
    o = oracle[sp]
    print(f"  {sp:10s} oracle T={o['T_oracle']:.2f}  "
          f"ECE {o['ece_before']:.3f} -> {o['ece_after']:.3f}  "
          f"(F1 unchanged at {o['f1_after']:.3f})")
OUT["oracle_temperature"] = oracle
OUT["fitted_temperatures"] = temps
print("\n  Reading: the real split wants a much larger T than the generated")
print("  validation split provides. The head is not miscalibrated by design,")
print("  it is calibrated on the wrong distribution.")

# --------------------------------------- 3. is confidence useful at all OOD?
J.banner("P0.3c  Is the confidence still useful as an error detector on real?")
ens = probs["test_real"].mean(0)
y = ys["test_real"]
conf = np.where(ens > 0.5, ens, 1 - ens)
correct = (ens > 0.5) == y
# AUC of confidence ranking correct vs incorrect predictions
pos, neg = conf[correct], conf[~correct]
auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) \
    if len(pos) and len(neg) else float("nan")
print(f"  confidence-vs-correctness AUC = {auc:.3f}  "
      f"(0.5 = confidence carries no information about correctness)")
print(f"  mean confidence when right = {pos.mean():.3f}, when wrong = {neg.mean():.3f}")

# selective prediction: accuracy if we only answer above a confidence threshold
sel = []
for thr in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    m = conf >= thr
    sel.append(dict(threshold=thr, coverage=float(m.mean()),
                    accuracy=float(correct[m].mean()) if m.sum() else None,
                    n=int(m.sum())))
    print(f"    thr={thr:.2f}  coverage={m.mean():.2f}  "
          f"accuracy={(correct[m].mean() if m.sum() else float('nan')):.3f}  n={m.sum()}")
OUT["error_detection"] = dict(auc=auc, mean_conf_correct=float(pos.mean()),
                              mean_conf_wrong=float(neg.mean()),
                              selective_prediction=sel)

# ------------------------------------------------------------- 4. the figure
J.banner("P0.3d  Reliability diagrams")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

fig, axes = plt.subplots(1, len(SPLITS), figsize=(3.1 * len(SPLITS), 3.2),
                         sharey=True)
for ax, sp in zip(axes, SPLITS):
    rows = [r for r in tab[sp]["reliability"] if r["count"] > 0]
    xs = [r["mean_conf"] for r in rows]
    ys_ = [r["accuracy"] for r in rows]
    ns = [r["count"] for r in rows]
    ax.plot([0.5, 1], [0.5, 1], "--", color="0.6", lw=1, label="perfect")
    ax.plot(xs, ys_, "o-", color="#1f4e79", lw=1.6,
            ms=[max(3, min(11, 3 + 8 * n / max(ns))) for n in ns][0])
    ax.scatter(xs, ys_, s=[18 + 90 * n / max(ns) for n in ns],
               color="#1f4e79", zorder=3)
    ax.set_title(f"{sp}\nECE={tab[sp]['ece']:.3f}", fontsize=9)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("confidence", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("empirical accuracy", fontsize=8)
fig.suptitle("Detection confidence is calibrated in distribution and "
             "badly miscalibrated on real designs", fontsize=9.5)
fig.tight_layout()
p = J.FIGURES / "fig_reliability.png"
fig.savefig(p, dpi=200)
print(f"  -> {p.relative_to(J.ROOT)}")

J.save("j02_calibration", OUT)
