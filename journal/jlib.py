"""Shared helpers for the journal-extension experiments (phase P0).

Nothing in here mutates the conference artifact. Splits and checkpoints are
read from ``../data`` and ``../results``; every output goes to
``journal/results`` and ``journal/figures``.

The central correction this module implements: ``test_real`` contains 68
labeled SAMPLES that are only 17 distinct crossings from 12 designs, each
synthesized at four optimization levels. Resampling those 68 rows as if they
were independent understates uncertainty by roughly a factor of two, so every
interval here is clustered by an explicit unit (design by default).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crossexplain.models import CrossExplainNet          # noqa: E402
from crossexplain.train import predict                   # noqa: E402

JROOT = Path(__file__).resolve().parent
RESULTS = JROOT / "results"
FIGURES = JROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SPLITS = ["train", "val", "test_id", "test_big", "test_real"]


# --------------------------------------------------------------------- data
def load(split: str, labeled_only: bool = True):
    with open(ROOT / "data" / "dataset" / f"{split}.pkl", "rb") as f:
        s = pickle.load(f)
    return [x for x in s if x["label_safe"] != -1] if labeled_only else s


def confirmation_set():
    """The 39 real-CDC crossings, rebuilt from the O0 netlists of the found
    designs.

    Every crossing is labeled SAFE. That label is an assumption, not a
    measurement: it holds if OpenTitan, pulp-platform, alexforencich and ZipCPU
    implement their own synchronizers correctly. State the assumption wherever
    these numbers are reported.

    Four scripts (j17, j18, j19, j21) each carried a private copy of this
    builder. They are identical; this is the shared one.
    """
    import json
    from dataclasses import asdict

    sys.path.insert(0, str(JROOT))
    from grouping import group_crossings_structural
    from corpus_tools import parse_netlist_structural
    from crossexplain.netlist_graph import (find_capture_ffs, extract_subgraph,
                                            node_features, CrossingSample)

    triage = json.loads((RESULTS / "j12_corpus_triage.json").read_text())
    out = []
    for r in triage:
        if not r.get("usable") or "duplicate_of" in r:
            continue
        jf = JROOT / "corpus" / "netlists" / f"{r['name']}.O0.json"
        if not jf.exists():
            continue
        nl = parse_netlist_structural(jf)
        caps = find_capture_ffs(nl)
        for grp in (group_crossings_structural(nl, caps) if caps else []):
            src_doms = set()
            for f in grp:
                src_doms |= caps[f][0]
            dst = nl.domain[grp[0]]
            nodes, edges, local = extract_subgraph(nl, grp, cid=None)
            cone = set(grp)
            for f in grp:
                sf, cb = nl.d_srcs[f]
                cone |= sf | cb
            out.append(asdict(CrossingSample(
                design=r["name"], level="O0", cid=-1, nodes=nodes,
                node_feats=node_features(nl, nodes, set(grp), src_doms, dst,
                                         edges),
                edges=edges, seeds=[local[f] for f in grp],
                loc_mask=[1 if g in cone else 0 for g in nodes],
                label_safe=1, label_class=-1, template="FOUND_SAFE",
                src_dom=min(src_doms) if src_doms else -1, dst_dom=dst)))
    return out


def wrapped_set():
    """The semi-synthetic wrapped corpus: third-party cores with authored
    crossings, at all four optimization levels.

    Semi-synthetic. Always report these separately from the found designs -
    the cores are real, the crossings are ours.
    """
    import json
    from dataclasses import asdict

    sys.path.insert(0, str(JROOT))
    from wrappers import extract_wrapped

    wrap = JROOT / "corpus" / "wrapped"
    levels = ["O0", "O1", "O2", "O3"]
    out = []
    for gtf in sorted(wrap.glob("*.json")):
        if any(gtf.name.endswith(f".{l}.json") for l in levels):
            continue
        gt = json.loads(gtf.read_text())
        for lvl in levels:
            jf = wrap / f"{gtf.stem}.{lvl}.json"
            if not jf.exists():
                continue
            try:
                ss, _ = extract_wrapped(jf, gt, gtf.stem, lvl)
            except Exception:                                  # noqa: BLE001
                continue
            out.extend(asdict(s) for s in ss if s.label_safe != -1)
    return out


def units(samples):
    """Clustering keys for one split, as parallel arrays."""
    return dict(
        sample=np.arange(len(samples)).astype(str),
        crossing=np.array([f"{s['design']}#{s['cid']}" for s in samples]),
        design=np.array([s["design"] for s in samples]),
    )


def y_of(samples):
    return np.array([s["label_safe"] == 0 for s in samples])   # True = UNSAFE


def levels(samples):
    return np.array([s["level"] for s in samples])


def effective_n(samples):
    u = units(samples)
    return dict(samples=len(samples),
                crossings=len(np.unique(u["crossing"])),
                designs=len(np.unique(u["design"])))


# ------------------------------------------------------------------ models
def load_seed_models(path=None, arch="gat", hidden=64, layers=5):
    """The five trained seeds cached by scripts/05_experiments.py."""
    path = path or (ROOT / "results" / "seed_models.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = []
    for k in sorted(blob):
        e = blob[k]
        state = e["state"] if isinstance(e, dict) and "state" in e else e
        m = CrossExplainNet(
            arch=e.get("arch", arch) if isinstance(e, dict) else arch,
            hidden=e.get("hidden", hidden) if isinstance(e, dict) else hidden,
            layers=e.get("layers", layers) if isinstance(e, dict) else layers)
        m.load_state_dict(state)
        m.eval()
        out.append(m)
    return out


def seed_probs(models, samples):
    """[n_seeds, n_samples] calibrated unsafe-probabilities."""
    return np.array([predict(m, samples)[0] for m in models])


# ----------------------------------------------------------------- metrics
def _counts(pred, y):
    pred = np.asarray(pred, bool)
    y = np.asarray(y, bool)
    return ((pred & y).sum(), (pred & ~y).sum(),
            ((~pred) & ~y).sum(), ((~pred) & y).sum())


def f1(pred, y):
    tp, fp, _, fn = _counts(pred, y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def accuracy(pred, y):
    return float((np.asarray(pred, bool) == np.asarray(y, bool)).mean())


def fpr(pred, y):
    _, fp, tn, _ = _counts(pred, y)
    return fp / (fp + tn) if fp + tn else 0.0


def metrics(pred, y):
    tp, fp, tn, fn = _counts(pred, y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(accuracy=accuracy(pred, y), precision=float(p), recall=float(r),
                f1=f1(pred, y), false_positive_rate=float(fpr(pred, y)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


METRICS = {"f1": f1, "accuracy": accuracy, "fpr": fpr}


# --------------------------------------------------------------- bootstrap
def _resample_index(unit, rng):
    groups = np.unique(unit)
    idx = {g: np.where(unit == g)[0] for g in groups}
    pick = rng.choice(groups, len(groups), replace=True)
    return np.concatenate([idx[g] for g in pick])


def clustered_ci(pred, y, unit, metric="f1", B=5000, seed=0, alpha=0.05):
    """Percentile bootstrap CI, resampling whole clusters with replacement."""
    fn = METRICS[metric] if isinstance(metric, str) else metric
    rng = np.random.default_rng(seed)
    pred, y = np.asarray(pred), np.asarray(y)
    vals = np.empty(B)
    for b in range(B):
        ii = _resample_index(unit, rng)
        vals[b] = fn(pred[ii], y[ii])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(fn(pred, y)), ci_lo=float(lo), ci_hi=float(hi),
                width=float(hi - lo), n_clusters=int(len(np.unique(unit))), B=B)


def paired_clustered(pred_a, pred_b, y, unit, metric="f1", B=10000, seed=0):
    """Bootstrap the DIFFERENCE a - b on the same resampled clusters.

    This is the correct comparison. Two overlapping marginal intervals do not
    constitute a test, because the two methods are evaluated on the very same
    designs and their errors are correlated.
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    rng = np.random.default_rng(seed)
    a, b, y = np.asarray(pred_a), np.asarray(pred_b), np.asarray(y)
    d = np.empty(B)
    for i in range(B):
        ii = _resample_index(unit, rng)
        d[i] = fn(a[ii], y[ii]) - fn(b[ii], y[ii])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(delta=float(fn(a, y) - fn(b, y)), boot_mean=float(d.mean()),
                ci_lo=float(lo), ci_hi=float(hi), p_value=float(min(p, 1.0)),
                significant=bool(lo > 0 or hi < 0),
                n_clusters=int(len(np.unique(unit))), B=B)


# ------------------------------------------------------------- calibration
def ece(prob_unsafe, y, bins=10):
    """Expected calibration error of the binary decision.

    Confidence is max(p, 1-p); a bin's gap is |empirical accuracy - mean
    confidence|, weighted by bin occupancy.
    """
    p = np.asarray(prob_unsafe, float)
    y = np.asarray(y, bool)
    conf = np.where(p > 0.5, p, 1 - p)
    correct = (p > 0.5) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability(prob_unsafe, y, bins=10):
    """Per-bin (mean confidence, empirical accuracy, count) for the diagram."""
    p = np.asarray(prob_unsafe, float)
    y = np.asarray(y, bool)
    conf = np.where(p > 0.5, p, 1 - p)
    correct = (p > 0.5) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), count=int(m.sum()),
                         mean_conf=float(conf[m].mean()) if m.sum() else None,
                         accuracy=float(correct[m].mean()) if m.sum() else None))
    return rows


def brier(prob_unsafe, y):
    return float(np.mean((np.asarray(prob_unsafe, float) - np.asarray(y, float)) ** 2))


# ------------------------------------------------------------------ output
def save(name: str, obj):
    import json
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")
