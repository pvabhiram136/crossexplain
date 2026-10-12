"""Evaluation: detection metrics, hazard-type confusion, localization IoU,
explanation fidelity/sparsity, and runtime."""
from __future__ import annotations

import time

import numpy as np
import torch

from .rtl_generator import HAZARD_CLASSES
from .models import collate
from .train import predict


def detection_metrics(pred_unsafe, samples):
    """pred_unsafe: list of 0/1 (1 = predicted unsafe)."""
    tp = fp = tn = fn = 0
    for p, s in zip(pred_unsafe, samples):
        is_unsafe = s["label_safe"] == 0
        if p and is_unsafe:
            tp += 1
        elif p and not is_unsafe:
            fp += 1
        elif not p and is_unsafe:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, len(samples))
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                false_positive_rate=fpr, tp=tp, fp=fp, tn=tn, fn=fn)


def type_confusion(type_pred, samples):
    k = len(HAZARD_CLASSES)
    cm = np.zeros((k, k), dtype=int)
    correct = total = 0
    for p, s in zip(type_pred, samples):
        if s["label_safe"] == 0 and s["label_class"] >= 0:
            cm[s["label_class"], p] += 1
            correct += int(p == s["label_class"])
            total += 1
    return cm, (correct / total if total else 0.0)


def localization_iou(loc_logits, samples, mode="topk"):
    """IoU between predicted important nodes and the ground-truth cone.
    mode 'topk': take the |cone| highest-scoring nodes (size-matched);
    mode 'thresh': sigmoid > 0.5."""
    ious = []
    for logits, s in zip(loc_logits, samples):
        truth = np.asarray(s["loc_mask"], dtype=bool)
        if truth.sum() == 0:
            continue
        scores = np.asarray(logits)
        if mode == "topk":
            k = int(truth.sum())
            pred = np.zeros_like(truth)
            pred[np.argsort(-scores)[:k]] = True
        else:
            pred = 1 / (1 + np.exp(-scores)) > 0.5
        inter = (pred & truth).sum()
        union = (pred | truth).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def explanation_fidelity(model, samples, top_frac=0.25, mode="edge"):
    """Fidelity+: removing the top-attended evidence should damp/flip the
    UNSAFE verdict. Two intervention protocols:

    mode='feature'  zero the features of the top-attended nodes (weak — the
                    graph structure still carries the signal)
    mode='edge'     delete every edge incident to the top-attended non-seed
                    nodes (structural removal — the correct protocol)

    Sparsity: fraction of nodes used as the explanation."""
    model.eval()
    unsafe = [s for s in samples if s["label_safe"] == 0]
    if not unsafe:
        return dict(fidelity=0.0, sparsity=0.0, mode=mode)
    flips, sparsities = [], []
    for s in unsafe:
        b = collate([s])
        det0, _, loc = model(b["x"], b["ei"], b["batch"], b["seed_mask"], 1)
        p0 = torch.sigmoid(det0)[0].item()
        n = len(s["node_feats"])
        k = max(1, int(top_frac * n))
        seeds = set(s["seeds"])
        top = [t for t in torch.argsort(loc, descending=True).tolist()
               if t not in seeds][:max(1, k - 1)]
        if mode == "feature":
            x2 = b["x"].clone()
            x2[top] = 0.0
            det1, _, _ = model(x2, b["ei"], b["batch"], b["seed_mask"], 1)
        else:  # edge removal
            drop = set(top)
            ei = b["ei"]
            keep = [i for i in range(ei.size(1))
                    if ei[0, i].item() not in drop and ei[1, i].item() not in drop]
            ei2 = ei[:, keep] if keep else torch.zeros(2, 0, dtype=torch.long)
            det1, _, _ = model(b["x"], ei2, b["batch"], b["seed_mask"], 1)
        p1 = torch.sigmoid(det1)[0].item()
        flips.append(p0 - p1)          # drop in unsafe-probability
        sparsities.append(len(top) / n)
    return dict(fidelity=float(np.mean(flips)),
                sparsity=float(np.mean(sparsities)), mode=mode)


def runtime_stats(model, samples, reps=3):
    t0 = time.time()
    for _ in range(reps):
        predict(model, samples)
    dt = (time.time() - t0) / reps
    n_params = sum(p.numel() for p in model.parameters())
    return dict(sec_per_1k_crossings=1000 * dt / max(1, len(samples)),
                n_params=n_params)


def evaluate_model(model, samples, name=""):
    labeled = [s for s in samples if s["label_safe"] != -1]
    det_p, type_pred, loc_logits = predict(model, labeled)
    det = detection_metrics([p > 0.5 for p in det_p], labeled)
    cm, type_acc = type_confusion(type_pred, labeled)
    iou = localization_iou(loc_logits, labeled)
    return dict(name=name, n=len(labeled), detection=det,
                type_accuracy=type_acc, confusion=cm.tolist(),
                localization_iou=iou)
