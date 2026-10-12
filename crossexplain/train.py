"""Training loop for CrossExplain (multi-task loss, CPU-friendly).

Reproducibility note
--------------------
The message-passing aggregation uses ``Tensor.index_add_``. On a
multi-threaded CPU backend that operation accumulates contributions in a
non-deterministic order, so identical seeds produced materially different
models (measured: |ΔF1| up to 0.20 on the real-design split). Training
therefore pins the thread count and enables PyTorch's deterministic
algorithms for the duration of a run; see ``deterministic_run``.
"""
from __future__ import annotations

import contextlib
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .models import CrossExplainNet, collate


@contextlib.contextmanager
def deterministic_run(seed: int, threads: int = 1):
    """Make a training run bit-reproducible, then restore global state."""
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(threads)          # removes parallel-reduction races
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # numpy optional at this layer
        pass
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det, warn_only=True)
        if prev_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas


def multitask_loss(out, y, w_det=1.0, w_type=0.7, w_loc=0.7):
    det_logit, type_logit, loc_logit = out
    l_det = F.binary_cross_entropy_with_logits(det_logit, y["det_y"])
    # hazard type: only unsafe, labeled samples (ignore_index -100)
    l_type = F.cross_entropy(type_logit, y["type_y"], ignore_index=-100)
    if torch.isnan(l_type):
        l_type = torch.zeros((), device=det_logit.device)
    l_loc = F.binary_cross_entropy_with_logits(loc_logit, y["loc_y"])
    return w_det * l_det + w_type * l_type + w_loc * l_loc, \
        (l_det.item(), l_type.detach().item(), l_loc.item())


def train_model(train_samples, val_samples, arch="gat", hidden=64, layers=5,
                epochs=30, batch_size=32, lr=2e-3, seed=0, verbose=True,
                ckpt: Path | None = None, threads: int = 1):
    # 5 message-passing layers: the qualifier chain of a mux-recirculation
    # crossing is ~5 hops from the capture FF — a 3-layer encoder cannot
    # even see the evidence that separates MUX_HOLD from MUX_UNSYNC_QUAL.
    with deterministic_run(seed, threads=threads):
        rng = random.Random(seed)
        model = CrossExplainNet(arch=arch, hidden=hidden, layers=layers)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # sort first so the shuffle starts from a canonical order regardless
        # of how the caller assembled the list
        data = [s for s in train_samples if s["label_safe"] != -1]
        data.sort(key=lambda s: (s["design"], s["level"], s["cid"]))
        best_f1, best_state = -1.0, None
        for ep in range(epochs):
            model.train()
            rng.shuffle(data)
            t0, tot = time.time(), 0.0
            for i in range(0, len(data), batch_size):
                b = collate(data[i:i + batch_size])
                out = model(b["x"], b["ei"], b["batch"], b["seed_mask"],
                            b["n_graphs"])
                loss, _ = multitask_loss(out, b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            sched.step()
            f1 = quick_f1(model, val_samples)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep{ep:02d} "
                      f"loss={tot / max(1, len(data) // batch_size):.4f} "
                      f"valF1={f1:.4f} best={best_f1:.4f} "
                      f"({time.time() - t0:.1f}s)")
        if best_state:
            model.load_state_dict(best_state)
        calibrate_temperature(model, val_samples)
    if ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "arch": arch,
                    "hidden": hidden, "layers": layers}, ckpt)
    return model


@torch.no_grad()
def predict(model, samples, batch_size=64):
    model.eval()
    det_p, type_pred, loc_logits = [], [], []
    for i in range(0, len(samples), batch_size):
        b = collate(samples[i:i + batch_size])
        det_logit, type_logit, loc_logit = model(
            b["x"], b["ei"], b["batch"], b["seed_mask"], b["n_graphs"])
        det_p.extend(torch.sigmoid(det_logit / model.temperature).tolist())
        type_pred.extend(type_logit.argmax(1).tolist())
        # split node logits back per sample
        off = 0
        for s in samples[i:i + batch_size]:
            n = len(s["node_feats"])
            loc_logits.append(loc_logit[off:off + n].tolist())
            off += n
    return det_p, type_pred, loc_logits


@torch.no_grad()
def quick_f1(model, samples):
    labeled = [s for s in samples if s["label_safe"] != -1]
    if not labeled:
        return 0.0
    det_p, _, _ = predict(model, labeled)
    tp = fp = fn = 0
    for p, s in zip(det_p, labeled):
        pred_unsafe = p > 0.5
        is_unsafe = s["label_safe"] == 0
        tp += pred_unsafe and is_unsafe
        fp += pred_unsafe and not is_unsafe
        fn += (not pred_unsafe) and is_unsafe
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


@torch.no_grad()
def calibrate_temperature(model, val_samples, grid=None):
    """Pick the temperature that minimizes val NLL (simple grid search)."""
    labeled = [s for s in val_samples if s["label_safe"] != -1]
    if not labeled:
        return
    b = collate(labeled)
    det_logit, _, _ = model(b["x"], b["ei"], b["batch"], b["seed_mask"],
                            b["n_graphs"])
    y = b["det_y"]
    best_t, best_nll = 1.0, float("inf")
    for t in grid or [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        nll = F.binary_cross_entropy_with_logits(det_logit / t, y).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    model.temperature.fill_(best_t)
