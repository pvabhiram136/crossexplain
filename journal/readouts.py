"""Readout variants, to test the mean-pooling dilution hypothesis.

j15 found an interaction rather than a main effect: safe-side false positives
need both the qualifier-held idiom and a large neighbourhood. `SYNC2` is fine
at 14 and at 69 nodes; `MUX_HOLD` is perfect at 27 nodes and fails completely
at 410.

The proposed mechanism is the graph readout. `CrossExplainNet.forward` builds
its graph vector as

    g = [ mean_pool(h over ALL nodes) ; seed_pool(h over capture FFs) ]

The evidence that makes a qualifier-held bus safe is a short synchronizer
chain, roughly five nodes. In a 27-node subgraph that is about a fifth of the
mean-pooled half; in a 410-node subgraph it is about one percent. Mean pooling
would therefore wash the evidence out exactly as the subgraph grows, which is
the observed interaction. It is also the same readout that made encoder depth
irrelevant in FINDINGS_P0.md section G.

Only the FIRST half of the concatenation is varied here. The seed half and the
head shapes are identical across variants, so this ablates the pooling and
nothing else.

  mean_seed   [mean ; seed]              the shipped model, baseline
  seed_only   [seed ; seed]              no global pooling at all
  dist_seed   [distance-weighted ; seed] nodes near a capture FF dominate
  attn_seed   [attention-pooled ; seed]  learned per-node weights
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crossexplain.models import CrossExplainNet, _agg_mean, collate  # noqa: E402
from crossexplain.netlist_graph import CELL_VOCAB                    # noqa: E402
from crossexplain.train import deterministic_run, multitask_loss     # noqa: E402

# index of the normalized distance-to-nearest-seed feature
DIST_IDX = len(CELL_VOCAB) + 7

READOUTS = ("mean_seed", "seed_only", "dist_seed", "attn_seed")


class ReadoutNet(CrossExplainNet):
    """CrossExplainNet with a configurable global-pooling half."""

    def __init__(self, readout="mean_seed", **kw):
        super().__init__(**kw)
        if readout not in READOUTS:
            raise ValueError(readout)
        self.readout = readout
        hidden = self.embed.out_features
        if readout == "attn_seed":
            self.pool_att = nn.Linear(hidden, 1)

    def _global_half(self, h, x, batch, n_graphs):
        if self.readout == "seed_only":
            return None                      # caller substitutes seed_pool
        if self.readout == "mean_seed":
            return _agg_mean(h, batch, n_graphs)
        if self.readout == "dist_seed":
            # weight nodes by proximity to the nearest capture flip-flop; the
            # feature is min(dist, 8) / 8, so w = 1 at a seed and 1/9 far away
            w = 1.0 / (1.0 + 8.0 * x[:, DIST_IDX])
        else:                                 # attn_seed
            e = self.pool_att(h).squeeze(-1)
            e = e - e.max()
            num = e.exp()
            den = torch.zeros(n_graphs, device=h.device)
            den.index_add_(0, batch, num)
            w = num / (den[batch] + 1e-9)
        num = torch.zeros(n_graphs, h.size(1), device=h.device)
        num.index_add_(0, batch, h * w.unsqueeze(-1))
        if self.readout == "dist_seed":
            den = torch.zeros(n_graphs, 1, device=h.device)
            den.index_add_(0, batch, w.unsqueeze(-1))
            num = num / den.clamp(min=1e-9)
        return num

    def forward(self, x, ei, batch, seed_mask, n_graphs):
        h = F.relu(self.embed(x))
        for layer in self.layers:
            h = layer(h, ei)

        # The global half is accumulated BEFORE the seed half, matching
        # CrossExplainNet.forward exactly. The two orders are mathematically
        # identical and behaviourally are not: swapping them reorders two
        # `index_add_` reductions, which changes the trained model's real-world
        # false-positive rate from 0.032 to 1.000 while leaving `test_id` F1
        # pinned at 1.0000. See FINDINGS_DIAGNOSTIC.md. Do not reorder.
        glob = self._global_half(h, x, batch, n_graphs)

        seed_h = h * seed_mask.unsqueeze(-1)
        seed_cnt = torch.zeros(n_graphs, 1, device=h.device)
        seed_cnt.index_add_(0, batch, seed_mask.unsqueeze(-1).float())
        seed_pool = torch.zeros(n_graphs, h.size(1), device=h.device)
        seed_pool.index_add_(0, batch, seed_h)
        seed_pool = seed_pool / seed_cnt.clamp(min=1.0)

        if glob is None:                     # seed_only
            glob = seed_pool

        g = torch.cat([glob, seed_pool], dim=1)
        return (self.detect_head(g).squeeze(-1),
                self.type_head(g),
                self.loc_head(h).squeeze(-1))


def train_readout(train_samples, val_samples, readout="mean_seed", arch="gat",
                  hidden=64, layers=5, epochs=40, batch_size=32, lr=2e-3,
                  seed=0, threads=1):
    """Train one readout variant, mirroring crossexplain.train.train_model."""
    import random
    with deterministic_run(seed, threads=threads):
        rng = random.Random(seed)
        model = ReadoutNet(readout=readout, arch=arch, hidden=hidden,
                           layers=layers)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        data = [s for s in train_samples if s["label_safe"] != -1]
        data.sort(key=lambda s: (s["design"], s["level"], s["cid"]))
        best_f1, best_state = -1.0, None
        for _ in range(epochs):
            model.train()
            rng.shuffle(data)
            for i in range(0, len(data), batch_size):
                b = collate(data[i:i + batch_size])
                out = model(b["x"], b["ei"], b["batch"], b["seed_mask"],
                            b["n_graphs"])
                loss, _ = multitask_loss(out, b)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            f1 = _quick_f1(model, val_samples)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _quick_f1(model, samples):
    labeled = [s for s in samples if s["label_safe"] != -1]
    if not labeled:
        return 0.0
    p = predict_readout(model, labeled)
    tp = fp = fn = 0
    for pi, s in zip(p, labeled):
        pu, iu = pi > 0.5, s["label_safe"] == 0
        tp += pu and iu
        fp += pu and not iu
        fn += (not pu) and iu
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


@torch.no_grad()
def predict_readout(model, samples, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(samples), batch_size):
        b = collate(samples[i:i + batch_size])
        det, _, _ = model(b["x"], b["ei"], b["batch"], b["seed_mask"],
                          b["n_graphs"])
        out.extend(torch.sigmoid(det / model.temperature).tolist())
    return out
