"""CrossExplain model: shared GNN encoder + four task heads.

Implemented in pure PyTorch (sparse index_add message passing) so the
project has no torch-geometric dependency.

Encoder variants:
  gcn   symmetric mean over undirected neighbourhood
  sage  separate in-edge / out-edge mean aggregation (directed signal flow)
  gat   additive attention on incoming edges; attention weights are kept
        and exposed for the explainability head

Heads:
  detection     graph-level: safe / unsafe          (BCE)
  hazard type   graph-level: 5 classes              (CE, unsafe samples)
  localization  node-level importance                (BCE vs hazard cone)
  confidence    calibrated detection probability (temperature scaled on val)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtl_generator import HAZARD_CLASSES
from .netlist_graph import N_FEATURES


def _agg_mean(msg, dst, n):
    out = torch.zeros(n, msg.size(1), device=msg.device)
    cnt = torch.zeros(n, 1, device=msg.device)
    out.index_add_(0, dst, msg)
    cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=msg.device))
    return out / cnt.clamp(min=1.0)


class SageLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w_self = nn.Linear(din, dout)
        self.w_in = nn.Linear(din, dout, bias=False)
        self.w_out = nn.Linear(din, dout, bias=False)

    def forward(self, h, ei):
        src, dst = ei
        m_in = _agg_mean(h[src], dst, h.size(0))    # messages along wires
        m_out = _agg_mean(h[dst], src, h.size(0))   # reverse direction
        return F.relu(self.w_self(h) + self.w_in(m_in) + self.w_out(m_out))


class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Linear(din, dout)

    def forward(self, h, ei):
        src, dst = ei
        both_src = torch.cat([src, dst])
        both_dst = torch.cat([dst, src])
        m = _agg_mean(h[both_src], both_dst, h.size(0))
        return F.relu(self.w(h + m))


class GATLayer(nn.Module):
    """Single-head additive attention over incoming edges (+ self loop).
    Stores the last attention distribution for explainability."""

    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Linear(din, dout, bias=False)
        self.w_self = nn.Linear(din, dout)
        self.att = nn.Linear(2 * dout, 1)
        self.last_alpha = None   # (edge_index, alpha) of the last forward

    def forward(self, h, ei):
        src, dst = ei
        z = self.w(h)
        e = F.leaky_relu(self.att(torch.cat([z[src], z[dst]], dim=1)),
                         negative_slope=0.2).squeeze(-1)
        # softmax over incoming edges of each dst node
        e = e - e.max()
        num = e.exp()
        den = torch.zeros(h.size(0), device=h.device)
        den.index_add_(0, dst, num)
        alpha = num / (den[dst] + 1e-9)
        self.last_alpha = (ei, alpha.detach())
        msg = torch.zeros(h.size(0), z.size(1), device=h.device)
        msg.index_add_(0, dst, z[src] * alpha.unsqueeze(-1))
        return F.relu(self.w_self(h) + msg)


_LAYERS = {"gcn": GCNLayer, "sage": SageLayer, "gat": GATLayer}


class CrossExplainNet(nn.Module):
    def __init__(self, arch: str = "gat", hidden: int = 64, layers: int = 3,
                 feat_dim: int = N_FEATURES, n_classes: int = len(HAZARD_CLASSES)):
        super().__init__()
        self.arch = arch
        self.embed = nn.Linear(feat_dim, hidden)
        Layer = _LAYERS[arch]
        self.layers = nn.ModuleList(Layer(hidden, hidden) for _ in range(layers))
        # graph readout: [mean pool ; seed pool] -> heads
        self.detect_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.type_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        self.loc_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x, ei, batch, seed_mask, n_graphs):
        """x [N,F]; ei [2,E]; batch [N] graph id; seed_mask [N] bool."""
        h = F.relu(self.embed(x))
        for layer in self.layers:
            h = layer(h, ei)

        mean_pool = _agg_mean(h, batch, n_graphs)
        seed_h = h * seed_mask.unsqueeze(-1)
        seed_cnt = torch.zeros(n_graphs, 1, device=h.device)
        seed_cnt.index_add_(0, batch, seed_mask.unsqueeze(-1).float())
        seed_pool = torch.zeros(n_graphs, h.size(1), device=h.device)
        seed_pool.index_add_(0, batch, seed_h)
        seed_pool = seed_pool / seed_cnt.clamp(min=1.0)

        g = torch.cat([mean_pool, seed_pool], dim=1)
        det_logit = self.detect_head(g).squeeze(-1)
        type_logit = self.type_head(g)
        loc_logit = self.loc_head(h).squeeze(-1)
        return det_logit, type_logit, loc_logit

    # ------------------------------------------------------ explainability
    def node_attention(self, loc_logit, batch, n_graphs):
        """Per-node attention scores in [0,1] (softmax within each graph)."""
        scores = torch.zeros_like(loc_logit)
        for g in range(n_graphs):
            m = batch == g
            scores[m] = torch.softmax(loc_logit[m], dim=0)
        return scores

    def confidence(self, det_logit):
        return torch.sigmoid(det_logit / self.temperature)


# ------------------------------------------------------------------ batching
def collate(samples, device="cpu"):
    """Merge crossing samples into one block-diagonal batch."""
    xs, eis, batch, seeds, det_y, type_y, loc_y = [], [], [], [], [], [], []
    offset = 0
    for gi, s in enumerate(samples):
        n = len(s["node_feats"])
        xs.append(torch.tensor(s["node_feats"], dtype=torch.float32))
        if s["edges"]:
            e = torch.tensor(s["edges"], dtype=torch.long).t()
        else:
            e = torch.zeros(2, 0, dtype=torch.long)
        eis.append(e + offset)
        batch.extend([gi] * n)
        sm = torch.zeros(n)
        sm[s["seeds"]] = 1.0
        seeds.append(sm)
        det_y.append(float(s["label_safe"] == 0))          # 1 = UNSAFE
        type_y.append(s["label_class"] if s["label_class"] >= 0 else -100)
        loc_y.append(torch.tensor(s["loc_mask"], dtype=torch.float32))
        offset += n
    return dict(
        x=torch.cat(xs).to(device),
        ei=torch.cat(eis, dim=1).to(device),
        batch=torch.tensor(batch, dtype=torch.long).to(device),
        seed_mask=torch.cat(seeds).to(device),
        det_y=torch.tensor(det_y).to(device),
        type_y=torch.tensor(type_y, dtype=torch.long).to(device),
        loc_y=torch.cat(loc_y).to(device),
        n_graphs=len(samples),
    )
