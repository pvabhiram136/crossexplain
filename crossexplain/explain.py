"""Explanation rendering: per-crossing report + attention heatmap figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .models import collate
from .rtl_generator import HAZARD_CLASSES
from .netlist_graph import CELL_VOCAB


@torch.no_grad()
def explain_sample(model, sample):
    """Run the model on one crossing and assemble the CrossExplain report."""
    model.eval()
    b = collate([sample])
    det_logit, type_logit, loc_logit = model(
        b["x"], b["ei"], b["batch"], b["seed_mask"], 1)
    p_unsafe = torch.sigmoid(det_logit / model.temperature)[0].item()
    scores = torch.softmax(loc_logit, dim=0).numpy()
    k = int(type_logit.argmax(1)[0])
    verdict = "UNSAFE" if p_unsafe > 0.5 else "SAFE"
    conf = p_unsafe if p_unsafe > 0.5 else 1 - p_unsafe
    top = np.argsort(-scores)[:6]
    return dict(
        verdict=verdict,
        confidence=conf,
        hazard_type=HAZARD_CLASSES[k] if verdict == "UNSAFE" else None,
        attention=scores,
        top_nodes=top.tolist(),
        truth=("UNSAFE" if sample["label_safe"] == 0 else "SAFE",
               sample["template"]),
    )


def _node_label(sample, i):
    f = sample["node_feats"][i]
    t = CELL_VOCAB[int(np.argmax(f[:len(CELL_VOCAB)]))]
    short = {"$_DFF_P_": "FF", "$_DFF_N_": "FFn", "FF_OTHER": "FF*",
             "$_AND_": "AND", "$_NAND_": "NAND", "$_OR_": "OR",
             "$_NOR_": "NOR", "$_XOR_": "XOR", "$_XNOR_": "XNOR",
             "$_NOT_": "NOT", "$_MUX_": "MUX", "$_ANDNOT_": "ANDN",
             "$_ORNOT_": "ORN", "PORT_IN": "IN", "OTHER": "?"}
    return short.get(t, t.strip("$_"))


def attention_figure(model, sample, out_png: Path, title=""):
    """Draw the crossing subgraph with attention-weighted node colors."""
    rep = explain_sample(model, sample)
    n = len(sample["node_feats"])
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(sample["edges"])
    pos = nx.spring_layout(G, seed=3, k=1.6 / max(1, n ** 0.5))

    att = rep["attention"]
    att = att / (att.max() + 1e-12)
    seeds = set(sample["seeds"])
    truth_cone = {i for i, v in enumerate(sample["loc_mask"]) if v}

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#8a93a6",
                           arrows=True, arrowsize=9, width=0.8, alpha=0.6)
    colors = plt.cm.inferno(0.15 + 0.8 * att)
    sizes = [420 if i in seeds else 240 for i in range(n)]
    edgecolors = ["#16a34a" if i in truth_cone else "#c9ccd4" for i in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=sizes, edgecolors=edgecolors,
                           linewidths=[2.2 if i in truth_cone else 0.8
                                       for i in range(n)])
    nx.draw_networkx_labels(
        G, pos, {i: _node_label(sample, i) for i in range(n)},
        ax=ax, font_size=6.5, font_color="white")

    truth_v, tpl = rep["truth"]
    ht = rep["hazard_type"] or "-"
    ax.set_title(f"{title}  verdict={rep['verdict']} "
                 f"(conf {rep['confidence']:.2f}, type {ht})   "
                 f"truth={truth_v}/{tpl}\n"
                 "color = model attention · green ring = ground-truth hazard cone · "
                 "large node = capture FF", fontsize=8.5)
    ax.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return rep


def text_report(rep, sample):
    lines = [
        f"CrossExplain report — {sample['design']} @ {sample['level']} "
        f"(crossing cdc{sample['cid']})",
        f"  verdict     : {rep['verdict']}  (confidence {rep['confidence']:.2f})",
    ]
    if rep["hazard_type"]:
        lines.append(f"  hazard type : {rep['hazard_type']}")
    lines.append(f"  top nodes   : {rep['top_nodes']}")
    lines.append(f"  ground truth: {rep['truth'][0]} ({rep['truth'][1]})")
    return "\n".join(lines)
