"""Baselines.

1. PatternMatcher — a classical structural CDC checker of the kind
   commercial tools implement: recognize 2-FF synchronizer shapes.
   Its verdict logic (on the extracted crossing subgraph):
     * comb logic between source FFs and a capture FF  -> UNSAFE
     * capture FF fans out to anything besides exactly one same-domain FF
       (the second stage)                              -> UNSAFE
     * otherwise                                       -> SAFE
   These are exactly the textbook rules — the experiments show where they
   produce false positives (MUX_HOLD) and false negatives (RECONV,
   BUS_NOSYNC), and how they degrade as synthesis mangles structure.

2. RandomForest / flat-ML — handcrafted per-crossing features, no graph.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .netlist_graph import CELL_INDEX, CELL_VOCAB

_FF_COLS = [CELL_INDEX["$_DFF_P_"], CELL_INDEX["$_DFF_N_"],
            CELL_INDEX["FF_OTHER"]]


def _node_type(feat):
    return int(np.argmax(feat[:len(CELL_VOCAB)]))


def _is_ff_feat(feat):
    return feat[len(CELL_VOCAB)] > 0.5


def _dom_flags(feat):
    base = len(CELL_VOCAB) + 1
    return feat[base] > 0.5, feat[base + 1] > 0.5, feat[base + 2] > 0.5
    # (dst_domain, src_domain, comb/none)


class PatternMatcher:
    """Classical structural 2FF-synchronizer rule set."""

    def predict(self, sample):
        feats = sample["node_feats"]
        edges = sample["edges"]
        seeds = set(sample["seeds"])
        n = len(feats)
        pred = {i: [] for i in range(n)}
        succ = {i: [] for i in range(n)}
        for a, b in edges:
            pred[b].append(a)
            succ[a].append(b)

        for cap in seeds:
            # rule 1: only FF sources may feed the capture FF directly
            stack = list(pred[cap])
            seen = set(stack)
            while stack:
                u = stack.pop()
                if _is_ff_feat(feats[u]):
                    continue
                is_dst, is_src, is_comb = _dom_flags(feats[u])
                # comb cell in the capture cone -> classic violation
                return 0  # UNSAFE
            # rule 2: capture FF must feed exactly one same-domain FF
            outs = succ[cap]
            ff_outs = [v for v in outs if _is_ff_feat(feats[v])]
            comb_outs = [v for v in outs if not _is_ff_feat(feats[v])]
            if comb_outs or len(ff_outs) != 1:
                return 0  # UNSAFE
        return 1  # SAFE


def pattern_matcher_predict(samples):
    m = PatternMatcher()
    return [m.predict(s) for s in samples]


# ------------------------------------------------------------------ flat ML
def flat_features(sample):
    feats = np.asarray(sample["node_feats"], dtype=np.float32)
    edges = sample["edges"]
    seeds = sample["seeds"]
    n = len(feats)
    n_ff = float(sum(_is_ff_feat(f) for f in feats))
    n_comb = n - n_ff
    succ = {i: 0 for i in range(n)}
    for a, b in edges:
        succ[a] += 1
    seed_fanout = max((succ[s] for s in seeds), default=0)
    return np.array([
        n, len(edges), len(seeds), n_ff, n_comb,
        n_comb / max(n, 1), seed_fanout,
        feats[:, -1].mean(),                     # mean distance-to-seed
        float(sum(_dom_flags(f)[1] for f in feats)),   # src-domain nodes
        float(sum(_dom_flags(f)[0] for f in feats)),   # dst-domain nodes
    ], dtype=np.float32)


class FlatML:
    def __init__(self, seed=0):
        self.det = RandomForestClassifier(n_estimators=200, random_state=seed)
        self.typ = RandomForestClassifier(n_estimators=200, random_state=seed)

    def fit(self, samples):
        labeled = [s for s in samples if s["label_safe"] != -1]
        X = np.stack([flat_features(s) for s in labeled])
        y = np.array([1 - s["label_safe"] for s in labeled])  # 1 = unsafe
        self.det.fit(X, y)
        unsafe = [s for s in labeled if s["label_safe"] == 0
                  and s["label_class"] >= 0]
        if unsafe:
            Xu = np.stack([flat_features(s) for s in unsafe])
            yu = np.array([s["label_class"] for s in unsafe])
            self.typ.fit(Xu, yu)
        return self

    def predict(self, samples):
        X = np.stack([flat_features(s) for s in samples])
        det = self.det.predict(X)          # 1 = unsafe
        typ = self.typ.predict(X)
        return [1 - int(d) for d in det], [int(t) for t in typ]
