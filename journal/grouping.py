"""P0.9 - name-blind crossing grouping that does not over-split buses.

P0.7 showed the shipped structural fallback produces 2.16x the true number of
crossings on test_id and 3.96x on test_real, at seed purity 0.997. It does not
mix crossings, it shatters them: a multi-bit crossing becomes one group per
bit, which destroys exactly the word-level context BUS_NOSYNC detection needs.

The cause is the grouping predicate. The shipped rule joins two capture
flip-flops when their source cones intersect. For a bus crossing, bit i is fed
by source bit i and bit j by source bit j, so the cones are disjoint by
construction and every bit lands in its own group.

Three name-blind signals recover the bus. Two capture flip-flops belong to one
crossing when they share a destination domain and any of:

  S1  their source cones intersect                (the shipped rule, kept)
  S2  their forward cones meet within k hops      (bits recombine into a word:
                                                   an adder, a comparator, a
                                                   parity tree)
  S3  their source flip-flops share a fan-in
      ancestor within k hops                      (bits of one register share
                                                   a carry chain or a common
                                                   enable)

S2 is the load-bearing one. A multi-bit crossing exists precisely because the
bits are consumed together downstream; if they never reconverge they really are
independent single-bit crossings and splitting them is correct.

Grouping is the transitive closure of the pairwise relation, computed with a
union-find, so a bus links through its neighbours rather than needing all pairs
to match.
"""
from __future__ import annotations

from collections import defaultdict, deque

FWD_HOPS = 3
ANC_HOPS = 3


class _DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _succ_pred(nl):
    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in nl.edges:
        succ[a].append(b)
        pred[b].append(a)
    return succ, pred


def _cone(adj, start, hops):
    seen = set(start)
    frontier = list(start)
    for _ in range(hops):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return seen


def group_crossings_structural(nl, caps, fwd_hops=FWD_HOPS, anc_hops=ANC_HOPS):
    """Name-blind grouping of capture flip-flops into crossing instances.

    `caps` is the {ff: (src_domains, own_domain)} mapping from
    netlist_graph.find_capture_ffs. Returns a list of sorted flip-flop lists.
    """
    succ, pred = _succ_pred(nl)
    ffs = sorted(caps)
    if not ffs:
        return []

    fwd = {f: _cone(succ, [f], fwd_hops) - {f} for f in ffs}
    src = {f: nl.d_srcs[f][0] for f in ffs}
    anc = {}
    for f in ffs:
        start = [p for s in src[f] for p in pred[s]]
        anc[f] = _cone(pred, start, anc_hops) if start else set()

    dsu = _DSU(ffs)
    for i, f in enumerate(ffs):
        for g in ffs[i + 1:]:
            if nl.domain[f] != nl.domain[g]:
                continue
            if src[f] & src[g]:                       # S1
                dsu.union(f, g)
            elif fwd[f] & fwd[g]:                     # S2
                dsu.union(f, g)
            elif anc[f] and anc[g] and (anc[f] & anc[g]):   # S3
                dsu.union(f, g)

    groups = defaultdict(list)
    for f in ffs:
        groups[dsu.find(f)].append(f)
    return [sorted(v) for v in groups.values()]


# ------------------------------------------------------- partition scoring
def pair_prf(pred_groups, true_groups):
    """Pairwise precision/recall/F1 of a predicted partition against the
    construction partition, over co-membership pairs. This is the right
    metric here: it penalizes both over-splitting and over-merging, which
    a raw group count and a purity score cannot do together."""
    def pairs(groups):
        out = set()
        for g in groups:
            g = sorted(g)
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    out.add((g[i], g[j]))
        return out

    P, T = pairs(pred_groups), pairs(true_groups)
    if not P and not T:
        return 1.0, 1.0, 1.0
    tp = len(P & T)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
