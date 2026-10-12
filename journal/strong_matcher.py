"""A fair structural CDC checker, for P0.5.

Why this exists
---------------
The baseline in crossexplain/baselines.py reduces to a single rule: "any
combinational cell directly driving the capture flip-flop means unsafe". It
computes the domain flags of that cell and then discards them, so it never
checks whether the combinational cone actually crosses a clock domain, and its
backward traversal is dead code (the stack is never extended). MUX_HOLD scoring
0.000 against it is therefore partly a genuine limitation of structural
matching and partly an artifact of a weak rule.

Before claiming "rules cannot do this, message passing can", the rules deserve
their best shot. This module implements what a real structural checker does:

  R1  domain-aware crossing path. Combinational logic only matters if it sits
      on a path from a source-domain flip-flop to the capture flip-flop.
      Combinational logic fed entirely from the destination domain is not a
      crossing hazard.
  R2  convergence. Two or more distinct source-domain flip-flops merging into
      one combinational cone before the capture flip-flop is a real hazard
      (this is what catches COMB_BEFORE_SYNC and RECONV).
  R3  qualified hold. A recirculating mux whose select is driven by a
      synchronized qualifier, with one data leg looping back from the capture
      register, is the standard safe bus-handover idiom. Recognized in both
      shapes it takes: an explicit $_MUX_ feeding a plain flip-flop (O0), and
      an enable flip-flop that absorbed the mux during optimization (O2+).
  R4  synchronizer chain and leakage. A clean two-stage chain is safe only if
      the first stage feeds exactly one same-domain flip-flop and nothing
      else. Any other consumer is a metastability leak.
  R5  multi-bit coherency. Several bits crossing with per-bit synchronizers is
      unsafe unless the encoding guarantees single-bit changes. Gray encoders
      are XOR-dominated; binary counters carry an AND/OR carry chain. The
      ratio is used as the structural proxy.

Every threshold here is fitted on the TRAIN split only. Tuning a baseline on
the test split would be exactly the sin we are trying to avoid.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from crossexplain.netlist_graph import CELL_VOCAB

NV = len(CELL_VOCAB)
IDX = {t: i for i, t in enumerate(CELL_VOCAB)}
FF_TYPES = {"$_DFF_P_", "$_DFF_N_", "FF_OTHER"}
XOR_TYPES = {"$_XOR_", "$_XNOR_"}
CARRY_TYPES = {"$_AND_", "$_NAND_", "$_OR_", "$_NOR_", "$_ANDNOT_", "$_ORNOT_",
               "$_AOI3_", "$_OAI3_", "$_AOI4_", "$_OAI4_"}

# fitted on train only, see fit_thresholds()
DEFAULT_GRAY_XOR_RATIO = 0.60


def _ctype(f):
    return CELL_VOCAB[int(np.argmax(f[:NV]))]


def _is_ff(f):
    return f[NV] > 0.5


def _dst(f):
    return f[NV + 1] > 0.5


def _src(f):
    return f[NV + 2] > 0.5


class StrongPatternMatcher:
    """Domain-aware structural checker. predict() returns 1 safe / 0 unsafe."""

    def __init__(self, gray_xor_ratio: float = DEFAULT_GRAY_XOR_RATIO,
                 enable_r3: bool = True, enable_r5: bool = True):
        self.gray_xor_ratio = gray_xor_ratio
        self.enable_r3 = enable_r3
        self.enable_r5 = enable_r5

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _graph(sample):
        pred, succ = defaultdict(list), defaultdict(list)
        for a, b in sample["edges"]:
            pred[b].append(a)
            succ[a].append(b)
        return pred, succ

    def _back_to_ffs(self, feats, pred, start, max_depth=12):
        """Walk backward through combinational cells only.

        Returns (ffs_reached, comb_cells_traversed, saw_mux).
        """
        ffs, comb, mux = set(), set(), False
        dq = deque((p, 0) for p in start)
        seen = set(start)
        while dq:
            u, d = dq.popleft()
            if _is_ff(feats[u]) or _ctype(feats[u]) == "PORT_IN":
                ffs.add(u)
                continue
            comb.add(u)
            if _ctype(feats[u]) == "$_MUX_":
                mux = True
            if d >= max_depth:
                continue
            for p in pred[u]:
                if p not in seen:
                    seen.add(p)
                    dq.append((p, d + 1))
        return ffs, comb, mux

    def _sync_depth(self, feats, pred, node, max_stages=4):
        """How many chained destination-domain FFs sit behind `node`, with no
        combinational logic between the stages."""
        depth, cur = 0, node
        for _ in range(max_stages):
            drivers = pred[cur]
            ffs = [d for d in drivers if _is_ff(feats[d]) and _dst(feats[d])]
            combs = [d for d in drivers if not _is_ff(feats[d])]
            if len(ffs) == 1 and not combs:
                depth += 1
                cur = ffs[0]
            elif len(ffs) == 1 and all(_ctype(feats[c]) == "$_NOT_" for c in combs):
                depth += 1
                cur = ffs[0]
            else:
                break
        return depth

    def _qualified_hold(self, sample, feats, pred, succ, cap):
        """R3: is this capture register a mux-recirculation hold driven by a
        synchronized qualifier?"""
        drivers = pred[cap]
        muxes = [d for d in drivers if _ctype(feats[d]) == "$_MUX_"]

        # shape A: explicit mux in front of a plain flip-flop
        if muxes:
            mux = muxes[0]
            legs = pred[mux]
            recirculates = cap in legs or any(
                cap in pred[l] for l in legs if not _is_ff(feats[l]))
            best = max((self._sync_depth(feats, pred, l)
                        for l in legs if _is_ff(feats[l]) and _dst(feats[l])),
                       default=0)
            if recirculates and best >= 2:
                return True
            # qualifier may arrive through a small comb cone (toggle detect)
            for l in legs:
                if _is_ff(feats[l]):
                    continue
                qffs, qcomb, _ = self._back_to_ffs(feats, pred, [l], max_depth=3)
                dstq = [q for q in qffs if _dst(feats[q])]
                if recirculates and dstq and max(
                        self._sync_depth(feats, pred, q) for q in dstq) >= 2:
                    return True
            return False

        # shape B: optimization folded the mux into an enable flip-flop
        if _ctype(feats[cap]) == "FF_OTHER":
            for d in drivers:
                if _is_ff(feats[d]) and _dst(feats[d]):
                    if self._sync_depth(feats, pred, d) >= 2:
                        return True
                if not _is_ff(feats[d]):
                    qffs, _, _ = self._back_to_ffs(feats, pred, [d], max_depth=3)
                    dstq = [q for q in qffs if _dst(feats[q])]
                    if dstq and max(self._sync_depth(feats, pred, q)
                                    for q in dstq) >= 2:
                        return True
        return False

    def _gray_like(self, sample, feats, pred, src_ffs):
        """R5 proxy: is the source-side encoding XOR-dominated (Gray) rather
        than carry-chain dominated (binary counter)?"""
        xor = carry = 0
        for s in src_ffs:
            cone, _, _ = self._back_to_ffs(feats, pred, pred[s], max_depth=6)
            dq = deque(pred[s])
            seen = set(dq)
            while dq:
                u = dq.popleft()
                if _is_ff(feats[u]):
                    continue
                t = _ctype(feats[u])
                if t in XOR_TYPES:
                    xor += 1
                elif t in CARRY_TYPES:
                    carry += 1
                for p in pred[u]:
                    if p not in seen:
                        seen.add(p)
                        dq.append(p)
        total = xor + carry
        if total == 0:
            return True          # no arithmetic at all, treat as benign
        return (xor / total) >= self.gray_xor_ratio

    # ------------------------------------------------------------- verdict
    def predict(self, sample):
        feats = sample["node_feats"]
        pred, succ = self._graph(sample)
        seeds = list(sample["seeds"])

        multibit_src = set()
        verdicts = []
        qualifier_protected = False
        for cap in seeds:
            drivers = pred[cap]
            src_direct = [d for d in drivers if _is_ff(feats[d]) and _src(feats[d])]
            ffs, comb, saw_mux = self._back_to_ffs(feats, pred, drivers)
            src_ffs = {f for f in ffs if _src(feats[f])}
            multibit_src |= src_ffs

            # R1: does combinational logic sit on the crossing path itself?
            comb_on_cross_path = bool(comb) and bool(src_ffs) and not src_direct

            # R3: the safe qualified-hold idiom.
            # Deliberately NOT gated on comb_on_cross_path. After optimization
            # the recirculation mux is absorbed into an enable flip-flop, so
            # the combinational evidence disappears and the source register
            # drives the D pin directly. Checking the idiom unconditionally is
            # the most generous reading of the rule we can give the baseline.
            if self.enable_r3 and bool(src_ffs) and \
                    self._qualified_hold(sample, feats, pred, succ, cap):
                verdicts.append(1)
                qualifier_protected = True
                continue

            # R2: convergence of several source flip-flops through logic
            if comb_on_cross_path and len(src_ffs) >= 2:
                verdicts.append(0)
                continue
            if comb_on_cross_path:
                verdicts.append(0)
                continue

            # R4: clean chain, but check the first stage does not leak
            outs = succ[cap]
            ff_outs = [v for v in outs if _is_ff(feats[v]) and _dst(feats[v])]
            other_outs = [v for v in outs if v not in ff_outs]
            if other_outs or len(ff_outs) != 1:
                verdicts.append(0)
                continue
            verdicts.append(1)

        if not verdicts:
            return 1
        if any(v == 0 for v in verdicts):
            return 0

        # R5: every bit looks like a clean synchronizer, but is the WORD
        # coherent? Multiple bits crossing independently is only safe under a
        # single-bit-change encoding.
        #
        # Exemption: a qualified hold (R3) already guarantees word coherency,
        # because the bus is captured under a synchronized enable while it is
        # stable. That is the entire point of the idiom, and it is why a
        # qualifier-held bus does not need Gray coding. Applying the encoding
        # test to such a crossing is precisely the false positive that the
        # original matcher makes.
        if self.enable_r5 and len(multibit_src) >= 2 and not qualifier_protected:
            if not self._gray_like(sample, feats, pred, multibit_src):
                return 0
        return 1


def strong_predict(samples, **kw):
    m = StrongPatternMatcher(**kw)
    return [m.predict(s) for s in samples]


def fit_thresholds(train_samples, grid=None):
    """Pick the Gray/carry ratio on TRAIN only, maximizing accuracy."""
    y = np.array([s["label_safe"] == 0 for s in train_samples])
    best, best_acc = DEFAULT_GRAY_XOR_RATIO, -1.0
    for r in (grid or np.arange(0.30, 0.95, 0.05)):
        pred = np.array([p == 0 for p in strong_predict(train_samples,
                                                        gray_xor_ratio=float(r))])
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc, best = acc, float(r)
    return best, best_acc
