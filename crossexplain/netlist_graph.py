"""Netlist → graph conversion, clock-domain inference, crossing extraction.

Input:  Yosys `write_json` netlist (flattened, gate-level).
Output: per-crossing samples: subgraph (nodes, edges, features), the capture
        seeds, the ground-truth localization mask, and labels joined from
        the generator's construction record.

Terminology:
  capture FF   the first flip-flop in the destination domain that samples a
               signal whose fan-in cone contains flip-flops of another domain
  crossing     a group of capture FFs belonging to one constructed CDC
               instance (grouped via the cdc<k>_ net-name convention, with a
               structural fallback for unlabeled netlists)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

FF_TYPES = {"$_DFF_P_", "$_DFF_N_", "$_DFFE_PP_", "$_DFFE_PN_",
            "$_DFF_PP0_", "$_DFF_PP1_", "$_SDFF_PP0_", "$_SDFF_PP1_",
            "$dff", "$dffe", "$sdff", "$adff"}

CELL_VOCAB = ["$_DFF_P_", "$_DFF_N_", "FF_OTHER",
              "$_AND_", "$_NAND_", "$_OR_", "$_NOR_", "$_XOR_", "$_XNOR_",
              "$_NOT_", "$_MUX_", "$_ANDNOT_", "$_ORNOT_",
              "$_AOI3_", "$_OAI3_", "$_AOI4_", "$_OAI4_",
              "PORT_IN", "OTHER"]
CELL_INDEX = {t: i for i, t in enumerate(CELL_VOCAB)}

CDC_RE = re.compile(r"(?:^|[.$\\])cdc(\d+)_")


def _is_ff(ctype: str) -> bool:
    return ctype in FF_TYPES or ctype.startswith("$_DFF") or ctype.startswith("$_SDFF")


def _vocab_type(ctype: str) -> str:
    if ctype in CELL_INDEX:
        return ctype
    if _is_ff(ctype):
        return "FF_OTHER"
    return "OTHER"


@dataclass
class Netlist:
    """Bit-level cell graph of one synthesized module."""
    cells: list            # [(name, type)]
    edges: list            # [(src_cell_idx, dst_cell_idx)]
    ff_idx: list           # indices of FF cells
    domain: list           # per-cell clock domain (-1 = comb / unknown)
    clocks: list           # clock port names, index = domain id
    cell_bits_out: list    # per-cell set of output bit ids
    bit_names: dict        # bit id -> [net names]
    fanin: list            # per-cell in-degree
    fanout: list           # per-cell out-degree
    cell_of_bit: dict      # bit id -> driving cell idx
    readers: dict          # bit id -> [reader cell idx]
    d_srcs: list = field(default_factory=list)  # per-FF: (src_ff_set, comb_set)


def parse_netlist(jfile: Path) -> Netlist:
    data = json.loads(Path(jfile).read_text())
    mods = data["modules"]
    top = None
    for name, m in mods.items():
        if m.get("attributes", {}).get("top") in (1, "1", "00000000000000000000000000000001"):
            top = m
            break
    if top is None:
        top = mods.get("top") or next(iter(mods.values()))

    ports = top.get("ports", {})
    netnames = top.get("netnames", {})
    cells_json = top.get("cells", {})

    bit_names = defaultdict(list)
    for nname, nn in netnames.items():
        for b in nn.get("bits", []):
            if isinstance(b, int):
                bit_names[b].append(nname)

    clocks = sorted([p for p, pj in ports.items()
                     if pj["direction"] == "input" and p.startswith("clk")])
    clock_bits = {}
    for ci, cname in enumerate(clocks):
        for b in ports[cname]["bits"]:
            if isinstance(b, int):
                clock_bits[b] = ci

    cells = []
    cell_bits_out = []
    cell_in_bits = []
    input_port_cells = {}

    # pseudo-cells for non-clock input ports (din, rst...)
    for pname, pj in ports.items():
        if pj["direction"] != "input" or pname in clocks:
            continue
        idx = len(cells)
        cells.append((pname, "PORT_IN"))
        cell_bits_out.append({b for b in pj["bits"] if isinstance(b, int)})
        cell_in_bits.append(set())
        input_port_cells[pname] = idx

    ff_idx = []
    ff_clk_bit = {}
    ff_d_bits = {}
    for cname, cj in cells_json.items():
        idx = len(cells)
        ctype = cj["type"]
        cells.append((cname, ctype))
        dirs = cj.get("port_directions", {})
        outs, ins = set(), set()
        for pname, bits in cj.get("connections", {}).items():
            d = dirs.get(pname)
            if d is None:  # infer: Y/Q outputs
                d = "output" if pname in ("Y", "Q") else "input"
            for b in bits:
                if not isinstance(b, int):
                    continue
                (outs if d == "output" else ins).add(b)
            if _is_ff(ctype):
                if pname in ("C", "CLK"):
                    for b in bits:
                        if isinstance(b, int):
                            ff_clk_bit[idx] = b
                if pname == "D":
                    ff_d_bits[idx] = [b for b in bits if isinstance(b, int)]
        cell_bits_out.append(outs)
        cell_in_bits.append(ins)
        if _is_ff(ctype):
            ff_idx.append(idx)

    cell_of_bit = {}
    for i, outs in enumerate(cell_bits_out):
        for b in outs:
            cell_of_bit[b] = i

    readers = defaultdict(list)
    edges = set()
    for i, ins in enumerate(cell_in_bits):
        for b in ins:
            readers[b].append(i)
            j = cell_of_bit.get(b)
            if j is not None and j != i:
                edges.add((j, i))

    n = len(cells)
    fanin = [0] * n
    fanout = [0] * n
    for a, bb in edges:
        fanout[a] += 1
        fanin[bb] += 1

    domain = [-1] * n
    for f in ff_idx:
        b = ff_clk_bit.get(f)
        domain[f] = clock_bits.get(b, -1) if b is not None else -1

    nl = Netlist(cells=cells, edges=sorted(edges), ff_idx=ff_idx, domain=domain,
                 clocks=clocks, cell_bits_out=cell_bits_out, bit_names=dict(bit_names),
                 fanin=fanin, fanout=fanout, cell_of_bit=cell_of_bit,
                 readers=dict(readers))

    # per-FF source analysis: walk D-pin fan-in through comb cells
    pred = defaultdict(list)
    for a, bb in nl.edges:
        pred[bb].append(a)
    d_srcs = {}
    for f in ff_idx:
        src_ffs, combs = set(), set()
        start = []
        for b in ff_d_bits.get(f, []):
            j = cell_of_bit.get(b)
            if j is not None:
                start.append(j)
        seen = set(start)
        dq = deque(start)
        while dq:
            u = dq.popleft()
            if _is_ff(cells[u][1]) or cells[u][1] == "PORT_IN":
                src_ffs.add(u)
                continue
            combs.add(u)
            for p in pred[u]:
                if p not in seen:
                    seen.add(p)
                    dq.append(p)
        d_srcs[f] = (src_ffs, combs)
    nl.d_srcs = d_srcs
    return nl


# --------------------------------------------------------------- crossings
@dataclass
class CrossingSample:
    design: str
    level: str
    cid: int                      # constructed crossing id (-1 if unmatched)
    nodes: list                   # netlist cell indices in subgraph order
    node_feats: list              # [N][F] float features
    edges: list                   # [(u,v)] subgraph-local indices
    seeds: list                   # subgraph-local capture FF indices
    loc_mask: list                # [N] 0/1 ground-truth hazard-cone mask
    label_safe: int = -1          # 1 safe, 0 unsafe, -1 unknown
    label_class: int = -1         # HAZARD_CLASSES index, -1 if safe/unknown
    template: str = ""
    src_dom: int = -1
    dst_dom: int = -1


def find_capture_ffs(nl: Netlist):
    """Return {ff_idx: (src_domains, own_domain)} for FFs fed (through comb
    logic only) by FFs of a different clock domain."""
    caps = {}
    for f in nl.ff_idx:
        own = nl.domain[f]
        if own < 0:
            continue
        src_ffs, _ = nl.d_srcs[f]
        doms = {nl.domain[s] for s in src_ffs
                if nl.cells[s][1] != "PORT_IN" and nl.domain[s] >= 0}
        others = {d for d in doms if d != own}
        if others:
            caps[f] = (others, own)
    return caps


def _cdc_ids_of_cell(nl: Netlist, idx: int) -> set:
    ids = set()
    outs = nl.cell_bits_out[idx]
    for b in outs:
        for name in nl.bit_names.get(b, []):
            m = CDC_RE.search(name)
            if m:
                ids.add(int(m.group(1)))
    # cell name itself may carry the prefix (yosys often derives it)
    m = CDC_RE.search(nl.cells[idx][0])
    if m:
        ids.add(int(m.group(1)))
    return ids


def group_crossings(nl: Netlist):
    """Group capture FFs into crossing instances.

    Primary: cdc<k>_ name aliases on the capture FF's output net.
    Fallback: connected grouping by (src_domain, dst_domain) over shared
    comb cones (for unlabeled/real netlists)."""
    caps = find_capture_ffs(nl)
    by_cid = defaultdict(list)
    unmatched = []
    for f in caps:
        ids = _cdc_ids_of_cell(nl, f)
        if len(ids) == 1:
            by_cid[ids.pop()].append(f)
        elif len(ids) > 1:
            by_cid[min(ids)].append(f)
        else:
            unmatched.append(f)

    # fallback grouping for unmatched capture FFs
    fallback = []
    used = set()
    for f in unmatched:
        if f in used:
            continue
        grp = [f]
        used.add(f)
        sf, _ = nl.d_srcs[f]
        for g in unmatched:
            if g in used:
                continue
            sg, _ = nl.d_srcs[g]
            if (nl.domain[f] == nl.domain[g]) and (sf & sg):
                grp.append(g)
                used.add(g)
        fallback.append(grp)
    return caps, dict(by_cid), fallback


def extract_subgraph(nl: Netlist, seeds: list, hops_fwd: int = 3,
                     hops_bwd: int = 6, cid: int | None = None):
    """Induced subgraph around a crossing's capture FFs."""
    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in nl.edges:
        succ[a].append(b)
        pred[b].append(a)

    keep = set(seeds)
    # backward: comb cone + source FFs + one more layer of context
    frontier = list(seeds)
    for _ in range(hops_bwd):
        nxt = []
        for u in frontier:
            for p in pred[u]:
                if p not in keep:
                    keep.add(p)
                    nxt.append(p)
        frontier = nxt
    # forward context
    frontier = list(seeds)
    for _ in range(hops_fwd):
        nxt = []
        for u in frontier:
            for s2 in succ[u]:
                if s2 not in keep:
                    keep.add(s2)
                    nxt.append(s2)
        frontier = nxt
    # everything carrying this crossing's name prefix (parallel chains etc.)
    if cid is not None:
        for i in range(len(nl.cells)):
            if i not in keep and cid in _cdc_ids_of_cell(nl, i):
                keep.add(i)

    nodes = sorted(keep)
    local = {g: i for i, g in enumerate(nodes)}
    sub_edges = [(local[a], local[b]) for a, b in nl.edges
                 if a in keep and b in keep]
    return nodes, sub_edges, local


def node_features(nl: Netlist, nodes: list, seeds: set, src_doms: set,
                  dst_dom: int, sub_edges: list):
    import math
    n = len(nodes)
    # BFS distance to nearest seed (undirected, within subgraph)
    adj = defaultdict(list)
    for a, b in sub_edges:
        adj[a].append(b)
        adj[b].append(a)
    local_seeds = [i for i, g in enumerate(nodes) if g in seeds]
    dist = [9] * n
    dq = deque()
    for s in local_seeds:
        dist[s] = 0
        dq.append(s)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                dq.append(v)

    feats = []
    for i, g in enumerate(nodes):
        name, ctype = nl.cells[g]
        onehot = [0.0] * len(CELL_VOCAB)
        onehot[CELL_INDEX[_vocab_type(ctype)]] = 1.0
        dom = nl.domain[g]
        f = onehot + [
            1.0 if _is_ff(ctype) else 0.0,
            1.0 if dom == dst_dom and dom >= 0 else 0.0,
            1.0 if dom in src_doms and dom >= 0 else 0.0,
            1.0 if dom < 0 else 0.0,
            math.log1p(nl.fanin[g]) / 3.0,
            math.log1p(nl.fanout[g]) / 3.0,
            1.0 if g in seeds else 0.0,
            min(dist[i], 8) / 8.0,
        ]
        feats.append(f)
    return feats


N_FEATURES = len(CELL_VOCAB) + 8


def extract_samples(jfile: Path, ground_truth: dict | None, design: str,
                    level: str):
    """All crossing samples of one synthesized netlist, labels joined from
    the generator ground truth (None → labels unknown / real design)."""
    from .rtl_generator import HAZARD_CLASSES, TEMPLATE_CLASS  # constants only

    nl = parse_netlist(jfile)
    caps, by_cid, fallback = group_crossings(nl)

    gt = {}
    if ground_truth:
        for c in ground_truth["crossings"]:
            gt[c["cid"]] = c

    samples = []

    def build(seed_ffs, cid):
        seeds_set = set(seed_ffs)
        src_doms = set()
        dst_dom = nl.domain[seed_ffs[0]]
        for f in seed_ffs:
            src_doms |= caps[f][0]
        nodes, sub_edges, local = extract_subgraph(nl, seed_ffs, cid=cid)
        feats = node_features(nl, nodes, seeds_set, src_doms, dst_dom, sub_edges)
        # localization mask: cells carrying this crossing's net prefix
        if cid is not None and cid >= 0:
            mask = [1 if cid in _cdc_ids_of_cell(nl, g) else 0 for g in nodes]
        else:
            mask = [0] * len(nodes)
        if sum(mask) == 0:
            # fallback: capture FFs + their comb cones + source FFs
            cone = set(seed_ffs)
            for f in seed_ffs:
                sf, cb = nl.d_srcs[f]
                cone |= sf | cb
            mask = [1 if g in cone else 0 for g in nodes]
        s = CrossingSample(
            design=design, level=level, cid=cid if cid is not None else -1,
            nodes=nodes, node_feats=feats, edges=sub_edges,
            seeds=[local[f] for f in seed_ffs], loc_mask=mask,
            src_dom=min(src_doms) if src_doms else -1, dst_dom=dst_dom,
        )
        if cid is not None and cid in gt:
            g = gt[cid]
            s.label_safe = 1 if g["safe"] else 0
            s.template = g["template"]
            cls = TEMPLATE_CLASS.get(g["template"])
            s.label_class = HAZARD_CLASSES.index(cls) if cls else -1
        return s

    for cid, ffs in sorted(by_cid.items()):
        samples.append(build(ffs, cid))
    for grp in fallback:
        samples.append(build(grp, None))
    return samples, nl
