"""P0.9 - does the improved grouping close the name-blind gap?

Compares three crossing partitions against the construction ground truth:

  names      the shipped name-based grouping (uses our cdc<k>_ convention)
  shipped    the shipped structural fallback (shared source cone only)
  improved   journal/grouping.py (adds forward reconvergence and shared
             source ancestors, transitively closed)

Scored two ways. Pairwise precision/recall/F1 over co-membership pairs judges
the partition itself and penalizes over-splitting and over-merging alike. Then
the trained ensemble is run on subgraphs built from each partition, which is
what actually matters.

Run:  python journal/scripts/j08_grouping.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from grouping import group_crossings_structural, pair_prf     # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.netlist_graph import (parse_netlist, find_capture_ffs,      # noqa: E402
                                        group_crossings, extract_subgraph,
                                        node_features, CrossingSample)
from crossexplain.rtl_generator import HAZARD_CLASSES, TEMPLATE_CLASS         # noqa: E402


def shipped_fallback(nl, caps):
    """The grouping in netlist_graph.group_crossings, applied to every capture
    flip-flop rather than only to the unnamed ones."""
    todo = sorted(caps)
    groups, used = [], set()
    for f in todo:
        if f in used:
            continue
        grp, now = [f], {f}
        sf, _ = nl.d_srcs[f]
        for g in todo:
            if g in used or g in now:
                continue
            sg, _ = nl.d_srcs[g]
            if nl.domain[f] == nl.domain[g] and (sf & sg):
                grp.append(g)
                now.add(g)
        used |= now
        groups.append(sorted(grp))
    return groups


def build(nl, caps, grp, gtmap, truth, design, level):
    seeds_set = set(grp)
    src_doms = set()
    for f in grp:
        src_doms |= caps[f][0]
    dst = nl.domain[grp[0]]
    nodes, edges, local = extract_subgraph(nl, grp, cid=None)
    feats = node_features(nl, nodes, seeds_set, src_doms, dst, edges)
    cone = set(grp)
    for f in grp:
        sf, cb = nl.d_srcs[f]
        cone |= sf | cb
    mask = [1 if g in cone else 0 for g in nodes]
    best_cid, best_ov = None, 0
    for cid, ffs in truth.items():
        ov = len(set(ffs) & seeds_set)
        if ov > best_ov:
            best_cid, best_ov = cid, ov
    s = CrossingSample(design=design, level=level,
                       cid=best_cid if best_cid is not None else -1,
                       nodes=nodes, node_feats=feats, edges=edges,
                       seeds=[local[f] for f in grp], loc_mask=mask,
                       src_dom=min(src_doms) if src_doms else -1, dst_dom=dst)
    if best_cid is not None and best_cid in gtmap:
        g = gtmap[best_cid]
        s.label_safe = 1 if g["safe"] else 0
        s.template = g["template"]
        cls = TEMPLATE_CLASS.get(g["template"])
        s.label_class = HAZARD_CLASSES.index(cls) if cls else -1
    return asdict(s)


J.banner("P0.9  Name-blind grouping: shipped fallback vs improved")
models = J.load_seed_models()
OUT = {}

for split in ["test_real", "test_id"]:
    ddir = J.ROOT / "data" / "designs" / split
    named = J.load(split)
    designs = sorted({s["design"] for s in named})
    agg = {"shipped": [0.0, 0.0, 0.0, 0], "improved": [0.0, 0.0, 0.0, 0]}
    samples = {"shipped": [], "improved": []}
    n_true, n_files = 0, 0

    for d in designs:
        gtf = ddir / f"{d}.json"
        if not gtf.exists():
            continue
        gt = json.loads(gtf.read_text())
        gtmap = {c["cid"]: c for c in gt["crossings"]}
        for lvl in ["O0", "O1", "O2", "O3"]:
            jf = ddir / f"{d}.{lvl}.json"
            if not jf.exists():
                continue
            nl = parse_netlist(jf)
            caps = find_capture_ffs(nl)
            if not caps:
                continue
            n_files += 1
            _, by_cid, _ = group_crossings(nl)
            truth = dict(by_cid)
            true_groups = [sorted(v) for v in truth.values()]
            n_true += len(true_groups)

            for name, groups in (("shipped", shipped_fallback(nl, caps)),
                                 ("improved", group_crossings_structural(nl, caps))):
                p, r, f1 = pair_prf(groups, true_groups)
                agg[name][0] += p
                agg[name][1] += r
                agg[name][2] += f1
                agg[name][3] += len(groups)
                for g in groups:
                    samples[name].append(build(nl, caps, g, gtmap, truth, d, lvl))

    n_files = max(1, n_files)
    print(f"\n  --- {split} ---   construction crossings: {n_true}")
    res = {"n_true_crossings": n_true, "n_named_rows": len(named)}
    for name in ("shipped", "improved"):
        p, r, f1, cnt = agg[name]
        lab = [s for s in samples[name] if s["label_safe"] != -1]
        y = J.y_of(lab)
        u = J.units(lab)
        ens = J.seed_probs(models, lab).mean(0) > 0.5
        m = J.metrics(ens, y)
        ci = J.clustered_ci(ens, y, u["design"], "f1", B=1500)
        res[name] = {
            "groups": cnt, "ratio_vs_truth": cnt / max(1, n_true),
            "pair_precision": p / n_files, "pair_recall": r / n_files,
            "pair_f1": f1 / n_files,
            "detection": m, "ci_design": ci, "n_labeled": len(lab),
        }
        print(f"    {name:9s} groups={cnt:5d} ({cnt / max(1, n_true):.2f}x truth)  "
              f"pairF1={f1 / n_files:.3f} (P={p / n_files:.3f} R={r / n_files:.3f})  "
              f"CrossExplain F1={m['f1']:.3f} [{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}]")

    y_n = J.y_of(named)
    ens_n = J.seed_probs(models, named).mean(0) > 0.5
    res["name_based_reference_f1"] = J.f1(ens_n, y_n)
    print(f"    {'names':9s} groups={len(named):5d} (1.00x truth)"
          f"{'':34s}CrossExplain F1={res['name_based_reference_f1']:.3f}")
    OUT[split] = res

J.banner("Verdict")
for sp, r in OUT.items():
    d_ratio = r["shipped"]["ratio_vs_truth"] - r["improved"]["ratio_vs_truth"]
    d_f1 = r["improved"]["detection"]["f1"] - r["shipped"]["detection"]["f1"]
    print(f"  {sp:10s} over-split {r['shipped']['ratio_vs_truth']:.2f}x -> "
          f"{r['improved']['ratio_vs_truth']:.2f}x (-{d_ratio:.2f}), "
          f"pairF1 {r['shipped']['pair_f1']:.3f} -> {r['improved']['pair_f1']:.3f}, "
          f"detection F1 {d_f1:+.3f} "
          f"(name-based reference {r['name_based_reference_f1']:.3f})")

J.save("j08_grouping", OUT)
