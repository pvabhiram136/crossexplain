"""P0.7 - name-blind operation.

The paper states that crossing discovery has "a structural fallback that groups
by shared source cone and destination domain, so that the method also applies
to netlists without our naming convention." That path is never exercised in any
reported number: every design in every split, including the hand-written ones,
carries the cdc<k>_ prefix, so grouping always takes the name-based branch.

This script runs the pipeline with names withheld:

  * capture flip-flops are discovered structurally (unchanged, never used names)
  * they are grouped ONLY by the structural fallback
  * subgraphs are extracted with cid=None, so no cell is pulled in by name
  * labels are transferred from the construction record by seed overlap, purely
    for scoring - the model never sees them

Two questions. Does structural grouping recover the same crossing partition?
And does a model trained on name-grouped samples still work on name-blind ones?

Run:  python journal/scripts/j05_name_blind.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.netlist_graph import (parse_netlist, find_capture_ffs,      # noqa: E402
                                        group_crossings, extract_subgraph,
                                        node_features, CrossingSample)
from crossexplain.rtl_generator import HAZARD_CLASSES, TEMPLATE_CLASS         # noqa: E402
from crossexplain.baselines import pattern_matcher_predict, FlatML            # noqa: E402
from dataclasses import asdict                                                # noqa: E402


def blind_group(nl):
    """Group every capture FF using ONLY the structural fallback rule:
    same destination domain and a shared source cone."""
    caps = find_capture_ffs(nl)
    todo = sorted(caps)
    groups, used = [], set()
    for f in todo:
        if f in used:
            continue
        grp, used_now = [f], {f}
        sf, _ = nl.d_srcs[f]
        for g in todo:
            if g in used or g in used_now:
                continue
            sg, _ = nl.d_srcs[g]
            if nl.domain[f] == nl.domain[g] and (sf & sg):
                grp.append(g)
                used_now.add(g)
        used |= used_now
        groups.append(sorted(grp))
    return caps, groups


def build_blind(jfile, gt, design, level):
    """Name-blind samples, with labels transferred by seed overlap."""
    nl = parse_netlist(jfile)
    caps, groups = blind_group(nl)
    # the name-based (ground truth) partition, for comparison and labelling
    _, by_cid, _ = group_crossings(nl)
    truth = {cid: set(ffs) for cid, ffs in by_cid.items()}
    gtmap = {c["cid"]: c for c in (gt or {}).get("crossings", [])}

    out = []
    for grp in groups:
        seeds_set = set(grp)
        src_doms = set()
        for f in grp:
            src_doms |= caps[f][0]
        dst_dom = nl.domain[grp[0]]
        nodes, sub_edges, local = extract_subgraph(nl, grp, cid=None)
        feats = node_features(nl, nodes, seeds_set, src_doms, dst_dom, sub_edges)
        cone = set(grp)
        for f in grp:
            sf, cb = nl.d_srcs[f]
            cone |= sf | cb
        mask = [1 if g in cone else 0 for g in nodes]

        # label transfer: the construction crossing sharing the most seeds
        best_cid, best_ov = None, 0
        for cid, ffs in truth.items():
            ov = len(ffs & seeds_set)
            if ov > best_ov:
                best_cid, best_ov = cid, ov
        s = CrossingSample(design=design, level=level,
                           cid=best_cid if best_cid is not None else -1,
                           nodes=nodes, node_feats=feats, edges=sub_edges,
                           seeds=[local[f] for f in grp], loc_mask=mask,
                           src_dom=min(src_doms) if src_doms else -1,
                           dst_dom=dst_dom)
        if best_cid is not None and best_cid in gtmap:
            g = gtmap[best_cid]
            s.label_safe = 1 if g["safe"] else 0
            s.template = g["template"]
            cls = TEMPLATE_CLASS.get(g["template"])
            s.label_class = HAZARD_CLASSES.index(cls) if cls else -1
        d = asdict(s)
        d["_n_seeds"] = len(grp)
        d["_purity"] = best_ov / max(1, len(grp))
        out.append(d)
    return out, len(truth), len(groups)


J.banner("P0.7  Name-blind crossing discovery")
OUT = {}
models = J.load_seed_models()
flat = FlatML(seed=0).fit(J.load("train"))

for split, folder in [("test_real", "test_real"), ("test_id", "test_id")]:
    designs = sorted({s["design"] for s in J.load(split)})
    ddir = J.ROOT / "data" / "designs" / folder
    blind, n_true_tot, n_blind_tot = [], 0, 0
    for d in designs:
        gtf = ddir / f"{d}.json"
        if not gtf.exists():
            continue
        import json
        gt = json.loads(gtf.read_text())
        for lvl in ["O0", "O1", "O2", "O3"]:
            jf = ddir / f"{d}.{lvl}.json"
            if not jf.exists():
                continue
            try:
                samples, n_true, n_blind = build_blind(jf, gt, d, lvl)
            except Exception as e:                      # noqa: BLE001
                print(f"    [skip] {d}@{lvl}: {e}")
                continue
            blind.extend(samples)
            n_true_tot += n_true
            n_blind_tot += n_blind

    lab = [s for s in blind if s["label_safe"] != -1]
    pure = float(np.mean([s["_purity"] for s in lab])) if lab else 0.0
    named = J.load(split)
    print(f"\n  --- {split} ---")
    print(f"    name-based crossings: {len(named):4d}   "
          f"name-blind groups: {len(blind):4d}   labelled: {len(lab):4d}")
    print(f"    construction crossings seen {n_true_tot}, structural groups "
          f"{n_blind_tot}  (ratio {n_blind_tot / max(1, n_true_tot):.2f})")
    print(f"    mean seed purity of structural groups: {pure:.3f}")

    if not lab:
        continue
    y = J.y_of(lab)
    u = J.units(lab)
    ens = J.seed_probs(models, lab).mean(0) > 0.5
    pm = np.array([p == 0 for p in pattern_matcher_predict(lab)])
    rf = np.array([d_ == 0 for d_ in flat.predict(lab)[0]])

    # the same model on the name-grouped version of the same split
    y_n = J.y_of(named)
    ens_n = J.seed_probs(models, named).mean(0) > 0.5

    res = {
        "n_named": len(named), "n_blind_labeled": len(lab),
        "group_ratio": n_blind_tot / max(1, n_true_tot), "seed_purity": pure,
        "blind": {"crossexplain": J.metrics(ens, y),
                  "pattern_matcher": J.metrics(pm, y),
                  "random_forest": J.metrics(rf, y)},
        "named_reference": {"crossexplain": J.metrics(ens_n, y_n)},
        "blind_ci_design": J.clustered_ci(ens, y, u["design"], "f1", B=2000),
    }
    OUT[split] = res
    print(f"    CrossExplain  name-based F1={res['named_reference']['crossexplain']['f1']:.3f}"
          f"   name-blind F1={res['blind']['crossexplain']['f1']:.3f}"
          f"  [{res['blind_ci_design']['ci_lo']:.3f},{res['blind_ci_design']['ci_hi']:.3f}]")
    print(f"    matcher       name-blind F1={res['blind']['pattern_matcher']['f1']:.3f}"
          f"   RF name-blind F1={res['blind']['random_forest']['f1']:.3f}")

J.banner("Verdict")
for sp, r in OUT.items():
    drop = r["named_reference"]["crossexplain"]["f1"] - r["blind"]["crossexplain"]["f1"]
    print(f"  {sp:10s} F1 drop when names are withheld: {drop:+.3f}   "
          f"(grouping recovers {r['group_ratio']:.2f}x the crossing count, "
          f"purity {r['seed_purity']:.2f})")

J.save("j05_name_blind", OUT)
