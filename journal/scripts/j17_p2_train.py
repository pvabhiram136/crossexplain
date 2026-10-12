"""P2: retrain with context-rich safe crossings, then confirm on held-out data.

Protocol, in this order, deliberately
-------------------------------------
JOURNAL_PLAN.md Part 3 warned that finding (d) was diagnosed on `test_real`, so
fixing the generator from it and re-evaluating there is fitting to the test
set. Since then the wrapped corpus (j14) and the size-by-idiom grid (j15) have
also been used for diagnosis, so all three are contaminated.

The **16 found designs** from triage have never been scored by any model. They
are third-party CDC libraries (OpenTitan `prim_*`, pulp `cc_cdc_*`,
alexforencich and ZipCPU asynchronous FIFOs), that is, reference
implementations whose crossings are correctly synchronized by construction.
They are labeled safe and used **once**, as the confirmation set.

The generator change is frozen before that set is touched. Its false-positive
rate is the number that decides whether P2 worked.

Balance
-------
Context-rich templates are added in both classes, two safe and two unsafe. Only
enlarging the safe side would teach "large implies safe", swapping one bias for
its mirror image.

Run:  python journal/scripts/j17_p2_train.py
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from gen_p2 import RTLGeneratorP2, ALL_CTX, SAFE_CTX          # noqa: E402
from grouping import group_crossings_structural               # noqa: E402
from wrappers import extract_wrapped, make_wrapper, CORES     # noqa: E402
from readouts import train_readout, predict_readout           # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.rtl_generator import write_design           # noqa: E402
from crossexplain.synthesize import synthesize                # noqa: E402
from crossexplain.netlist_graph import (extract_samples, parse_netlist,  # noqa: E402
                                        find_capture_ffs, extract_subgraph,
                                        node_features, CrossingSample)

CORPUS = J.JROOT / "corpus"
P2DIR = CORPUS / "p2designs"
P2DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]
N_DESIGNS = 60
LEVELS_TRAIN = ["O0", "O2"]
LEVELS = ["O0", "O1", "O2", "O3"]

# ---------------------------------------------------------------- A. build
J.banner("A. Generating context-rich designs")
ctx = []
for sd in range(N_DESIGNS):
    g = RTLGeneratorP2(seed=5000 + sd)
    spec = g.generate_ctx(f"p2_{sd:03d}", n_clocks=g.rng.choice([2, 3]),
                          n_crossings=g.rng.choice([3, 4]))
    write_design(spec, P2DIR)
    for lvl in LEVELS_TRAIN:
        jf = P2DIR / f"{spec.name}.{lvl}.json"
        if not jf.exists() and not synthesize(P2DIR / f"{spec.name}.v", jf, lvl):
            continue
        try:
            ss, _ = extract_samples(jf, spec.ground_truth(), spec.name, lvl)
        except Exception:                                      # noqa: BLE001
            continue
        ctx.extend(asdict(s) for s in ss if s.label_safe != -1)
    if (sd + 1) % 20 == 0:
        print(f"  {sd + 1}/{N_DESIGNS} designs, {len(ctx)} samples")

cn = np.array([len(s["node_feats"]) for s in ctx])
cy = J.y_of(ctx)
print(f"  {len(ctx)} samples, safe {int((~cy).sum())} / unsafe {int(cy.sum())}")
print(f"  nodes: mean {cn.mean():.0f}, median {int(np.median(cn))}, "
      f"range {cn.min()}-{cn.max()}")

base_train = J.load("train")
val = J.load("val")
p2_train = base_train + ctx
bn = np.array([len(s["node_feats"]) for s in base_train])
print(f"\n  baseline train {len(base_train)} samples, mean {bn.mean():.1f} nodes")
print(f"  P2 train       {len(p2_train)} samples, "
      f"mean {np.concatenate([bn, cn]).mean():.1f} nodes")

FREEZE = dict(frozen_utc=datetime.utcnow().isoformat() + "Z",
              n_ctx_designs=N_DESIGNS, n_ctx_samples=len(ctx),
              templates=ALL_CTX, safe_templates=SAFE_CTX,
              ctx_mean_nodes=float(cn.mean()))
print(f"\n  GENERATOR FROZEN at {FREEZE['frozen_utc']}")

# ------------------------------------------------ B. confirmation set
J.banner("B. Confirmation set: 16 found designs, never scored before")
triage = json.loads((J.JROOT / "results" / "j12_corpus_triage.json").read_text())
found = [r for r in triage if r.get("usable") and "duplicate_of" not in r]
conf = []
for r in found:
    jf = CORPUS / "netlists" / f"{r['name']}.O0.json"
    if not jf.exists():
        continue
    nl = parse_netlist(jf)
    caps = find_capture_ffs(nl)
    if not caps:
        continue
    for grp in group_crossings_structural(nl, caps):
        seeds = set(grp)
        src_doms = set()
        for f in grp:
            src_doms |= caps[f][0]
        dst = nl.domain[grp[0]]
        nodes, edges, local = extract_subgraph(nl, grp, cid=None)
        feats = node_features(nl, nodes, seeds, src_doms, dst, edges)
        cone = set(grp)
        for f in grp:
            sf, cb = nl.d_srcs[f]
            cone |= sf | cb
        s = CrossingSample(
            design=r["name"], level="O0", cid=-1, nodes=nodes,
            node_feats=feats, edges=edges,
            seeds=[local[f] for f in grp],
            loc_mask=[1 if g in cone else 0 for g in nodes],
            label_safe=1,           # reference implementations: safe by design
            label_class=-1, template="FOUND_SAFE",
            src_dom=min(src_doms) if src_doms else -1, dst_dom=dst)
        conf.append(asdict(s))
cnodes = np.array([len(s["node_feats"]) for s in conf])
print(f"  {len(conf)} crossings from {len({s['design'] for s in conf})} designs")
print(f"  all labeled safe; nodes mean {cnodes.mean():.0f}, "
      f"range {cnodes.min()}-{cnodes.max()}")
print("  ASSUMPTION: these libraries implement CDC correctly, so every")
print("  crossing in them is safe. False positives here are model errors.")

# ------------------------------------------------------- C. other sets
test_id, real = J.load("test_id"), J.load("test_real")


def load_dir(folder, cellmap=None):
    out = []
    d = CORPUS / folder
    for core in CORES:
        for style in (cellmap or {}):
            name = f"sz_{core}_{style}"
            _, gt = make_wrapper(core, [style], module_name=name)
            for lvl in LEVELS:
                jf = d / f"{name}.{lvl}.json"
                if jf.exists():
                    try:
                        ss, _ = extract_wrapped(jf, gt, name, lvl)
                    except Exception:                          # noqa: BLE001
                        continue
                    for s in ss:
                        if s.label_safe != -1:
                            x = asdict(s)
                            x["cell"] = cellmap[style]
                            out.append(x)
    return out


wrapped = []
for gtf in sorted((CORPUS / "wrapped").glob("*.json")):
    if any(gtf.name.endswith(f".{l}.json") for l in LEVELS):
        continue
    gt = json.loads(gtf.read_text())
    for lvl in LEVELS:
        jf = CORPUS / "wrapped" / f"{gtf.stem}.{lvl}.json"
        if jf.exists():
            try:
                ss, _ = extract_wrapped(jf, gt, gtf.stem, lvl)
            except Exception:                                  # noqa: BLE001
                continue
            wrapped.extend(asdict(s) for s in ss if s.label_safe != -1)

CELL = {"pin_sync2": ("SYNC2", "small"), "sync2_deep": ("SYNC2", "large"),
        "mux_hold_narrow": ("MUX_HOLD", "small"),
        "sync2_held": ("MUX_HOLD", "large")}
grid = load_dir("sizeidiom", CELL)

# ------------------------------------------------------------ D. train
J.banner("D. Training baseline vs P2 (3 seeds each)")
RESULTS = {}
for tag, tr in (("baseline", base_train), ("p2", p2_train)):
    pr = {k: [] for k in ("id", "re", "wr", "g", "cf")}
    for sd in SEEDS:
        m = train_readout(tr, val, readout="mean_seed", seed=sd, epochs=40)
        pr["id"].append(np.array(predict_readout(m, test_id)))
        pr["re"].append(np.array(predict_readout(m, real)))
        pr["wr"].append(np.array(predict_readout(m, wrapped)))
        pr["g"].append(np.array(predict_readout(m, grid)))
        pr["cf"].append(np.array(predict_readout(m, conf)))
    e = {k: np.mean(v, axis=0) > 0.5 for k, v in pr.items()}

    y_wr = J.y_of(wrapped)
    g_cell = np.array([f"{s['cell'][0]}/{s['cell'][1]}" for s in grid])
    RESULTS[tag] = dict(
        test_id_f1=J.f1(e["id"], J.y_of(test_id)),
        test_real=J.metrics(e["re"], J.y_of(real)),
        wrapped_safe_fpr=float(e["wr"][~y_wr].mean()),
        wrapped_unsafe_recall=float(e["wr"][y_wr].mean()),
        grid={c: float(e["g"][g_cell == c].mean()) for c in sorted(set(g_cell))},
        confirmation_fpr=float(e["cf"].mean()),
        confirmation_n=len(conf))
    r = RESULTS[tag]
    print(f"\n  --- {tag} ---")
    print(f"    test_id F1            {r['test_id_f1']:.4f}")
    print(f"    test_real F1/FPR      {r['test_real']['f1']:.3f} / "
          f"{r['test_real']['false_positive_rate']:.3f}")
    print(f"    wrapped safe-FPR      {r['wrapped_safe_fpr']:.3f}   "
          f"unsafe-recall {r['wrapped_unsafe_recall']:.3f}")
    print(f"    grid                  " +
          "  ".join(f"{c}={v:.3f}" for c, v in r["grid"].items()))
    print(f"    CONFIRMATION FPR      {r['confirmation_fpr']:.3f}  "
          f"(n={r['confirmation_n']}, never seen before)")

J.banner("E. Did P2 work?")
b, p = RESULTS["baseline"], RESULTS["p2"]
print(f"  {'metric':26s} {'baseline':>9s} {'P2':>9s} {'change':>9s}")
for label, key in (("test_id F1 (no regress)", None),
                   ("wrapped safe-FPR", "wrapped_safe_fpr"),
                   ("wrapped unsafe-recall", "wrapped_unsafe_recall"),
                   ("CONFIRMATION FPR", "confirmation_fpr")):
    if key is None:
        print(f"  {label:26s} {b['test_id_f1']:9.4f} {p['test_id_f1']:9.4f} "
              f"{p['test_id_f1'] - b['test_id_f1']:+9.4f}")
    else:
        print(f"  {label:26s} {b[key]:9.3f} {p[key]:9.3f} {p[key] - b[key]:+9.3f}")
for c in sorted(b["grid"]):
    print(f"  {'grid ' + c:26s} {b['grid'][c]:9.3f} {p['grid'][c]:9.3f} "
          f"{p['grid'][c] - b['grid'][c]:+9.3f}")

J.save("j17_p2_train", dict(freeze=FREEZE, results=RESULTS,
                            ctx_mean_nodes=float(cn.mean()),
                            confirmation_designs=sorted(
                                {s["design"] for s in conf})))
