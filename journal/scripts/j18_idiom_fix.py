"""P2 redefined: does adding handshake idioms close the gap to real CDC code?

Four training arms, each evaluated on the same held-out data:

  baseline   the original generated training split
  ctx        + context-rich templates (j17, mean 420 nodes)
  idiom      + the five handshake protocol pairs (gen_idioms.py)
  both       + context and idioms

Confirmation set
----------------
j17 used 11 crossings from 6 designs, all OpenTitan, because
`netlist_graph.parse_netlist` finds a clock only if the port name starts with
`clk` and only OpenTitan satisfies that. With structural clock detection all
sixteen found designs yield crossings, so the set is rebuilt and split:

  FRESH  the 10 designs never scored by any model (axis, wb2axip, all pulp)
  USED   the 6 OpenTitan designs, scored once in j17, reported separately

The FRESH column is the one that carries evidential weight. All crossings are
labeled safe on the stated assumption that these libraries implement CDC
correctly, so every positive is a false positive.

Run:  python journal/scripts/j18_idiom_fix.py
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from gen_p2 import RTLGeneratorP2                             # noqa: E402
from gen_idioms import IdiomGenerator, ALL_IDIOMS             # noqa: E402
from grouping import group_crossings_structural               # noqa: E402
from corpus_tools import parse_netlist_structural             # noqa: E402
from readouts import train_readout, predict_readout           # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.rtl_generator import write_design           # noqa: E402
from crossexplain.synthesize import synthesize                # noqa: E402
from crossexplain.netlist_graph import (extract_samples, find_capture_ffs,  # noqa: E402
                                        extract_subgraph, node_features,
                                        CrossingSample)

CORPUS = J.JROOT / "corpus"
SEEDS = [0, 1, 2]
LEVELS_TRAIN = ["O0", "O2"]
USED_PREFIX = "ot_prim_"          # scored once already in j17


def build(gen_fn, outdir, n, tag):
    outdir.mkdir(parents=True, exist_ok=True)
    got = []
    for sd in range(n):
        spec = gen_fn(sd)
        write_design(spec, outdir)
        for lvl in LEVELS_TRAIN:
            jf = outdir / f"{spec.name}.{lvl}.json"
            if not jf.exists() and not synthesize(outdir / f"{spec.name}.v",
                                                  jf, lvl):
                continue
            try:
                ss, _ = extract_samples(jf, spec.ground_truth(), spec.name, lvl)
            except Exception:                                  # noqa: BLE001
                continue
            got.extend(asdict(s) for s in ss if s.label_safe != -1)
    y = J.y_of(got)
    nd = np.array([len(s["node_feats"]) for s in got])
    print(f"  {tag:8s} {len(got):5d} samples  safe {int((~y).sum()):4d} / "
          f"unsafe {int(y.sum()):4d}  mean {nd.mean():5.0f} nodes")
    return got


J.banner("A. Building the two augmentations")


def _ctx(sd):
    g = RTLGeneratorP2(seed=5000 + sd)
    return g.generate_ctx(f"p2_{sd:03d}", n_clocks=g.rng.choice([2, 3]),
                          n_crossings=g.rng.choice([3, 4]))


def _idiom(sd):
    g = IdiomGenerator(seed=8000 + sd)
    return g.generate_idioms(f"idm_{sd:03d}", n_clocks=g.rng.choice([2, 3]),
                             n_crossings=g.rng.choice([3, 4]),
                             templates=ALL_IDIOMS)


ctx = build(_ctx, CORPUS / "p2designs", 60, "context")
idm = build(_idiom, CORPUS / "idiomdesigns", 60, "idioms")

base_train, val = J.load("train"), J.load("val")
ARMS = {"baseline": base_train, "ctx": base_train + ctx,
        "idiom": base_train + idm, "both": base_train + ctx + idm}
FREEZE = datetime.now(timezone.utc).isoformat()
print(f"\n  GENERATORS FROZEN at {FREEZE}")

# ------------------------------------------------- confirmation set
J.banner("B. Confirmation set from all 16 found designs")
triage = json.loads((J.JROOT / "results" / "j12_corpus_triage.json").read_text())
found = [r for r in triage if r.get("usable") and "duplicate_of" not in r]
conf = []
for r in found:
    jf = CORPUS / "netlists" / f"{r['name']}.O0.json"
    if not jf.exists():
        continue
    nl = parse_netlist_structural(jf)
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
        d = asdict(CrossingSample(
            design=r["name"], level="O0", cid=-1, nodes=nodes,
            node_feats=feats, edges=edges, seeds=[local[f] for f in grp],
            loc_mask=[1 if g in cone else 0 for g in nodes],
            label_safe=1, label_class=-1, template="FOUND_SAFE",
            src_dom=min(src_doms) if src_doms else -1, dst_dom=dst))
        d["fresh"] = not r["name"].startswith(USED_PREFIX)
        conf.append(d)

fresh = np.array([s["fresh"] for s in conf])
cn = np.array([len(s["node_feats"]) for s in conf])
print(f"  {len(conf)} crossings from "
      f"{len({s['design'] for s in conf})} designs")
print(f"  FRESH (never scored): {int(fresh.sum())} crossings from "
      f"{len({s['design'] for s in conf if s['fresh']})} designs")
print(f"  USED  (j17, OpenTitan): {int((~fresh).sum())} crossings from "
      f"{len({s['design'] for s in conf if not s['fresh']})} designs")
print(f"  nodes mean {cn.mean():.0f}, range {cn.min()}-{cn.max()}")
print("  all labeled safe: any positive is a false positive")

test_id, real = J.load("test_id"), J.load("test_real")
y_id, y_re = J.y_of(test_id), J.y_of(real)
u_re = J.units(real)

J.banner("C. Training four arms x 3 seeds")
OUT = {}
for tag, tr in ARMS.items():
    pid, pre, pcf = [], [], []
    for sd in SEEDS:
        m = train_readout(tr, val, readout="mean_seed", seed=sd, epochs=40)
        pid.append(np.array(predict_readout(m, test_id)))
        pre.append(np.array(predict_readout(m, real)))
        pcf.append(np.array(predict_readout(m, conf)))
    eid = np.mean(pid, axis=0) > 0.5
    ere = np.mean(pre, axis=0) > 0.5
    ecf = np.mean(pcf, axis=0) > 0.5
    OUT[tag] = dict(
        n_train=len(tr),
        test_id_f1=J.f1(eid, y_id),
        test_real=J.metrics(ere, y_re),
        test_real_ci=J.clustered_ci(ere, y_re, u_re["design"], "f1", B=2000),
        conf_fpr_all=float(ecf.mean()),
        conf_fpr_fresh=float(ecf[fresh].mean()),
        conf_fpr_used=float(ecf[~fresh].mean()),
        per_design={d: float(ecf[[s["design"] == d for s in conf]].mean())
                    for d in sorted({s["design"] for s in conf})})
    r = OUT[tag]
    print(f"\n  --- {tag} (train n={len(tr)}) ---")
    print(f"    test_id F1        {r['test_id_f1']:.4f}")
    print(f"    test_real F1/FPR  {r['test_real']['f1']:.3f} / "
          f"{r['test_real']['false_positive_rate']:.3f}")
    print(f"    confirmation FPR  all {r['conf_fpr_all']:.3f}   "
          f"FRESH {r['conf_fpr_fresh']:.3f}   used {r['conf_fpr_used']:.3f}")

J.banner("D. Did the idioms close the gap?")
print(f"  {'arm':10s} {'test_id':>8s} {'real F1':>8s} {'real FPR':>9s} "
      f"{'conf FRESH':>11s} {'conf used':>10s}")
for tag, r in OUT.items():
    print(f"  {tag:10s} {r['test_id_f1']:8.4f} {r['test_real']['f1']:8.3f} "
          f"{r['test_real']['false_positive_rate']:9.3f} "
          f"{r['conf_fpr_fresh']:11.3f} {r['conf_fpr_used']:10.3f}")
b = OUT["baseline"]
for tag in ("ctx", "idiom", "both"):
    d = OUT[tag]["conf_fpr_fresh"] - b["conf_fpr_fresh"]
    print(f"  {tag:10s} change on FRESH confirmation: {d:+.3f}")

print("\n  per-design FPR on the FRESH designs (baseline -> idiom):")
for d in sorted({s["design"] for s in conf if s["fresh"]}):
    print(f"    {d:30s} {b['per_design'][d]:.3f} -> "
          f"{OUT['idiom']['per_design'][d]:.3f}")

J.save("j18_idiom_fix", dict(frozen=FREEZE, arms=OUT,
                             n_conf=len(conf),
                             n_fresh=int(fresh.sum())))
