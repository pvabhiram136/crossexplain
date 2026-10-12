"""Separate neighbourhood size from crossing idiom.

The first wrapped run (j14) could not distinguish two explanations for the
safe-side false positives, because the two safe templates had disjoint size
ranges: every SYNC2 sample was 14 nodes, every MUX_HOLD was 72 or more. Both
"large context causes false positives" and "the qualifier-held idiom causes
false positives" fit those data.

This script crosses the two factors properly. Four cells:

               small subgraph            large subgraph
  SYNC2        pin_sync2   (existing)    sync2_deep       (new)
  MUX_HOLD     mux_hold_narrow (new)     sync2_held       (existing)

`sync2_deep` keeps the two-flop synchronizer but puts a wide registered
datapath in the source domain, so the backward cone grows while the idiom and
the safety argument are unchanged. `mux_hold_narrow` keeps the qualifier-held
bus but captures into wrapper registers rather than core registers, so the
subgraph stays small.

Both new cells are safe by construction, so the whole experiment is about
false positives. If size drives them, the two large cells should be flagged and
the two small ones should not. If idiom drives them, both MUX_HOLD cells should
be flagged and both SYNC2 cells should not.

Run:  python journal/scripts/j15_size_vs_idiom.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from wrappers import make_wrapper, CORES, extract_wrapped     # noqa: E402
from corpus_tools import synthesize_multi, find_blackboxes    # noqa: E402
from strong_matcher import strong_predict                     # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import pattern_matcher_predict    # noqa: E402

CORPUS = J.JROOT / "corpus"
OUT = CORPUS / "sizeidiom"
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = ["O0", "O1", "O2", "O3"]

CELLS = [
    ("SYNC2",    "small", "pin_sync2"),
    ("SYNC2",    "large", "sync2_deep"),
    ("MUX_HOLD", "small", "mux_hold_narrow"),
    ("MUX_HOLD", "large", "sync2_held"),
]
CORE_SRC = {"uart16550": "uart16550", "i2c_master": "i2c",
            "simple_spi": "simple_spi"}


def core_sources(core):
    root = CORPUS / CORE_SRC[core]
    return root, sorted((root / "rtl/verilog").glob("*.v")), [root / "rtl/verilog"]


J.banner("Crossing idiom against neighbourhood size (all four cells safe)")
samples = []
for core in CORES:
    root, csrcs, incs = core_sources(core)
    if not csrcs:
        continue
    for idiom, size, style in CELLS:
        v, gt = make_wrapper(core, [style], module_name=f"sz_{core}_{style}")
        name = gt["design"]
        vf = OUT / f"{name}.v"
        vf.write_text(v, encoding="utf-8", newline="\n")
        for lvl in LEVELS:
            jf = OUT / f"{name}.{lvl}.json"
            ok, msg = synthesize_multi([vf] + csrcs, name, jf, lvl,
                                       incdirs=incs, sv=False, cwd=root)
            if not ok:
                print(f"  {name} @{lvl}: synth failed: {msg[:70]}")
                continue
            if find_blackboxes(json.loads(jf.read_text())):
                print(f"  {name} @{lvl}: blackboxed")
                continue
            ss, _ = extract_wrapped(jf, gt, name, lvl)
            for s in ss:
                if s.label_safe == -1:
                    continue
                d = asdict(s)
                d["cell_idiom"], d["cell_size"], d["core"] = idiom, size, core
                samples.append(d)

if not samples:
    print("  no samples produced")
    sys.exit(1)

y = J.y_of(samples)
nodes = np.array([len(s["node_feats"]) for s in samples])
print(f"\n  {len(samples)} samples, all safe: {bool((~y).all())}")

models = J.load_seed_models()
ens = J.seed_probs(models, samples).mean(0) > 0.5
pm = np.array([p == 0 for p in pattern_matcher_predict(samples)])
sm = np.array([p == 0 for p in strong_predict(samples, gray_xor_ratio=0.70)])

idiom = np.array([s["cell_idiom"] for s in samples])
size = np.array([s["cell_size"] for s in samples])

J.banner("The 2x2: false-positive rate on safe crossings")
print(f"  {'idiom':10s} {'size':6s} {'n':>3s} {'nodes':>7s}   "
      f"{'CrossExplain':>12s} {'textbook':>9s} {'domain':>7s}")
grid = {}
for i in ["SYNC2", "MUX_HOLD"]:
    for s_ in ["small", "large"]:
        m = (idiom == i) & (size == s_)
        if not m.sum():
            continue
        cell = dict(n=int(m.sum()), mean_nodes=float(nodes[m].mean()),
                    ens_fpr=float(ens[m].mean()),
                    textbook_fpr=float(pm[m].mean()),
                    domain_fpr=float(sm[m].mean()))
        grid[f"{i}/{s_}"] = cell
        print(f"  {i:10s} {s_:6s} {int(m.sum()):3d} {nodes[m].mean():7.0f}   "
              f"{ens[m].mean():12.3f} {pm[m].mean():9.3f} {sm[m].mean():7.3f}")

J.banner("Which factor explains the false positives?")
by_size = {s_: float(ens[size == s_].mean()) for s_ in ["small", "large"]}
by_idiom = {i: float(ens[idiom == i].mean()) for i in ["SYNC2", "MUX_HOLD"]}
print(f"  marginal FPR by size : small {by_size['small']:.3f}"
      f"   large {by_size['large']:.3f}"
      f"   (spread {abs(by_size['large'] - by_size['small']):.3f})")
print(f"  marginal FPR by idiom: SYNC2 {by_idiom['SYNC2']:.3f}"
      f"   MUX_HOLD {by_idiom['MUX_HOLD']:.3f}"
      f"   (spread {abs(by_idiom['MUX_HOLD'] - by_idiom['SYNC2']):.3f})")

ds, di = (abs(by_size["large"] - by_size["small"]),
          abs(by_idiom["MUX_HOLD"] - by_idiom["SYNC2"]))
verdict = ("SIZE dominates" if ds > di + 0.15 else
           "IDIOM dominates" if di > ds + 0.15 else
           "neither factor dominates cleanly")
print(f"\n  verdict: {verdict}")

# does size matter *within* an idiom? that is the clean contrast
print("\n  within-idiom size effect (the controlled comparison):")
for i in ["SYNC2", "MUX_HOLD"]:
    a = grid.get(f"{i}/small"), grid.get(f"{i}/large")
    if all(a):
        print(f"    {i:10s} small {a[0]['ens_fpr']:.3f} "
              f"({a[0]['mean_nodes']:.0f} nodes) -> "
              f"large {a[1]['ens_fpr']:.3f} ({a[1]['mean_nodes']:.0f} nodes)"
              f"   delta {a[1]['ens_fpr'] - a[0]['ens_fpr']:+.3f}")

J.save("j15_size_vs_idiom", dict(
    n_samples=len(samples), all_safe=bool((~y).all()), grid=grid,
    marginal_by_size=by_size, marginal_by_idiom=by_idiom,
    delta_size=ds, delta_idiom=di, verdict=verdict,
    samples=[dict(design=s["design"], level=s["level"], core=s["core"],
                  idiom=s["cell_idiom"], size=s["cell_size"],
                  n_nodes=len(s["node_feats"]), label_safe=s["label_safe"])
             for s in samples]))
