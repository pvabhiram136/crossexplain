"""P1 option 3 - build and validate the wrapped-core corpus.

Generates two-clock wrappers around the three real single-clock Wishbone cores
that triage found (uart16550, i2c_master, simple_spi), synthesizes each at all
four optimization levels, and checks that the authored crossings are actually
discovered and correctly labeled.

The point is not merely to raise the design count. It is to obtain crossings of
*known* safety sitting inside hundreds of flip-flops of real peripheral logic,
which is the only way to test P0 finding (d): that the sim-to-real gap comes
from safe crossings embedded in wide busy logic rather than from size alone.

These designs are semi-synthetic and must be reported as such. See
journal/wrappers.py for the honesty boundary.

Run:  python journal/scripts/j13_wrapper_corpus.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from wrappers import make_wrapper, CORES, extract_wrapped     # noqa: E402
from corpus_tools import synthesize_multi, find_blackboxes    # noqa: E402

sys.path.insert(0, str(J.ROOT))


CORPUS = J.JROOT / "corpus"
OUT = CORPUS / "wrapped"
OUT.mkdir(parents=True, exist_ok=True)

# Where each core's sources live, matching the triage manifest.
CORE_SRC = {
    "uart16550": ("uart16550", ["rtl/verilog"], ["rtl/verilog"]),
    "i2c_master": ("i2c", ["rtl/verilog"], ["rtl/verilog"]),
    "simple_spi": ("simple_spi", ["rtl/verilog"], ["rtl/verilog"]),
}

# Each combination gives one safe and one unsafe crossing where possible, so
# every wrapped design contributes to both classes.
COMBOS = [
    ["sync2_held", "pin_sync2"],     # both safe
    ["bus_nosync", "pin_nosync"],    # both unsafe
    ["sync2_held", "pin_nosync"],    # mixed
    ["bus_nosync", "pin_sync2"],     # mixed
]
LEVELS = ["O0", "O1", "O2", "O3"]


def core_sources(core):
    repo, srcdirs, incdirs = CORE_SRC[core]
    root = CORPUS / repo
    srcs = []
    for d in srcdirs:
        srcs.extend(sorted((root / d).glob("*.v")))
    return root, srcs, [root / d for d in incdirs]


J.banner("P1 option 3: two-clock wrappers around real peripheral cores")
print(f"  {len(CORES)} cores x {len(COMBOS)} crossing combinations "
      f"x {len(LEVELS)} optimization levels")

rows, all_samples = [], []
for core in CORES:
    root, csrcs, incs = core_sources(core)
    if not csrcs:
        print(f"  {core:14s} SKIP: core sources not found under {root}")
        continue
    for combo in COMBOS:
        v, gt = make_wrapper(core, combo)
        name = gt["design"]
        vfile = OUT / f"{name}.v"
        vfile.write_text(v, encoding="utf-8", newline="\n")
        (OUT / f"{name}.json").write_text(json.dumps(gt, indent=1),
                                          encoding="utf-8")
        row = dict(design=name, core=core, styles=combo,
                   expected=len(gt["crossings"]))
        per_level = {}
        for lvl in LEVELS:
            jf = OUT / f"{name}.{lvl}.json"
            ok, msg = synthesize_multi([vfile] + csrcs, name, jf, lvl,
                                       incdirs=incs, sv=False, cwd=root)
            if not ok:
                per_level[lvl] = dict(status="synth failed", detail=msg)
                continue
            data = json.loads(jf.read_text())
            bb = find_blackboxes(data)
            if bb:
                per_level[lvl] = dict(status="blackboxed", detail=str(bb))
                continue
            try:
                samples, nl = extract_wrapped(jf, gt, name, lvl)
            except Exception as e:                             # noqa: BLE001
                per_level[lvl] = dict(status="extract failed", detail=str(e))
                continue
            labeled = [s for s in samples if s.label_safe != -1]
            per_level[lvl] = dict(
                status="ok", n_cells=len(data["modules"][name]["cells"])
                if name in data["modules"] else None,
                found=len(labeled),
                templates=sorted({s.template for s in labeled}),
                mean_nodes=(round(sum(len(s.node_feats) for s in labeled)
                                  / max(1, len(labeled)), 1)),
                clocks=nl.clocks)
            all_samples.extend(
                dict(design=name, level=lvl, cid=s.cid, template=s.template,
                     label_safe=s.label_safe, n_nodes=len(s.node_feats),
                     n_edges=len(s.edges), n_seeds=len(s.seeds),
                     semi_synthetic=True)
                for s in labeled)
        row["levels"] = per_level
        ok_lv = [l for l, r in per_level.items() if r["status"] == "ok"]
        found = per_level[ok_lv[0]]["found"] if ok_lv else 0
        mn = per_level[ok_lv[0]]["mean_nodes"] if ok_lv else 0
        rows.append(row)
        flag = "OK " if len(ok_lv) == len(LEVELS) else f"{len(ok_lv)}/4"
        print(f"  {name:44s} {flag}  crossings {found}/{row['expected']}"
              f"  mean nodes {mn}")

J.banner("Validation")
good = [r for r in rows
        if all(v["status"] == "ok" for v in r["levels"].values())
        and r["levels"]["O0"]["found"] == r["expected"]]
print(f"  designs built            : {len(rows)}")
print(f"  clean at all four levels : {len(good)}")
print(f"  labeled crossing samples : {len(all_samples)}")

if all_samples:
    nodes = [s["n_nodes"] for s in all_samples]
    safe = [s for s in all_samples if s["label_safe"] == 1]
    unsafe = [s for s in all_samples if s["label_safe"] == 0]
    ns = sorted(nodes)
    print(f"\n  safe / unsafe samples    : {len(safe)} / {len(unsafe)}")
    print(f"  subgraph nodes: median {ns[len(ns) // 2]}, "
          f"min {min(nodes)}, max {max(nodes)}")
    print(f"  mean {sum(nodes) / len(nodes):.1f} nodes "
          f"(train split mean was 29.1, test_real 43.0)")
    big = sum(1 for n in nodes if n > 43)
    print(f"  samples larger than the old real-split mean: {big}/{len(nodes)}")

J.save("j13_wrapper_corpus", dict(designs=rows, samples=all_samples))
