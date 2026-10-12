"""j30 - the dispersion distribution, with enough draws to describe its shape.

j24 and j28 established that dispersion under mathematically-null change exists
and bounded its magnitude. Both used one run per cell, which is why neither
says anything about the SHAPE of the distribution, and why the apparent
asymmetry between architectures could not be discussed.

This runs 60 draws per architecture. A draw is one combination of two null
axes: which of the two disjoint scatter-add reductions runs first, and an
isomorphic relabeling of the graph nodes. Both are verified no-ops --- the
relabeling changes the detection logit by at most 1.5e-08 over 256 probe
samples and changes no decision (j38). Thread count is held at 1 throughout,
so every process is
single-threaded and several can run at once without interfering.

  3 architectures x 2 orderings x 30 relabelings = 180 trainings, 60 per arch.

Note on what "more evidence" means here. Every cell is a DISTINCT null
configuration, and each is individually deterministic -- j30_worker.py
reproduces a given cell bit for bit. Repeating a cell therefore yields the
identical model and adds nothing. The only way to enlarge this evidence is to
enumerate more of the space, which is what raising the relabeling count does.

A caution kept in front of the numbers: these draws enumerate a space of
implementation choices, they are not a random sample from a population. The
spread below is therefore descriptive. What it can support is a claim that the
spread exists and is large; what it cannot support on its own is a parametric
claim about the distribution that generated it.
"""
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

JROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

ARCHS = ["gat", "gcn", "sage"]
POOLS = ["global_first", "seed_first"]
PERMS = ["none"] + [str(i) for i in range(1, 30)]
MAX_PARALLEL = 4               # RAM-bound, not core-bound
OUTDIR = J.RESULTS / "j30"


def run_grid():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cells = list(itertools.product(ARCHS, POOLS, PERMS))
    todo = [c for c in cells
            if not (OUTDIR / f"{c[0]}_{c[1]}_perm{c[2]}.json").exists()]
    print(f"  {len(cells)} cells, {len(cells) - len(todo)} already done, "
          f"{len(todo)} to run, {MAX_PARALLEL} at a time")

    running, done, t0 = [], 0, time.time()
    while todo or running:
        while todo and len(running) < MAX_PARALLEL:
            arch, pool, perm = todo.pop(0)
            p = subprocess.Popen(
                [sys.executable, str(SCRIPTS / "j30_worker.py"),
                 arch, pool, perm],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            running.append((p, f"{arch}/{pool}/perm{perm}"))
        time.sleep(5)
        still = []
        for p, name in running:
            if p.poll() is None:
                still.append((p, name))
                continue
            done += 1
            mins = (time.time() - t0) / 60
            if p.returncode != 0:
                err = p.stderr.read().decode(errors="replace")[-400:]
                print(f"  [{done:2d}/{len(cells)}] FAILED {name}\n{err}",
                      flush=True)
            else:
                print(f"  [{done:2d}/{len(cells)}] {name}  "
                      f"({mins:.0f} min elapsed)", flush=True)
        running = still


def load_rows():
    rows = []
    for f in sorted(OUTDIR.glob("*.json")):
        rows.append(json.loads(f.read_text()))
    return rows


def describe(v):
    v = np.asarray(v, dtype=float)
    return dict(n=len(v), min=float(v.min()), max=float(v.max()),
                range=float(v.max() - v.min()), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                median=float(np.median(v)),
                q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)),
                n_distinct=int(len(np.unique(np.round(v, 6)))),
                n_low=int((v < 0.10).sum()),
                n_mid=int(((v >= 0.10) & (v <= 0.90)).sum()),
                n_high=int((v > 0.90).sum()))


def main():
    J.banner("j30  dispersion distribution, 20 draws per architecture")
    run_grid()

    rows = load_rows()
    if len(rows) < len(ARCHS) * len(POOLS) * len(PERMS):
        print(f"\n  WARNING: {len(rows)} of "
              f"{len(ARCHS) * len(POOLS) * len(PERMS)} cells present. "
              f"Re-run to fill the rest.")

    J.banner("Every draw")
    print(f"  {'arch':>5s} {'pooling':>13s} {'perm':>5s} {'test_id':>8s} "
          f"{'real':>7s} {'wr.rec':>7s} {'FPR':>6s} {'med p':>7s}")
    for r in sorted(rows, key=lambda r: (r["arch"], r["pooling"],
                                         str(r["node_perm"]))):
        print(f"  {r['arch']:>5s} {r['pooling']:>13s} "
              f"{str(r['node_perm']):>5s} {r['test_id_f1']:8.4f} "
              f"{r['test_real_f1']:7.4f} {r['wrapped_recall']:7.3f} "
              f"{r['conf_fpr']:6.3f} {r['conf_median_p']:7.4f}")

    all_f1 = [r["test_id_f1"] for r in rows]
    J.banner("Is the benchmark constant?")
    d = describe(all_f1)
    print(f"  test_id F1 over all {d['n']} draws: "
          f"{d['min']:.4f} to {d['max']:.4f}, "
          f"{d['n_distinct']} distinct value(s)")

    J.banner("Real-CDC false-positive rate, per architecture")
    print(f"  {'arch':>5s} {'n':>3s} {'min':>6s} {'max':>6s} {'range':>6s} "
          f"{'mean':>6s} {'sd':>6s} {'<0.1':>5s} {'mid':>4s} {'>0.9':>5s}")
    per_arch = {}
    for a in ARCHS:
        v = [r["conf_fpr"] for r in rows if r["arch"] == a]
        if not v:
            continue
        s = describe(v)
        per_arch[a] = s
        print(f"  {a:>5s} {s['n']:3d} {s['min']:6.3f} {s['max']:6.3f} "
              f"{s['range']:6.3f} {s['mean']:6.3f} {s['sd']:6.3f} "
              f"{s['n_low']:5d} {s['n_mid']:4d} {s['n_high']:5d}")

    pooled = describe([r["conf_fpr"] for r in rows])
    print(f"\n  pooled over all {pooled['n']} draws: "
          f"{pooled['min']:.3f} to {pooled['max']:.3f}, "
          f"sd {pooled['sd']:.3f}, "
          f"{pooled['n_low']} low / {pooled['n_mid']} mid / "
          f"{pooled['n_high']} high")

    J.banner("Shape")
    frac_extreme = (pooled["n_low"] + pooled["n_high"]) / pooled["n"]
    print(f"  draws at an extreme (<0.1 or >0.9): "
          f"{frac_extreme:.0%} ({pooled['n_low'] + pooled['n_high']} of "
          f"{pooled['n']})")
    if frac_extreme >= 0.8:
        print("  -> the distribution is essentially bimodal: a model either "
              "clears real CDC code or flags nearly all of it, with little "
              "in between")
    else:
        print("  -> the distribution is not concentrated at the extremes; "
              "intermediate behaviour is common")

    J.save("j30_multi_dispersion", dict(
        archs=ARCHS, pools=POOLS, perms=PERMS,
        n_draws=len(rows), rows=rows,
        test_id=describe(all_f1), per_arch=per_arch, pooled=pooled,
        frac_extreme=frac_extreme))


if __name__ == "__main__":
    main()
