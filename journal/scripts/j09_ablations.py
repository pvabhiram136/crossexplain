"""P0.8 - hop-count and feature-group ablations.

Two reviewer questions the paper cannot currently answer: why six backward
hops, and which of the 27 node features carry the signal.

Feature groups are ablated by zeroing the group in both training and
evaluation, which keeps the tensor shape and so keeps every configuration
comparable. Hop counts require genuinely re-extracting the subgraphs from the
netlists, since the distance-to-seed feature and the induced edge set both
depend on the extraction radius.

Expect the generated splits to stay saturated throughout. Per finding G, that
is the point: the interesting column is test_real, and the interesting result
would be a feature group whose removal does NOT hurt.

Run:  python journal/scripts/j09_ablations.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.netlist_graph import CELL_VOCAB, extract_samples            # noqa: E402
import crossexplain.netlist_graph as NG                                       # noqa: E402
from crossexplain.train import train_model, predict                           # noqa: E402

NV = len(CELL_VOCAB)
FEATURE_GROUPS = {
    "cell_type_onehot": list(range(0, NV)),
    "is_ff_flag": [NV],
    "domain_flags": [NV + 1, NV + 2, NV + 3],
    "fanin_fanout": [NV + 4, NV + 5],
    "seed_flag": [NV + 6],
    "dist_to_seed": [NV + 7],
}
SEEDS_FEAT = [0, 1, 2]
SEEDS_HOP = [0, 1]
HOPS_BWD = [2, 4, 6, 8]


def zero_groups(samples, cols):
    if not cols:
        return samples
    out = []
    for s in samples:
        t = dict(s)
        f = np.asarray(s["node_feats"], dtype=np.float32).copy()
        f[:, cols] = 0.0
        t["node_feats"] = f.tolist()
        out.append(t)
    return out


def _extract_job(args):
    jfile, gt, design, level, hb, hf = args
    orig = NG.extract_subgraph

    def patched(nl, seeds, hops_fwd=hf, hops_bwd=hb, cid=None):
        return orig(nl, seeds, hops_fwd=hf, hops_bwd=hb, cid=cid)

    NG.extract_subgraph = patched
    try:
        samples, _ = extract_samples(Path(jfile), gt, design, level)
        return [asdict(s) for s in samples]
    except Exception:                                          # noqa: BLE001
        return []
    finally:
        NG.extract_subgraph = orig


def rebuild(hb, hf=3, workers=6):
    """Re-extract every split at a given extraction radius."""
    out = {}
    for split, levels in [("train", ["O0", "O2"]), ("val", ["O0", "O2"]),
                          ("test_id", ["O0", "O1", "O2", "O3"]),
                          ("test_real", ["O0", "O1", "O2", "O3"])]:
        ddir = J.ROOT / "data" / "designs" / split
        jobs = []
        for gtf in sorted(ddir.glob("*.json")):
            if any(gtf.name.endswith(f".{l}.json") for l in
                   ["O0", "O1", "O2", "O3"]):
                continue
            gt = json.loads(gtf.read_text())
            d = gtf.stem
            for lvl in levels:
                jf = ddir / f"{d}.{lvl}.json"
                if jf.exists():
                    jobs.append((str(jf), gt, d, lvl, hb, hf))
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_extract_job, jobs, chunksize=4):
                rows.extend(r)
        out[split] = [s for s in rows if s["label_safe"] != -1]
    return out


def evaluate(train_s, val_s, id_s, real_s, seeds, tag):
    pid, preal = [], []
    for sd in seeds:
        m = train_model(train_s, val_s, arch="gat", layers=5, epochs=40,
                        verbose=False, seed=sd)
        pid.append(np.array(predict(m, id_s)[0]))
        preal.append(np.array(predict(m, real_s)[0]))
    y_id, y_re = J.y_of(id_s), J.y_of(real_s)
    u = J.units(real_s)
    ens_re = np.mean(preal, axis=0)
    return {
        "tag": tag,
        "test_id_f1_mean": float(np.mean([J.f1(p > 0.5, y_id) for p in pid])),
        "test_real_f1_mean": float(np.mean([J.f1(p > 0.5, y_re) for p in preal])),
        "test_real_f1_std": float(np.std([J.f1(p > 0.5, y_re) for p in preal])),
        "test_real_f1_ens": J.f1(ens_re > 0.5, y_re),
        "test_real_ci_design": J.clustered_ci(ens_re > 0.5, y_re, u["design"],
                                              "f1", B=1500),
        "n_train": len(train_s),
    }


def main():
    t0 = time.time()
    OUT = {}

    # ------------------------------------------------ feature-group ablation
    J.banner("P0.8a  Feature-group ablation (3 seeds each, no re-extraction)")
    train_s, val_s = J.load("train"), J.load("val")
    id_s, real_s = J.load("test_id"), J.load("test_real")

    rows = [evaluate(train_s, val_s, id_s, real_s, SEEDS_FEAT, "full (27 features)")]
    for name, cols in FEATURE_GROUPS.items():
        rows.append(evaluate(
            zero_groups(train_s, cols), zero_groups(val_s, cols),
            zero_groups(id_s, cols), zero_groups(real_s, cols),
            SEEDS_FEAT, f"minus {name}"))
        r = rows[-1]
        print(f"  {r['tag']:28s} test_id={r['test_id_f1_mean']:.4f}  "
              f"real={r['test_real_f1_mean']:.3f}+-{r['test_real_f1_std']:.3f}  "
              f"real_ens={r['test_real_f1_ens']:.3f}")
    base = rows[0]
    print(f"\n  {base['tag']:28s} test_id={base['test_id_f1_mean']:.4f}  "
          f"real={base['test_real_f1_mean']:.3f}+-{base['test_real_f1_std']:.3f}  "
          f"real_ens={base['test_real_f1_ens']:.3f}")
    OUT["feature_groups"] = rows

    # ------------------------------------------------------- hop ablation
    J.banner("P0.8b  Extraction-radius ablation (re-extracts, 2 seeds each)")
    hop_rows = []
    for hb in HOPS_BWD:
        t1 = time.time()
        d = rebuild(hb)
        r = evaluate(d["train"], d["val"], d["test_id"], d["test_real"],
                     SEEDS_HOP, f"bwd={hb}, fwd=3")
        mean_nodes = float(np.mean([len(s["node_feats"]) for s in d["train"]]))
        r["mean_train_nodes"] = mean_nodes
        r["rebuild_sec"] = round(time.time() - t1, 1)
        hop_rows.append(r)
        print(f"  {r['tag']:16s} nodes={mean_nodes:5.1f}  "
              f"test_id={r['test_id_f1_mean']:.4f}  "
              f"real={r['test_real_f1_mean']:.3f}+-{r['test_real_f1_std']:.3f}  "
              f"real_ens={r['test_real_f1_ens']:.3f}  ({r['rebuild_sec']}s)")
    OUT["hops"] = hop_rows

    J.banner("Summary")
    worst = min(OUT["feature_groups"][1:], key=lambda r: r["test_real_f1_ens"])
    best_hop = max(hop_rows, key=lambda r: r["test_real_f1_ens"])
    print(f"  most load-bearing feature group: {worst['tag']} "
          f"(real ens {worst['test_real_f1_ens']:.3f} vs "
          f"{base['test_real_f1_ens']:.3f} full)")
    print(f"  best extraction radius on real: {best_hop['tag']} "
          f"(ens {best_hop['test_real_f1_ens']:.3f}); paper uses bwd=6")
    print(f"  total wall time {time.time() - t0:.0f}s")
    J.save("j09_ablations", OUT)


if __name__ == "__main__":
    main()
