"""Tighten the two diagnostic claims before drafting.

1. The nearest-neighbour result was computed on one trained seed. Here it is
   repeated across all five RELEASED checkpoints (`results/seed_models.pt`),
   which is the artifact the paper actually ships, so the 155-of-155 unanimity
   is reported over seeds rather than assumed stable.

2. Every confirmation false-positive rate is given a design-clustered
   percentile interval, so section VIII carries uncertainty. Clustering is by
   design because crossings inside one library are not independent.

All models here are released checkpoints or fitted baselines; nothing is
retrained, so these numbers correspond to the shipped artifact.

Run:  python journal/scripts/j21_tighten.py
"""
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from grouping import group_crossings_structural               # noqa: E402
from corpus_tools import parse_netlist_structural             # noqa: E402
from strong_matcher import strong_predict                     # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import (pattern_matcher_predict, FlatML,  # noqa: E402
                                    flat_features)
from crossexplain.models import CrossExplainNet, collate, _agg_mean  # noqa: E402
from crossexplain.train import predict                        # noqa: E402
from crossexplain.netlist_graph import (find_capture_ffs, extract_subgraph,  # noqa: E402
                                        node_features, CrossingSample)

CORPUS = J.JROOT / "corpus"
RES = J.ROOT / "results"

# ------------------------------------------------------ confirmation set
J.banner("Confirmation set")
triage = json.loads((J.JROOT / "results" / "j12_corpus_triage.json").read_text())
found = [r for r in triage if r.get("usable") and "duplicate_of" not in r]
conf = []
for r in found:
    jf = CORPUS / "netlists" / f"{r['name']}.O0.json"
    if not jf.exists():
        continue
    nl = parse_netlist_structural(jf)
    caps = find_capture_ffs(nl)
    for grp in (group_crossings_structural(nl, caps) if caps else []):
        src_doms = set()
        for f in grp:
            src_doms |= caps[f][0]
        dst = nl.domain[grp[0]]
        nodes, edges, local = extract_subgraph(nl, grp, cid=None)
        cone = set(grp)
        for f in grp:
            sf, cb = nl.d_srcs[f]
            cone |= sf | cb
        conf.append(asdict(CrossingSample(
            design=r["name"], level="O0", cid=-1, nodes=nodes,
            node_feats=node_features(nl, nodes, set(grp), src_doms, dst, edges),
            edges=edges, seeds=[local[f] for f in grp],
            loc_mask=[1 if g in cone else 0 for g in nodes],
            label_safe=1, label_class=-1, template="FOUND_SAFE",
            src_dom=min(src_doms) if src_doms else -1, dst_dom=dst)))
u = J.units(conf)
designs = u["design"]
y_safe = np.zeros(len(conf), dtype=bool)     # all safe: positives are errors
print(f"  {len(conf)} crossings, {len(set(designs))} designs")

train = J.load("train")
y_tr = J.y_of(train)

# ============================================ 1. k-NN across released seeds
J.banner("1. Nearest-neighbour analysis across all five released seeds")
models = J.load_seed_models()


@torch.no_grad()
def embed(model, samples, bs=64):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        b = collate(samples[i:i + bs])
        h = torch.relu(model.embed(b["x"]))
        for layer in model.layers:
            h = layer(h, b["ei"])
        mean_pool = _agg_mean(h, b["batch"], b["n_graphs"])
        sm = h * b["seed_mask"].unsqueeze(-1)
        cnt = torch.zeros(b["n_graphs"], 1)
        cnt.index_add_(0, b["batch"], b["seed_mask"].unsqueeze(-1).float())
        sp = torch.zeros(b["n_graphs"], h.size(1))
        sp.index_add_(0, b["batch"], sm)
        out.append(torch.cat([mean_pool, sp / cnt.clamp(min=1.0)], dim=1))
    return torch.cat(out).numpy()


K = 5
per_seed = []
for si, m in enumerate(models):
    Etr, Ecf = embed(m, train), embed(m, conf)
    Etr /= np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-9
    Ecf /= np.linalg.norm(Ecf, axis=1, keepdims=True) + 1e-9
    sim = Ecf @ Etr.T
    unsafe = 0
    tpl = Counter()
    for i in range(len(conf)):
        for j in np.argsort(-sim[i])[:K]:
            unsafe += int(y_tr[j])
            tpl[train[j]["template"]] += 1
    tot = len(conf) * K
    top, topn = tpl.most_common(1)[0]
    per_seed.append(dict(seed=si, unsafe_frac=unsafe / tot,
                         top_template=top, top_frac=topn / tot))
    print(f"  seed {si}: unsafe {unsafe}/{tot} = {unsafe / tot:.1%}   "
          f"top template {top} ({topn / tot:.1%})")

uf = [p["unsafe_frac"] for p in per_seed]
tops = {p["top_template"] for p in per_seed}
print(f"\n  across seeds: unsafe fraction {min(uf):.1%} to {max(uf):.1%}")
print(f"  top template is {'the same on every seed: ' + tops.pop() if len(tops) == 1 else tops}")

# ================================== 2. clustered intervals on every method
J.banner("2. Confirmation FPR with design-clustered 95% intervals")
flat = FlatML(seed=0).fit(train)
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402
from sklearn.pipeline import make_pipeline                       # noqa: E402
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
    np.stack([flat_features(s) for s in train]), y_tr.astype(int))


def load_ckpt(name):
    ck = torch.load(RES / name, map_location="cpu", weights_only=False)
    m = CrossExplainNet(arch=ck["arch"], hidden=ck["hidden"],
                        layers=ck["layers"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


preds = {
    "CrossExplain (5-seed ens)": J.seed_probs(models, conf).mean(0) > 0.5,
    "CrossExplain (released)": np.array(predict(load_ckpt("crossexplain_gat.pt"), conf)[0]) > 0.5,
    "GCN ablation": np.array(predict(load_ckpt("ablation_gcn.pt"), conf)[0]) > 0.5,
    "GraphSAGE ablation": np.array(predict(load_ckpt("ablation_sage.pt"), conf)[0]) > 0.5,
    "single-task ablation": np.array(predict(load_ckpt("ablation_singletask.pt"), conf)[0]) > 0.5,
    "Random Forest": np.array([d == 0 for d in flat.predict(conf)[0]]),
    "Logistic regression": lr.predict(np.stack([flat_features(s) for s in conf])).astype(bool),
    "matcher, textbook": np.array([p == 0 for p in pattern_matcher_predict(conf)]),
    "matcher, domain-aware": np.array([p == 0 for p in strong_predict(conf, gray_xor_ratio=0.70)]),
}

OUT = {}
print(f"  {'method':28s} {'FPR':>6s}  {'95% CI (design-clustered)':>26s}")
for name, p in preds.items():
    ci = J.clustered_ci(p, y_safe, designs, "accuracy", B=5000)
    # accuracy against an all-safe truth equals 1 - FPR, so invert
    fpr, lo, hi = 1 - ci["point"], 1 - ci["ci_hi"], 1 - ci["ci_lo"]
    OUT[name] = dict(fpr=float(fpr), ci_lo=float(lo), ci_hi=float(hi),
                     n_flagged=int(p.sum()), n=len(p))
    print(f"  {name:28s} {fpr:6.3f}  [{lo:.3f}, {hi:.3f}]   "
          f"{int(p.sum())}/{len(p)} flagged")

J.banner("3. Is the rule engine's advantage resolvable?")
best_learned = min(
    (k for k in preds if "matcher" not in k), key=lambda k: OUT[k]["fpr"])
r = J.paired_clustered(preds["matcher, domain-aware"], preds[best_learned],
                       y_safe, designs, "accuracy", B=10000)
print(f"  domain-aware matcher vs best learned ({best_learned}):")
print(f"    delta accuracy {r['delta']:+.3f}  95% CI "
      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p={r['p_value']:.3f}   "
      f"{'SIGNIFICANT' if r['significant'] else 'not significant'}")
print("    (accuracy on an all-safe set is 1 - FPR, so positive favours"
      " the matcher)")

J.save("j21_tighten", dict(knn_per_seed=per_seed, confirmation=OUT,
                           matcher_vs_best_learned=r,
                           n_conf=len(conf), n_designs=len(set(designs))))
