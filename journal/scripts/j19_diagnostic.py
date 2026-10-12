"""Diagnostic pivot: why does the model flag correct synchronizers, and does
anything trained on generated CDC data transfer at all?

Two corrective hypotheses have been proposed against held-out data and
falsified: mean-pooling dilution (j16) and idiom coverage (j18). The model
flags about 95% of crossings in correct third-party CDC libraries while
scoring 1.000 on every generated split. Rather than try a third fix, this asks
what is actually happening.

Three experiments, all evaluated on the 31-crossing confirmation set built from
the sixteen found designs, every one of which is labeled safe, so **every
positive is a false positive**.

  1. Transfer across model families. GAT, GCN, GraphSAGE, a Random Forest and
     a logistic regression are all trained on the same generated split, and
     both structural matchers are run for reference. If every learned family
     fails while the rule engines do not, the fault lies in the generated
     training distribution rather than in any architecture.

  2. What do flagged crossings resemble? Each confirmation crossing is embedded
     with the trained encoder and matched to its nearest training samples. If
     correct synchronizers land next to UNSAFE training examples, the template
     they are being confused with is identifiable by name.

  3. Which features shift? Node-feature means are compared between generated
     safe crossings and found safe crossings, to see what the encoder is
     actually seeing differently.

Run:  python journal/scripts/j19_diagnostic.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from grouping import group_crossings_structural               # noqa: E402
from corpus_tools import parse_netlist_structural             # noqa: E402
from readouts import train_readout, predict_readout, ReadoutNet  # noqa: E402
from strong_matcher import strong_predict                     # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.baselines import pattern_matcher_predict, FlatML, flat_features  # noqa: E402
from crossexplain.models import collate                       # noqa: E402
from crossexplain.netlist_graph import (find_capture_ffs, extract_subgraph,  # noqa: E402
                                        node_features, CrossingSample,
                                        CELL_VOCAB)

CORPUS = J.JROOT / "corpus"
SEEDS = [0, 1, 2]

# ------------------------------------------------- confirmation set
J.banner("Confirmation set: correct third-party CDC, every crossing safe")
triage = json.loads((J.JROOT / "results" / "j12_corpus_triage.json").read_text())
found = [r for r in triage if r.get("usable") and "duplicate_of" not in r]
conf = []
for r in found:
    jf = CORPUS / "netlists" / f"{r['name']}.O0.json"
    if not jf.exists():
        continue
    nl = parse_netlist_structural(jf)
    caps = find_capture_ffs(nl)
    for grp in group_crossings_structural(nl, caps) if caps else []:
        src_doms = set()
        for f in grp:
            src_doms |= caps[f][0]
        dst = nl.domain[grp[0]]
        nodes, edges, local = extract_subgraph(nl, grp, cid=None)
        feats = node_features(nl, nodes, set(grp), src_doms, dst, edges)
        cone = set(grp)
        for f in grp:
            sf, cb = nl.d_srcs[f]
            cone |= sf | cb
        conf.append(asdict(CrossingSample(
            design=r["name"], level="O0", cid=-1, nodes=nodes,
            node_feats=feats, edges=edges, seeds=[local[f] for f in grp],
            loc_mask=[1 if g in cone else 0 for g in nodes],
            label_safe=1, label_class=-1, template="FOUND_SAFE",
            src_dom=min(src_doms) if src_doms else -1, dst_dom=dst)))
print(f"  {len(conf)} crossings from {len({s['design'] for s in conf})} designs")

train, val = J.load("train"), J.load("val")
test_id = J.load("test_id")
y_id = J.y_of(test_id)
y_tr = J.y_of(train)

# =============================================== 1. transfer across families
J.banner("1. Does any model family transfer?")
rows = {}

for arch in ("gat", "gcn", "sage"):
    pc, pi = [], []
    for sd in SEEDS:
        m = train_readout(train, val, readout="mean_seed", arch=arch,
                          seed=sd, epochs=40)
        pc.append(np.array(predict_readout(m, conf)))
        pi.append(np.array(predict_readout(m, test_id)))
    ec = np.mean(pc, axis=0) > 0.5
    rows[f"GNN-{arch}"] = dict(conf_fpr=float(ec.mean()),
                               test_id_f1=J.f1(np.mean(pi, axis=0) > 0.5, y_id))

flat = FlatML(seed=0).fit(train)
rows["RandomForest"] = dict(
    conf_fpr=float(np.mean([d == 0 for d in flat.predict(conf)[0]])),
    test_id_f1=J.f1(np.array([d == 0 for d in flat.predict(test_id)[0]]), y_id))

from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
from sklearn.pipeline import make_pipeline                         # noqa: E402
Xtr = np.stack([flat_features(s) for s in train])
lr = make_pipeline(StandardScaler(),
                   LogisticRegression(max_iter=2000)).fit(Xtr, y_tr.astype(int))
rows["LogisticReg"] = dict(
    conf_fpr=float(lr.predict(np.stack([flat_features(s) for s in conf])).mean()),
    test_id_f1=J.f1(lr.predict(np.stack([flat_features(s) for s in test_id])).astype(bool), y_id))

pm_c = np.array([p == 0 for p in pattern_matcher_predict(conf)])
sm_c = np.array([p == 0 for p in strong_predict(conf, gray_xor_ratio=0.70)])
rows["matcher-textbook"] = dict(
    conf_fpr=float(pm_c.mean()),
    test_id_f1=J.f1(np.array([p == 0 for p in pattern_matcher_predict(test_id)]), y_id))
rows["matcher-domain"] = dict(
    conf_fpr=float(sm_c.mean()),
    test_id_f1=J.f1(np.array([p == 0 for p in strong_predict(test_id, gray_xor_ratio=0.70)]), y_id))
rows["always-safe"] = dict(conf_fpr=0.0, test_id_f1=0.0)

print(f"  {'method':20s} {'trained?':>9s} {'test_id F1':>11s} "
      f"{'conf FPR':>9s}")
for k, v in rows.items():
    tr = "no" if "matcher" in k or k == "always-safe" else "yes"
    print(f"  {k:20s} {tr:>9s} {v['test_id_f1']:11.4f} {v['conf_fpr']:9.3f}")

learned = [v["conf_fpr"] for k, v in rows.items()
           if "matcher" not in k and k != "always-safe"]
ruled = [v["conf_fpr"] for k, v in rows.items() if "matcher" in k]
print(f"\n  learned models : FPR {min(learned):.3f} to {max(learned):.3f}")
print(f"  rule engines   : FPR {min(ruled):.3f} to {max(ruled):.3f}")

# ================================================ 2. what do they resemble?
J.banner("2. What do flagged correct synchronizers resemble?")
model = train_readout(train, val, readout="mean_seed", seed=0, epochs=40)


@torch.no_grad()
def embed(samples, bs=64):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        b = collate(samples[i:i + bs])
        h = torch.relu(model.embed(b["x"]))
        for layer in model.layers:
            h = layer(h, b["ei"])
        sm = h * b["seed_mask"].unsqueeze(-1)
        cnt = torch.zeros(b["n_graphs"], 1)
        cnt.index_add_(0, b["batch"], b["seed_mask"].unsqueeze(-1).float())
        sp = torch.zeros(b["n_graphs"], h.size(1))
        sp.index_add_(0, b["batch"], sm)
        sp = sp / cnt.clamp(min=1.0)
        gl = model._global_half(h, b["x"], b["batch"], b["n_graphs"])
        out.append(torch.cat([gl, sp], dim=1))
    return torch.cat(out).numpy()


Etr, Ecf = embed(train), embed(conf)
Etr /= (np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-9)
Ecf /= (np.linalg.norm(Ecf, axis=1, keepdims=True) + 1e-9)
sim = Ecf @ Etr.T
K = 5
from collections import Counter                                    # noqa: E402
votes, tpl_votes = Counter(), Counter()
for i in range(len(conf)):
    idx = np.argsort(-sim[i])[:K]
    for j in idx:
        votes["unsafe" if y_tr[j] else "safe"] += 1
        tpl_votes[train[j]["template"]] += 1
tot = sum(votes.values())
print(f"  {K}-nearest training neighbours of each confirmation crossing:")
print(f"    unsafe {votes['unsafe']}/{tot} = {votes['unsafe'] / tot:.1%}   "
      f"safe {votes['safe']}/{tot} = {votes['safe'] / tot:.1%}")
print("    nearest templates:")
for t, c in tpl_votes.most_common(6):
    print(f"      {t:18s} {c:4d}  ({c / tot:.1%})")

# ================================================= 3. which features shift?
J.banner("3. Which node features shift between generated and found safe?")
FEATNAMES = list(CELL_VOCAB) + ["is_ff", "dst_dom", "src_dom", "comb",
                                "log_fanin", "log_fanout", "seed", "dist"]


def feat_mean(samples):
    return np.stack([np.asarray(s["node_feats"], dtype=np.float32).mean(0)
                     for s in samples]).mean(0)


gen_safe = feat_mean([s for s in train if s["label_safe"] == 1])
gen_unsafe = feat_mean([s for s in train if s["label_safe"] == 0])
found_safe = feat_mean(conf)
d = found_safe - gen_safe
order = np.argsort(-np.abs(d))[:8]
print(f"  {'feature':14s} {'gen safe':>9s} {'gen unsafe':>11s} "
      f"{'found safe':>11s} {'shift':>8s}")
shifts = []
for i in order:
    shifts.append(dict(feature=FEATNAMES[i], gen_safe=float(gen_safe[i]),
                       gen_unsafe=float(gen_unsafe[i]),
                       found_safe=float(found_safe[i]), shift=float(d[i])))
    toward = ("toward UNSAFE" if abs(found_safe[i] - gen_unsafe[i])
              < abs(found_safe[i] - gen_safe[i]) else "")
    print(f"  {FEATNAMES[i]:14s} {gen_safe[i]:9.3f} {gen_unsafe[i]:11.3f} "
          f"{found_safe[i]:11.3f} {d[i]:+8.3f}  {toward}")

n_toward = sum(1 for i in order
               if abs(found_safe[i] - gen_unsafe[i]) < abs(found_safe[i] - gen_safe[i]))
print(f"\n  of the 8 largest shifts, {n_toward} move the found-safe crossings")
print(f"  closer to the generated UNSAFE mean than to the generated SAFE mean")

J.save("j19_diagnostic", dict(transfer=rows, n_conf=len(conf),
                              knn_unsafe_fraction=votes["unsafe"] / tot,
                              knn_templates=dict(tpl_votes.most_common(8)),
                              feature_shifts=shifts,
                              n_shifts_toward_unsafe=int(n_toward)))
