"""Extract every numeric literal in the paper with its context, and print the
ground-truth values it must match, so each can be checked by eye.

Not a checker: it is a worksheet generator. Deciding whether "0.878" in a
sentence refers to test_id matcher F1 needs reading, not regex.

Run:  python journal/scripts/j10_audit_numbers.py > journal/results/audit_numbers.txt
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TEX = ROOT / "paper" / "crossexplain_ieee.tex"

body = TEX.read_text(encoding="utf-8")
# strip the TikZ figure block: its coordinates are not claims
start = body.find("\\begin{tikzpicture}")
end = body.find("\\end{tikzpicture}")
if start > 0:
    body = body[:start] + "\n%%% TIKZ REMOVED %%%\n" + body[end:]

print("=" * 78)
print("PART 1 - every numeric literal in the paper, with context")
print("=" * 78)

lines = body.split("\n")
num = re.compile(r"\$?\b\d+(?:[.,]\d+)?\b\$?")
seen = 0
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if not s or s.startswith("%"):
        continue
    if s.startswith("\\bibitem") or "Accessed:" in s or "\\url{" in s:
        continue
    hits = [h for h in num.findall(s)
            if not re.fullmatch(r"\$?[01]\$?", h)]      # drop bare 0/1
    if not hits:
        continue
    seen += 1
    ctx = " ".join(lines[max(0, i - 2):i + 1]).strip()
    ctx = re.sub(r"\s+", " ", ctx)
    print(f"\nL{i:4d}  {sorted(set(hits))}")
    print(f"      ...{ctx[-230:]}")
print(f"\n({seen} lines carry numeric claims)")

print()
print("=" * 78)
print("PART 2 - ground truth")
print("=" * 78)


def show(title, obj):
    print(f"\n--- {title} ---")
    print(json.dumps(obj, indent=1, default=float)[:3000])


R = json.loads((ROOT / "results" / "results.json").read_text())
E = json.loads((ROOT / "results" / "error_analysis.json").read_text())
JD = ROOT / "journal" / "results"


def jload(n):
    p = JD / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else {}


print("\n--- detection F1 / FPR, all methods, all splits (results.json) ---")
for sp in ["test_id", "test_big", "test_real"]:
    for m, v in R[sp].items():
        if isinstance(v, dict) and "detection" in v:
            d = v["detection"]
            print(f"  {sp:10s} {m:22s} F1={d['f1']:.4f} FPR={d['false_positive_rate']:.4f} "
                  f"acc={d['accuracy']:.4f} P={d['precision']:.4f} R={d['recall']:.4f}")

print("\n--- type acc / loc IoU / xai / runtime ---")
for sp in ["test_id", "test_big", "test_real"]:
    g = R[sp].get("crossexplain_gat", {})
    print(f"  {sp:10s} type={g.get('type_accuracy')} iou={g.get('localization_iou')}")
for sp, v in R["xai"].items():
    print(f"  {sp:10s} fid_edge={v['edge']['fidelity']:.4f} "
          f"fid_feat={v['feature']['fidelity']:.4f} sparsity={v['edge']['sparsity']:.4f}")
print(f"  runtime {R['runtime']}")

print("\n--- confusion, test_id GAT (for the RECONV claim) ---")
print(f"  {R['test_id']['crossexplain_gat']['confusion']}")

print("\n--- per-template test_id (results.json) ---")
for t, v in R["per_template"].items():
    print(f"  {t:18s} n={v['n']:4d} pm={v['pm_acc']:.3f} gnn={v['gnn_acc']:.3f}")

print("\n--- robustness by level ---")
for lv, v in R["robustness"].items():
    print(f"  {lv} n={v['n']:4d} pm_F1={v['pattern_matcher']['f1']:.4f} "
          f"gnn_F1={v['crossexplain']['f1']:.4f}")

print("\n--- selection protocols / seed variance ---")
for k, v in R["selection_protocols"].items():
    print(f"  {k:28s} F1={v['f1']:.4f} n={v['tp'] + v['fp'] + v['tn'] + v['fn']}")
for k, v in R["seed_variance_test_real"].items():
    print(f"  seed{k} P={v['precision']:.3f} R={v['recall']:.3f} "
          f"FPR={v['false_positive_rate']:.3f} F1={v['f1']:.4f}")

print("\n--- error_analysis.json ---")
print(f"  ci_width_mean={E['ci_width_mean']:.4f} seed_spread={E['seed_spread']:.4f}")
print(f"  always_right={E['always_right']} always_wrong={E['always_wrong']} "
      f"unstable={E['unstable']}")
print(f"  structural_stats={ {k: round(v['nodes'], 2) for k, v in E['structural_stats'].items()} }")
print(f"  accuracy_by_size={ {k: v['acc'] for k, v in E['accuracy_by_size'].items()} }")
print(f"  per_template_real={ {k: round(v['gnn_acc'], 3) for k, v in E['per_template_real'].items()} }")
print(f"  matcher_real   ={ {k: round(v['matcher_acc'], 3) for k, v in E['per_template_real'].items()} }")

j1 = jload("j01_clustered_stats")
if j1:
    print("\n--- j01 clustered stats ---")
    for m, v in j1["ci_by_unit_test_real"].items():
        d = v["design"]
        print(f"  {m:22s} F1={d['point']:.4f} design CI=[{d['ci_lo']:.3f},{d['ci_hi']:.3f}] w={d['width']:.3f}")
    for k, v in j1["paired_tests_test_real"].items():
        print(f"  {k:34s} d={v['delta']:+.4f} CI=[{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] p={v['p_value']:.3f}")
    print(f"  per-seed design CI widths: "
          f"{[round(v['design']['width'], 3) for v in j1['per_seed_ci_test_real'].values()]}")
    ws = [v['design']['width'] for v in j1['per_seed_ci_test_real'].values()]
    print(f"  MEAN design CI width = {sum(ws) / len(ws):.4f}")
    for k, v in j1["per_seed_ci_test_real"].items():
        d = v["design"]
        print(f"    seed{k} F1={d['point']:.4f} CI=[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]")
    print(f"  effective_n: { {k: (v['samples'], v['crossings'], v['designs']) for k, v in j1['effective_n'].items()} }")
    print(f"  identical O2==O3: test_id={j1['effective_n']['test_id']['identical_level_pairs'].get('O2==O3')} "
          f"test_big={j1['effective_n']['test_big']['identical_level_pairs'].get('O2==O3')}")
    for sp, v in j1["generated_splits"].items():
        print(f"  {sp}: vs matcher d={v['paired_vs_matcher']['delta']:+.4f} p={v['paired_vs_matcher']['p_value']:.3f}"
              f" | vs RF d={v['paired_vs_rf']['delta']:+.4f} p={v['paired_vs_rf']['p_value']:.3f}")
    print(f"  power: {j1['power_projection']['projection'][:5]}")

j2 = jload("j02_calibration")
if j2:
    print("\n--- j02 calibration ---")
    for sp, v in j2["calibration"].items():
        print(f"  {sp:10s} ECE={v['ece']:.4f} meanconf={v['mean_confidence']:.3f} acc={v['accuracy']:.3f}")
    print(f"  error-detection AUC={j2['error_detection']['auc']:.4f}")
    print(f"  fitted temperatures={j2['fitted_temperatures']}")
    print(f"  oracle T on test_real={j2['oracle_temperature']['test_real']}")

j3 = jload("j03_depth_ablation")
if j3:
    print("\n--- j03 depth ---")
    for d, v in j3["per_depth"].items():
        mh = v["mean_per_template_acc"].get("MUX_HOLD")
        mu = v["mean_per_template_acc"].get("MUX_UNSYNC_QUAL")
        print(f"  depth {d}: test_id={v['mean_test_id_f1']:.4f} real={v['mean_test_real_f1']:.3f} "
              f"MUX_HOLD={mh:.3f} MUX_UNSYNC_QUAL={mu:.3f}")

j4 = jload("j04_strong_baseline")
if j4:
    print("\n--- j04 strong matcher ---")
    print(f"  fit: {j4['fit']}")
    for sp, v in j4["detection"].items():
        for m in ["matcher_original", "matcher_strong", "random_forest", "crossexplain_ens"]:
            if m in v:
                print(f"  {sp:10s} {m:18s} F1={v[m]['f1']:.4f} FPR={v[m]['false_positive_rate']:.4f} "
                      f"R={v[m]['recall']:.4f} tp={v[m]['tp']} fn={v[m]['fn']}")
        for m in ["paired_ens_vs_matcher_strong", "paired_ens_vs_random_forest",
                  "paired_ens_vs_matcher_original"]:
            if m in v:
                print(f"  {sp:10s} {m:32s} d={v[m]['delta']:+.4f} p={v[m]['p_value']:.3f}")
    for sp, rows in j4.get("per_template", {}).items():
        print(f"  --- per-template {sp} ---")
        for t, r in rows.items():
            print(f"    {t:18s} n={r['n']:3d} {r['label']:6s} orig={r['matcher_original']:.3f} "
                  f"strong={r['matcher_strong']:.3f} ens={r['crossexplain_ens']:.3f}")

for n in ["j05_name_blind", "j06_paper_figures", "j07_opencdc", "j08_grouping",
          "j09_ablations"]:
    d = jload(n)
    if d:
        show(n, d if n != "j09_ablations" else
             {"feature_groups": [{k: r[k] for k in ("tag", "test_id_f1_mean",
                                                    "test_real_f1_mean",
                                                    "test_real_f1_ens")}
                                 for r in d["feature_groups"]],
              "hops": [{k: r[k] for k in ("tag", "mean_train_nodes",
                                          "test_id_f1_mean", "test_real_f1_mean",
                                          "test_real_f1_ens")} for r in d["hops"]]})

print()
print("=" * 78)
print("PART 3 - citations")
print("=" * 78)
cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", body):
    for c in m.group(1).split(","):
        cites.add(c.strip())
items = set(re.findall(r"\\bibitem\{([^}]+)\}", TEX.read_text(encoding='utf-8')))
print(f"  cited: {len(cites)}   bibitems: {len(items)}")
print(f"  cited but no bibitem : {sorted(cites - items) or 'none'}")
print(f"  bibitem never cited  : {sorted(items - cites) or 'none'}")
