"""j34 - audit every number in the drafted journal sections.

The conference paper went through a line-by-line reconciliation and six errors
turned up. This does the same job for the journal draft, and does it as a script
so it can be re-run after any result changes.

Each check names a claim, the value asserted in the .tex, and the value derived
from the result JSON. Anything that does not reconcile is printed as FAIL.

Claims that cannot be checked against JSON -- those sourced from FINDINGS_P1
prose rather than a machine-readable result -- are listed separately as MANUAL
so they are never silently assumed correct.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

SEC = JROOT / "paper" / "sections"
FAILS, CHECKS = [], 0

# Every value this script asserts, recorded so that j47_orphan_numbers.py can
# tell an unverified number in the .tex from one this script computes. Several
# claims here are means or spreads over stored values rather than stored values
# themselves, so they appear in no result JSON and would otherwise look like
# orphans to a scan of the paper.
ASSERTED = []


def res(name):
    return json.loads((J.RESULTS / f"{name}.json").read_text())


def chk(claim, asserted, derived, tol=5e-4):
    """Compare a number claimed in the prose against one derived from JSON."""
    global CHECKS
    CHECKS += 1
    ASSERTED.append(asserted)
    ok = (asserted == derived if isinstance(asserted, (str, int))
          and isinstance(derived, (str, int))
          else abs(float(asserted) - float(derived)) <= tol)
    status = "ok  " if ok else "FAIL"
    if not ok:
        FAILS.append((claim, asserted, derived))
    print(f"  [{status}] {claim:<58s} tex={asserted}  json={derived}")


def tex(name):
    return (SEC / name).read_text(encoding="utf-8")


def main():
    J.banner("j34  audit of the drafted journal sections")

    # =========================================================== Section V
    J.banner("Section V - corpus")
    c = res("j25_corpus_table")
    chk("V: 21 designs", 21, c["n_designs"])
    chk("V: 39 crossings", 39, c["n_crossings"])
    chk("V: 4 to 244 flip-flops (min)", 4, c["ff_min"])
    chk("V: 4 to 518 flip-flops (max)", 518, c["ff_max"])
    chk("V: 9 to 4486 cells (min)", 9, c["cells_min"])
    chk("V: 9 to 9807 cells (max)", 9807, c["cells_max"])

    w = res("j13_wrapper_corpus")
    flat = json.dumps(w)
    wrapped = J.wrapped_set()
    y_w = J.y_of(wrapped)
    nodes = np.array([len(s["node_feats"]) for s in wrapped])
    chk("V: wrapped 12 designs", 12,
        len({s["design"] for s in wrapped}))
    chk("V: wrapped 84 labeled samples", 84, len(wrapped))
    chk("V: wrapped 42 unsafe", 42, int(y_w.sum()))
    chk("V: wrapped 42 safe", 42, int((~y_w).sum()))
    chk("V: wrapped mean 254.2 nodes", 254.2, round(float(nodes.mean()), 1),
        tol=0.06)
    chk("V: wrapped median 134 nodes", 134, int(np.median(nodes)))
    chk("V: wrapped range 14 to 837 (min)", 14, int(nodes.min()))
    chk("V: wrapped range 14 to 837 (max)", 837, int(nodes.max()))

    fs = res("j41_feature_shift")
    chk("V: multiplexers 20.9% of real CDC nodes", 0.209, fs["mux_found"],
        tol=1e-3)
    chk("V: 2.4% in generated safe", 0.024, fs["mux_gen_safe"], tol=1e-3)
    chk("V: 3.2% in generated unsafe", 0.032, fs["mux_gen_unsafe"], tol=1e-3)
    chk("V: seven of eight shifts toward the unsafe mean", 7,
        fs["n_toward_unsafe"])
    chk("V: eight features compared", 8, fs["n_features"])
    comb = next(r for r in fs["rows"] if r["feature"] == "comb")
    chk("V: combinational 77.6% real", 0.776, comb["found_safe"], tol=1e-3)
    chk("V: combinational 53.9% gen safe", 0.539, comb["gen_safe"], tol=1e-3)
    chk("V: combinational 57.3% gen unsafe", 0.573, comb["gen_unsafe"],
        tol=1e-3)

    # ========================================================= Section VII
    J.banner("Section VII - saturation")
    s7 = res("j33_saturation_table")
    chk("VII: 18 configurations", 18, s7["n_configs"])
    chk("VII: generated span 0.0037", 0.0037, s7["gen_range"], tol=5e-5)
    chk("VII: hand-written span 0.1828", 0.1828, s7["real_range"], tol=5e-5)
    chk("VII: hand-written min 0.5498", 0.5498, s7["real_min"])
    chk("VII: hand-written max 0.7325", 0.7325, s7["real_max"])
    chk("VII: 13 of 18 at or above 0.999", 13, s7["n_at_ceiling"])
    chk("VII: ratio about 49x", 49, round(s7["real_range"] / s7["gen_range"]),
        tol=0.6)

    depth = [r for r in s7["rows"] if r["group"] == "depth"]
    dg = [r["gen"] for r in depth]
    chk("VII: depth span 0.9988 to 1.0000 (min)", 0.9988, min(dg))
    chk("VII: depth span 0.9988 to 1.0000 (max)", 1.0000, max(dg))
    pmin = float(depth[0]["note"].split("k")[0])
    pmax = float(depth[-1]["note"].split("k")[0])
    chk("VII: depth 1 to 7 is 2.6x parameters", 2.6, pmax / pmin, tol=0.05)

    feats = [r["gen"] for r in s7["rows"] if r["group"] == "features"]
    chk("VII: feature span 0.9963 to 1.0000 (min)", 0.9963, min(feats))
    chk("VII: feature span 0.9963 to 1.0000 (max)", 1.0000, max(feats))
    chk("VII: six feature groups removed", 6, len(feats) - 1)

    cone = [r for r in s7["rows"] if r["group"] == "cone"]
    n8 = float([r for r in cone if "bwd=8" in r["tag"]][0]["note"].split()[0])
    n2 = float([r for r in cone if "bwd=2" in r["tag"]][0]["note"].split()[0])
    chk("VII: cone 8 hops is 36.3 nodes", 36.3, n8, tol=0.05)
    chk("VII: cone 2 hops is 15.9 nodes", 15.9, n2, tol=0.05)
    chk("VII: cone at 2 hops still 1.0000", 1.0000,
        [r for r in cone if "bwd=2" in r["tag"]][0]["gen"])

    # ======================================================== Section VIII
    J.banner("Section VIII - transfer")
    t = res("j32_transfer_table")
    by = {r["method"]: r for r in t["rows"]}
    chk("VIII: six of nine at 1.0000", 6, t["n_at_ceiling"])
    chk("VIII: nine methods", 9, len(t["rows"]))
    chk("VIII: benchmark spans 0.1358", 0.1358, t["gen_range"], tol=5e-5)
    chk("VIII: real spans 0.897", 0.897, t["real_range"], tol=1e-3)
    chk("VIII: Spearman rho -0.374", -0.374, t["spearman_rho"], tol=1e-3)
    chk("VIII: Spearman p 0.321", 0.321, t["spearman_p"], tol=1e-3)
    chk("VIII: ensemble real FPR 0.974", 0.974,
        by["CrossExplain (5-seed ens.)"]["conf_fpr"], tol=1e-3)
    chk("VIII: GCN real FPR 0.103", 0.103, by["GCN ablation"]["conf_fpr"],
        tol=1e-3)
    m = by["matcher, domain-aware"]
    chk("VIII: domain matcher gen 1.0000", 1.0000, m["test_id_f1"])
    chk("VIII: domain matcher real F1 0.7619", 0.7619, m["test_real_f1"])
    chk("VIII: domain matcher wrapped recall 1.000", 1.000,
        m["wrapped_recall"])
    chk("VIII: domain matcher FPR 0.692", 0.692, m["conf_fpr"], tol=1e-3)
    rank = sorted(t["rows"], key=lambda r: -r["test_real_f1"])
    chk("VIII: domain matcher is 2nd on test_real", "matcher, domain-aware",
        rank[1]["method"])
    tb = by["matcher, textbook"]
    chk("VIII: textbook matcher gen 0.8778", 0.8778, tb["test_id_f1"])
    chk("VIII: textbook matcher FPR 1.000", 1.000, tb["conf_fpr"])
    chk("VIII: single-task misses 29% of wrapped", 0.29,
        1 - by["single-task ablation"]["wrapped_recall"], tol=6e-3)
    # Section VIII quoted 0.806 here, which is 25/31 -- the single-task FPR on
    # the pre-merge corpus. It is 31/39 now. The wrapped-recall claim beside
    # it was checked all along; the rate itself was not.
    chk("VIII: single-task real-code FPR 0.795", 0.795,
        by["single-task ablation"]["conf_fpr"], tol=6e-4)
    chk("VIII: GraphSAGE misses 14% of wrapped", 0.14,
        1 - by["GraphSAGE ablation"]["wrapped_recall"], tol=6e-3)
    chk("VIII: RF gen 0.9963", 0.9963, by["Random Forest"]["test_id_f1"])
    # test_big: the split built to be harder turned out to be more saturated.
    chk("VII: seven of nine at 1.0000 on test_big", 7, t["n_at_ceiling_big"])
    chk("VII: RF reaches 1.0000 on test_big", 1.0000,
        by["Random Forest"]["test_big_f1"])
    chk("VII: test_big has more ties than test_id", True,
        t["n_at_ceiling_big"] > t["n_at_ceiling"])
    chk("VIII: best on real is the GCN ablation", "GCN ablation",
        t["best_on_real"])

    # ========================================================== Section IX
    J.banner("Section IX - the null change")
    a = res("j23_pooling_order")
    b = res("j23b_divergence_trace")
    chk("IX: fixed reproduces reference (agreement 1.0)", 1.0,
        a["models"]["fixed"]["agree_with_reference"])
    chk("IX: legacy agrees on 12.8%", 0.128,
        a["models"]["legacy"]["agree_with_reference"], tol=1e-3)
    chk("IX: global-first FPR 0.103", 0.103,
        a["models"]["fixed"]["conf_fpr"], tol=1e-3)
    chk("IX: seed-first FPR 0.974", 0.974, a["models"]["legacy"]["conf_fpr"],
        tol=1e-3)
    for k in ("reference", "fixed", "legacy"):
        chk(f"IX: {k} test_id F1 1.0000", 1.0000,
            a["models"][k]["test_id_f1"])
    chk("IX: forward pass difference is zero", 0.0, b["forward_max_diff"])
    # The prose said the two runs' validation F1 tracks "within 0.02 at every
    # epoch". At epoch 4 the gap is 0.0206. Derived now, so the bound stated
    # in Section IX cannot be tighter than the trace supports.
    tr = res("j23b_divergence_trace")["trace"]
    chk("IX: val F1 gap never exceeds 0.021", 0.021,
        max(abs(r["val_f1_fixed"] - r["val_f1_legacy"]) for r in tr),
        tol=6e-4)

    chk("IX: first loss 2.302386761 (fixed)", 2.302386761,
        b["first_loss_fixed"], tol=5e-9)
    chk("IX: first loss identical", 0.0,
        abs(b["first_loss_fixed"] - b["first_loss_legacy"]), tol=1e-9)
    tr = {r["epoch"]: r for r in b["trace"]}
    chk("IX: epoch 0 relative divergence 8.67e-07", 8.67e-7,
        tr[0]["rel_delta"], tol=5e-9)
    chk("IX: epoch 7 relative divergence 1.17e-02", 1.17e-2,
        tr[7]["rel_delta"], tol=5e-5)
    chk("IX: val F1 at epoch 0, fixed", 0.7700, tr[0]["val_f1_fixed"])
    chk("IX: val F1 at epoch 0, legacy", 0.7700, tr[0]["val_f1_legacy"])
    chk("IX: val F1 at epoch 7, fixed", 0.9854, tr[7]["val_f1_fixed"])
    chk("IX: val F1 at epoch 7, legacy", 0.9665, tr[7]["val_f1_legacy"])

    # =========================================================== Section X
    J.banner("Section X - dispersion")
    d = res("j30_multi_dispersion")
    q = res("j24_dispersion")
    g = res("j27_why_gcn")
    chk("X: 60 draws", 60, d["n_draws"])
    chk("X: 20 draws per architecture", 20, d["per_arch"]["gcn"]["n"])
    chk("X: benchmark min 0.9951", 0.9951, d["test_id"]["min"])
    chk("X: benchmark max 1.0000", 1.0000, d["test_id"]["max"])
    chk("X: benchmark range 0.0049", 0.0049, d["test_id"]["range"], tol=5e-5)
    n_at_one = sum(1 for r in d["rows"] if abs(r["test_id_f1"] - 1) < 1e-9)
    chk("X: 59 of 60 at exactly 1.0000", 59, n_at_one)
    chk("X: real min 0.103", 0.103, d["pooled"]["min"], tol=1e-3)
    chk("X: real max 1.000", 1.000, d["pooled"]["max"])
    chk("X: real range 0.897", 0.897, d["pooled"]["range"], tol=1e-3)
    chk("X: real sd 0.253", 0.253, d["pooled"]["sd"], tol=1e-3)
    chk("X: ratio of ranges about 180", 180,
        d["pooled"]["range"] / d["test_id"]["range"], tol=5)
    chk("X: 30 of 60 intermediate", 30, d["pooled"]["n_mid"])
    for a_, sd in (("gat", 0.041), ("sage", 0.049), ("gcn", 0.372)):
        chk(f"X: {a_} sd {sd}", sd, d["per_arch"][a_]["sd"], tol=1e-3)
    chk("X: GAT interval min 0.821", 0.821, d["per_arch"]["gat"]["min"],
        tol=1e-3)
    chk("X: GAT interval max 1.000", 1.000, d["per_arch"]["gat"]["max"])
    chk("X: SAGE interval min 0.564", 0.564, d["per_arch"]["sage"]["min"],
        tol=1e-3)
    chk("X: SAGE interval max 0.744", 0.744, d["per_arch"]["sage"]["max"],
        tol=1e-3)
    chk("X: GAT and SAGE intervals are disjoint", True,
        d["per_arch"]["sage"]["max"] < d["per_arch"]["gat"]["min"])
    chk("X: GCN mean 0.729", 0.729, d["per_arch"]["gcn"]["mean"], tol=1e-3)
    chk("X: SAGE mean 0.682", 0.682, d["per_arch"]["sage"]["mean"], tol=1e-3)
    gcn_v = sorted(r["conf_fpr"] for r in d["rows"] if r["arch"] == "gcn")
    chk("X: 4 of 20 GCN draws at the 0.103 minimum", 4,
        sum(1 for x in gcn_v if abs(x - 0.103) < 1e-3))
    chk("X: 12-config grid test_real range 0.1303", 0.1303,
        q["summary"]["test_real"]["range"], tol=5e-5)
    chk("X: 12-config grid wrapped range 0.1905", 0.1905,
        q["summary"]["wrapped"]["range"], tol=5e-5)
    chk("X: 12-config grid benchmark range 0.0000", 0.0000,
        q["summary"]["test_id"]["range"])
    gc = g["ablation_gcn"]
    chk("X: released GCN median p 0.111", 0.111, gc["median"], tol=1e-3)
    chk("X: released GCN, none within 0.05 of threshold", 0,
        gc["n_near_threshold"])
    chk("X: released GCN wrapped separation 0.712", 0.712, gc["separation"],
        tol=1e-3)
    chk("X: released GAT median p 0.999", 0.999,
        g["crossexplain_gat"]["median"], tol=1e-3)

    # =================================================== ported sections
    J.banner("Sections III, IV, VI - ported from the conference paper")
    # These carry over from the audited conference version. We re-derive the
    # ones that come from data we still hold, rather than trusting the port.
    for split, n_samp, n_cross, n_des in (
            ("train", 1435, 723, 160), ("val", 193, 97, 20),
            ("test_id", 735, 189, 40), ("test_big", 512, 128, 16),
            ("test_real", 68, 17, 12)):
        s = J.load(split)
        u = J.units(s)
        chk(f"III: {split} samples", n_samp, len(s))
        chk(f"III: {split} crossings", n_cross, len(set(u["crossing"])))
        chk(f"III: {split} designs", n_des, len(set(u["design"])))

    # Upstream projects. The abstract and Section I both said "21 third-party
    # CDC libraries" and then listed five project names, conflating designs
    # with the projects they come from; the list also named verilog-pcie,
    # which contributes no crossings at all. Both counts are derived now.
    PROJECT = {"ot": "OpenTitan", "pulp": "pulp-platform",
               "axis": "verilog-axis", "wb2axip": "wb2axip",
               "pcie": "verilog-pcie"}
    conf_designs = {s["design"] for s in J.confirmation_set()}
    contributing = {PROJECT[d.split("_")[0]] for d in conf_designs}
    chk("I: 20 designs contribute crossings", 20, len(conf_designs))
    chk("I: four upstream projects contribute", 4, len(contributing))
    chk("I: verilog-pcie contributes nothing", True,
        "verilog-pcie" not in contributing)

    # One mislabeled crossing shifts every false-positive rate by 1/n. Section
    # XII said "about three points", which was 1/31 and survived the corpus
    # growing to 39.
    chk("XII: one crossing moves each rate by 2.6 points", 2.6,
        100.0 / res("j23_pooling_order")["n_conf"], tol=0.06)

    # Near-miss pair sizes. The conference version quoted the training split
    # while reading as a claim about the templates; j47 caught that nothing
    # derived those numbers at all. These are corpus-wide, from j48.
    nm = res("j48_nearmiss_pairs")["pairs"]
    mh = nm["MUX_UNSYNC_QUAL|MUX_HOLD"]
    chk("III: MUX pair unsafe range 32 (min)", 32, mh["unsafe_min"])
    chk("III: MUX pair unsafe range 101 (max)", 101, mh["unsafe_max"])
    chk("III: MUX pair safe range 34 (min)", 34, mh["safe_min"])
    chk("III: MUX pair safe range 107 (max)", 107, mh["safe_max"])
    chk("III: MUX pair range overlap 0.893", 0.893, mh["range_overlap"],
        tol=1e-3)
    chk("III: MUX pair unsafe mean 57.7", 57.7, mh["unsafe_mean"], tol=0.06)
    chk("III: MUX pair safe mean 57.9", 57.9, mh["safe_mean"], tol=0.06)
    chk("III: MUX pair mean gap 0.1", 0.1, mh["mean_gap"], tol=0.06)
    sl = nm["SYNC2_LEAK|SYNC2"]
    chk("III: SYNC2_LEAK pair is the widest gap, 4.2", 4.2,
        max(r["mean_gap"] for r in nm.values()), tol=0.06)
    chk("III: SYNC2_LEAK pair gap 4.2", 4.2, sl["mean_gap"], tol=0.06)
    bysp = res("j48_nearmiss_pairs")["by_split"]
    chk("III: train-split MUX means 60.7", 60.7,
        bysp["train"]["MUX_UNSYNC_QUAL|MUX_HOLD"]["unsafe_mean"], tol=0.06)
    chk("III: train-split MUX means 59.2", 59.2,
        bysp["train"]["MUX_UNSYNC_QUAL|MUX_HOLD"]["safe_mean"], tol=0.06)
    chk("III: test_id MUX means invert, 51.2", 51.2,
        bysp["test_id"]["MUX_UNSYNC_QUAL|MUX_HOLD"]["unsafe_mean"], tol=0.06)
    chk("III: test_id MUX means invert, 53.8", 53.8,
        bysp["test_id"]["MUX_UNSYNC_QUAL|MUX_HOLD"]["safe_mean"], tol=0.06)

    n_feat = len(J.load("train")[0]["node_feats"][0])
    chk("IV: 27 node features", 27, n_feat)

    dep = res("j03_depth_ablation")
    d5 = [r for r in s7["rows"] if r["group"] == "depth"]
    chk("IV: 1 layer within 0.0012 of 5 layers", 0.0012,
        abs(d5[4]["gen"] - d5[0]["gen"]), tol=5e-5)
    chk("IV: 64,845 parameters at depth 5", 64845,
        dep["per_depth"]["5"]["per_seed"][0]["n_params"])

    # Grouping: assert the values, not substrings. The JSON carries full
    # precision, so a substring test on a rounded figure fails spuriously.
    e = res("j08_grouping")["test_real"]
    chk("IV: shipped fallback recovers 3.96x true crossings", 3.96,
        e["shipped"]["ratio_vs_truth"], tol=5e-3)
    chk("IV: shipped fallback real F1 0.534", 0.534,
        e["shipped"]["detection"]["f1"])
    chk("IV: fixed grouping recovers 0.94x", 0.94,
        e["improved"]["ratio_vs_truth"], tol=5e-3)
    chk("IV: fixed grouping real F1 0.767", 0.767,
        e["improved"]["detection"]["f1"])
    chk("IV: name-based reference 0.781", 0.781,
        e["name_based_reference_f1"])

    # Interval widths are means over the FIVE RELEASED SEEDS, not over
    # methods. Section VI states that basis; getting it wrong pairs 0.525
    # with the wrong denominator.
    ps = res("j01_clustered_stats")["per_seed_ci_test_real"]
    w_des = float(np.mean([v["design"]["width"] for v in ps.values()]))
    w_sam = float(np.mean([v["sample"]["width"] for v in ps.values()]))
    pts = [v["design"]["point"] for v in ps.values()]
    chk("VI: five released seeds", 5, len(ps))
    chk("VI: design-clustered mean width 0.525", 0.525, w_des, tol=1e-3)
    chk("VI: sample-level mean width 0.252", 0.252, w_sam, tol=1e-3)
    chk("VI: understated by about a factor of two", 2.0, w_des / w_sam,
        tol=0.1)
    chk("VI: seed spread 0.158", 0.158, max(pts) - min(pts), tol=1e-3)

    cal = res("j02_calibration")
    chk("VI: real-design ECE 0.267", 0.267,
        cal["calibration"]["test_real"]["ece"], tol=1e-3)
    chk("VI: self-correctness AUC 0.483", 0.483,
        cal["error_detection"]["auc"], tol=1e-3)

    # Reliability diagram (Fig. 5). Section VI-C quotes three specific bins to
    # show the curve is non-monotonic, so those bins are checked individually.
    rel = {round(b["mean_conf"], 2): b
           for b in cal["calibration"]["test_real"]["reliability"]
           if b["count"] > 0}
    chk("VI: in-distribution ECE 0.013", 0.013,
        cal["calibration"]["test_id"]["ece"], tol=1e-3)
    chk("VI: 0.00 accuracy in the 0.52 bin", 0.00, rel[0.52]["accuracy"])
    chk("VI: 0.80 accuracy in the 0.72 bin", 0.80, rel[0.72]["accuracy"],
        tol=5e-3)
    chk("VI: 0.29 accuracy in the 0.76 bin", 0.29, rel[0.76]["accuracy"],
        tol=5e-3)
    accs = [b["accuracy"] for b in
            sorted((b for b in cal["calibration"]["test_real"]["reliability"]
                    if b["count"] > 0), key=lambda b: b["mean_conf"])]
    chk("VI: real-design reliability is non-monotonic", True,
        any(accs[i] > accs[i + 1] for i in range(len(accs) - 1)))

    # ============================================================== manual
    J.banner("MANUAL - claims not backed by a result JSON")
    # Nine Section V claims that were listed here are now checked
    # automatically against the triage record and the j12 run logs by
    # j37_audit_corpus_history.py (22 checks). Run it alongside this script.
    h = J.RESULTS / "j37_audit_corpus_history.json"
    if h.exists():
        hh = json.loads(h.read_text())
        print(f"  Section V corpus history: automated by j37, "
              f"{hh['n_pass']}/{hh['n_checks']} checks reconcile")
    else:
        print("  Section V corpus history: run j37_audit_corpus_history.py")

    # The last two were closed by j38, which re-derives both and saves JSON.
    p38 = J.RESULTS / "j38_close_manual.json"
    if p38.exists():
        d38 = json.loads(p38.read_text())
        ck, rl = d38["clock_rule"], d38["relabel"]
        chk("V: structural clock rule, zero disagreements", 0, ck["disagree"])
        chk("V: validated on 632 generated netlists", 632, ck["n_checked"])
        chk("X: relabeling max logit change 1.5e-08", 1.5e-8,
            rl["max_abs_diff"], tol=5e-10)
        chk("X: relabeling changes no decision", 0, rl["decisions_changed"])
        chk("X: 256 probe samples", 256, rl["n_samples"])
        print("\n  No MANUAL claims remain. j37 covers the Section V corpus "
              "history and j38\n  covers the two one-off verifications.")
    else:
        print("  [MANUAL] run j38_close_manual.py to check the clock rule "
              "and the\n           node-relabeling no-op")

    # ================================================ self-referential
    J.banner("Section XI - the paper's claim about its own audit")
    # Section XI states how many checks this script runs. If that drifts, the
    # paper is wrong about itself -- the one error nothing else would catch.
    # The two checks below are part of the total they verify, hence CHECKS + 2.
    expected = CHECKS + 2
    xi = tex("sec11_repro.tex")
    m = re.search(r"(\d+) checks against the result JSON", xi)
    chk("XI: states this script's own check count",
        int(m.group(1)) if m else -1, expected)
    h2 = J.RESULTS / "j37_audit_corpus_history.json"
    m2 = re.search(r"and (\d+) more that re-derive", xi)
    chk("XI: states j37's check count",
        int(m2.group(1)) if m2 else -1,
        json.loads(h2.read_text())["n_checks"] if h2.exists() else -1)

    J.banner("Result")
    print(f"  {CHECKS - len(FAILS)} of {CHECKS} automated checks reconcile")
    if FAILS:
        print("\n  FAILURES:")
        for claim, asserted, derived in FAILS:
            print(f"    {claim}: tex={asserted} json={derived}")
    J.save("j34_audit_journal", dict(
        n_checks=CHECKS, n_pass=CHECKS - len(FAILS),
        asserted=ASSERTED,
        failures=[dict(claim=c, tex=a, json=d) for c, a, d in FAILS]))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
