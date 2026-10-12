"""P0.6 - OpenCDC as an external baseline.

Both matchers in this project are ours, which limits how much the P0.5 result
can settle. OpenCDC is the only open-source CDC netlist checker we could find,
and we did not write it. See journal/external/PROVENANCE.md for the commit,
the licence, and the two run-time compatibility shims applied below.

Two things are measured, because OpenCDC does two different amounts of work:

  1. Crossing discovery. What fraction of our construction crossings does it
     find at all? It follows direct flip-flop to flip-flop net connections, so
     any crossing through combinational logic is invisible to it.
  2. Verdict quality. It has no synchronizer recognition, so scored as a
     detector it necessarily predicts unsafe for everything it reports. We
     score that policy honestly rather than pretending it classifies.

Run:  python journal/scripts/j07_opencdc.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402

sys.path.insert(0, str(J.JROOT / "external"))
import opencdc.parser as ocdc_parser                          # noqa: E402
import opencdc.checker as ocdc_checker                        # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.netlist_graph import parse_netlist, group_crossings, CDC_RE  # noqa: E402

# --------------------------------------------------- compatibility shims
# (1) gate-level flip-flop primitives. Unmodified, the tool matches only
#     word-level types and finds zero flip-flops in a gate-level netlist.
_GATE_FF = {"$_dff_p_", "$_dff_n_", "$_dffe_pp_", "$_dffe_pn_",
            "$_dff_pp0_", "$_dff_pp1_", "$_sdff_pp0_", "$_sdff_pp1_",
            "$_dffe_pp0p_", "$_dffe_pp1p_", "$_sdffe_pp0p_", "$_sdffe_pp1p_"}
_orig_get_ffs = ocdc_parser.get_flip_flops


def _patched_get_flip_flops(netlist):
    cells = ocdc_parser.get_cells(netlist)
    out = []
    for name, cell in cells.items():
        t = cell.get("type", "").lower()
        if t in {"$dff", "$dffe", "$sdff", "$adff", "$fdre", "$fdce"} \
                or t in _GATE_FF or t.startswith("$_dff") or t.startswith("$_sdff"):
            out.append({"name": name, "type": cell["type"],
                        "connections": cell.get("connections", {})})
    return out


# (2) gate-level cells name the clock pin C, not CLK
def _patched_ff_domain(ff):
    conns = ff.get("connections", {})
    for pin in ("CLK", "C"):
        nets = conns.get(pin, [])
        if nets:
            return nets[0]
    return None


ocdc_parser.get_flip_flops = _patched_get_flip_flops
ocdc_checker.get_flip_flops = _patched_get_flip_flops
ocdc_checker._ff_domain = _patched_ff_domain

J.banner("P0.6  OpenCDC as an external baseline")
print("  vendored commit 6666dac, MIT; two format shims applied at run time")

OUT = {}
for split, folder in [("test_real", "test_real"), ("test_id", "test_id")]:
    ddir = J.ROOT / "data" / "designs" / folder
    ours = J.load(split)
    designs = sorted({s["design"] for s in ours})

    n_ocdc, n_ours, per_design = 0, 0, {}
    ocdc_seed_hits, total_true = 0, 0
    unmodified_ff_count = 0
    for d in designs:
        for lvl in ["O0", "O1", "O2", "O3"]:
            jf = ddir / f"{d}.{lvl}.json"
            if not jf.exists():
                continue
            raw = json.loads(jf.read_text())
            unmodified_ff_count += len(_orig_get_ffs(raw))

            cross = ocdc_checker.find_cross_domain_nets(raw)
            # OpenCDC reports one row per (net, sink FF); collapse to the set
            # of sink flip-flops it considers to be capturing a crossing
            ocdc_caps = {c["sink_ff"] for c in cross}

            nl = parse_netlist(jf)
            _, by_cid, _ = group_crossings(nl)
            true_caps = {nl.cells[f][0] for ffs in by_cid.values() for f in ffs}

            # a construction crossing counts as found if OpenCDC flags any of
            # its capture flip-flops
            for cid, ffs in by_cid.items():
                total_true += 1
                names = {nl.cells[f][0] for f in ffs}
                if names & ocdc_caps:
                    ocdc_seed_hits += 1

            n_ocdc += len(ocdc_caps)
            n_ours += len(true_caps)
            per_design.setdefault(d, {"ocdc": 0, "ours": 0})
            per_design[d]["ocdc"] += len(ocdc_caps)
            per_design[d]["ours"] += len(true_caps)

    recall = ocdc_seed_hits / max(1, total_true)
    print(f"\n  --- {split} ---")
    print(f"    flip-flops found by UNMODIFIED OpenCDC: {unmodified_ff_count} "
          f"(this is why the shim is necessary)")
    print(f"    capture FFs flagged: OpenCDC {n_ocdc}, ours {n_ours}")
    print(f"    construction crossings discovered: {ocdc_seed_hits}/{total_true} "
          f"= {recall:.3f}")

    # verdict policy: everything it reports is called unsafe; everything it
    # does not report is, implicitly, not flagged
    y = J.y_of(ours)
    u = J.units(ours)
    flagged = []
    for s in ours:
        jf = ddir / f"{s['design']}.{s['level']}.json"
        raw = json.loads(jf.read_text())
        caps = {c["sink_ff"] for c in ocdc_checker.find_cross_domain_nets(raw)}
        nl = parse_netlist(jf)
        _, by_cid, _ = group_crossings(nl)
        names = {nl.cells[f][0] for f in by_cid.get(s["cid"], [])}
        flagged.append(bool(names & caps))
    flagged = np.array(flagged)
    m = J.metrics(flagged, y)
    ci = J.clustered_ci(flagged, y, u["design"], "f1", B=2000)
    print(f"    as a detector: F1={m['f1']:.3f} [{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}]"
          f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}"
          f"  FPR={m['false_positive_rate']:.3f}")
    print(f"    (flags {flagged.mean():.1%} of crossings; base rate of unsafe "
          f"is {y.mean():.1%})")

    OUT[split] = {
        "unmodified_ff_count": unmodified_ff_count,
        "crossing_discovery_recall": recall,
        "n_flagged_capture_ffs": n_ocdc, "n_our_capture_ffs": n_ours,
        "detector": m, "detector_ci_design": ci,
        "flag_rate": float(flagged.mean()), "unsafe_base_rate": float(y.mean()),
    }

J.banner("Verdict")
for sp, r in OUT.items():
    print(f"  {sp:10s} discovers {r['crossing_discovery_recall']:.1%} of "
          f"construction crossings; as a detector F1={r['detector']['f1']:.3f} "
          f"at FPR={r['detector']['false_positive_rate']:.3f}")
print("\n  OpenCDC provides no safety verdict, so it is a crossing enumerator")
print("  rather than a classifier. Reported here as the external reference")
print("  point P0.5 lacked, not as a peer of a sign-off engine.")

J.save("j07_opencdc", OUT)
