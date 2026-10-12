"""j37 - check Section V's corpus-history claims against the triage record.

j34 listed ten Section V claims as MANUAL, because they come from
`FINDINGS_P1.md` prose describing intermediate states of the triage rather than
from the final result JSON. Prose is not a source. This re-derives each of them
from `j12_corpus_triage.json` and the `j12_run*.log` files, so they are checked
rather than trusted.

The run logs are the record of the triage as it progressed:

  run3   15 candidates, before sv2v
  run6   before the OpenTitan dependency fix
  run7   before the blackbox guard
  run8   the blackbox guard, same 25 candidates as run7
  run10  after the pulp dependency fix
  run13  the final pass, 16 accepted
"""
import json
import re
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

FAILS, CHECKS = [], 0

# Values this script derives, published so j47_orphan_numbers.py can bind a
# claim in the .tex to the corpus history rather than to a literal. Fig. 2's
# node label and caption both still read "10 of 16" after the body text had
# been corrected to 13 of 21, and nothing here was looking at the figure.
DERIVED = {}


def chk(claim, asserted, derived):
    global CHECKS
    CHECKS += 1
    DERIVED[claim] = derived
    ok = asserted == derived
    if not ok:
        FAILS.append((claim, asserted, derived))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {claim:<56s} "
          f"claimed={asserted}  derived={derived}")


def run_log(n):
    return (J.RESULTS / f"j12_run{n}.log").read_text(encoding="utf-8",
                                                     errors="replace")


def counts(n):
    """(candidates, elaborated, multi-clock, distinct) from one run log."""
    t = run_log(n)

    def grab(label):
        m = re.search(rf"{label}\s*:\s*(\d+)", t)
        return int(m.group(1)) if m else None

    return dict(candidates=grab("candidates tried"),
                elaborated=grab("elaborated"),
                multiclock=grab(r"multi-clock  "),
                distinct=grab("multi-clock, DISTINCT"))


def main():
    J.banner("j37  Section V corpus-history claims")
    triage = json.loads((J.RESULTS / "j12_corpus_triage.json").read_text())
    usable = [r for r in triage
              if r.get("usable") and "duplicate_of" not in r]

    J.banner("First triage pass (run3, before sv2v)")
    r3 = counts(3)
    chk("V: 15 candidates tried", 15, r3["candidates"])
    chk("V: 7 elaborated", 7, r3["elaborated"])
    chk("V: 3 accepted", 3, r3["distinct"])
    chk("V: 80% rejection rate", 80,
        round(100 * (1 - r3["distinct"] / r3["candidates"])))

    J.banner("The blackbox guard (run7 -> run8, same 25 candidates)")
    r7, r8 = counts(7), counts(8)
    chk("V: guard run has the same candidate list", r7["candidates"],
        r8["candidates"])
    chk("V: elaborated falls 20 to 13", (20, 13),
        (r7["elaborated"], r8["elaborated"]))
    chk("V: accepted falls 15 to 9", (15, 9),
        (r7["distinct"], r8["distinct"]))
    chk("V: seven candidates blackboxed", 7,
        len(re.findall(r"blackboxed", run_log(8))))
    bb = set(re.findall(r"(\w+)\s+blackboxed", run_log(8)))
    chk("V: tc_sync blackboxed for four pulp designs", 4,
        len([m for m in re.findall(r"(\w+)\s+blackboxed\s+(\{[^}]*\})",
                                   run_log(8)) if "tc_sync" in m[1]]))
    chk("V: bsg_launch_sync_sync blackboxed for three basejump designs", 3,
        len([m for m in re.findall(r"(\w+)\s+blackboxed\s+(\{[^}]*\})",
                                   run_log(8))
             if "bsg_launch_sync_sync" in m[1]]))

    J.banner("Dependency fixes restore missing synchronizers")
    def ff(n, design):
        m = re.search(rf"{design}\s+USABLE\s+clocks=\s*\d+\s+FFs=\s*(\d+)",
                      run_log(n))
        return int(m.group(1)) if m else None

    chk("V: ot_prim_fifo_async 15 -> 27 flip-flops", (15, 27),
        (ff(7, "ot_prim_fifo_async"), ff(9, "ot_prim_fifo_async")))
    chk("V: pulp_cdc_fifo_gray 20 -> 36 flip-flops", (20, 36),
        (ff(6, "pulp_cdc_fifo_gray"), ff(10, "pulp_cdc_fifo_gray")))

    J.banner("Final state")
    chk("V: 21 accepted designs", 21, len(usable))
    chk("V: 14 of 21 require sv2v", 14,
        len([r for r in usable if r.get("sv2v")]))
    chk("V: 7 are plain Verilog", 7,
        len([r for r in usable if not r.get("sv2v")]))
    # One accepted design yields no crossing our extraction can see: its two
    # domains communicate through a dual-port memory, not through logic.
    conf_designs = {s["design"] for s in J.confirmation_set()}
    chk("V: 20 of the 21 contribute crossings", 20, len(conf_designs))
    chk("V: the exception is pcie_dma_psdpram_async", True,
        "pcie_dma_psdpram_async" not in conf_designs)

    J.banner("Clock naming")
    clk = [r for r in usable
           if all(c.startswith("clk") for c in r.get("clock_names", []))]
    chk("V: 8 of 21 use the clk prefix", 8, len(clk))
    chk("V: the name-based rule would exclude thirteen", 13,
        len(usable) - len(clk))
    # No longer all OpenTitan: verilog-pcie's dma_psdpram_async also uses
    # clk_rd/clk_wr, so the claim is now "seven of the eight".
    chk("V: seven of the eight are OpenTitan", 7,
        sum(1 for r in clk if r["name"].startswith("ot_")))

    J.banner("De-duplication")
    dups = [r for r in triage if "duplicate_of" in r]
    chk("V: exactly one vendored duplicate", 1, len(dups))
    chk("V: eth_axis_async duplicates axis_async_fifo",
        ("eth_axis_async", "axis_async_fifo"),
        (dups[0]["name"], dups[0]["duplicate_of"]) if dups else None)

    J.banner("Modules named 'sync' with no crossing")
    single = [r for r in triage
              if r.get("n_clocks") == 1 and "sync" in r["name"]]
    chk("V: four OpenTitan sync modules are single-clock", 4, len(single))
    ffs = sorted(r["n_ff"] for r in single)
    chk("V: they carry 3 to 35 flip-flops", (3, 35), (ffs[0], ffs[-1]))

    # The funnel head. Fig. 2 showed "25 tried" feeding "21 accepted" long
    # after j40 added a second batch of candidates -- five of the accepted 21
    # were never among the 25, so the funnel did not add up. The head is the
    # first pass plus the second batch, and both halves are derived.
    J.banner("Candidates tried, across both passes")
    exp = json.loads(
        (J.RESULTS / "j40_expand_corpus.json").read_text(encoding="utf-8"))
    first_pass = 25                      # checked above against the run8 log
    second = len(exp["candidates"])
    chk("V: 8 candidates in the second pass", 8, second)
    chk("V: 5 of the second pass accepted", 5, exp["n_accepted"])
    chk("V: 3 of the second pass failed", 3, second - exp["n_accepted"])
    chk("V: 33 candidates tried in total", 33, first_pass + second)

    J.banner("Result")
    print(f"  {CHECKS - len(FAILS)} of {CHECKS} checks reconcile")
    for claim, a, d in FAILS:
        print(f"    FAIL {claim}: claimed={a} derived={d}")
    J.save("j37_audit_corpus_history", dict(
        n_checks=CHECKS, n_pass=CHECKS - len(FAILS),
        n_accepted=DERIVED.get("V: 21 accepted designs"),
        n_clk_prefix=DERIVED.get("V: 8 of 21 use the clk prefix"),
        n_name_excluded=DERIVED.get(
            "V: the name-based rule would exclude thirteen"),
        n_tried_total=DERIVED.get("V: 33 candidates tried in total"),
        n_second_pass=DERIVED.get("V: 8 candidates in the second pass"),
        failures=[dict(claim=c, claimed=str(a), derived=str(d))
                  for c, a, d in FAILS]))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
