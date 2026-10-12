"""j40 - extend the third-party corpus beyond the 16 designs of j12.

The corpus size is the binding constraint on every positive claim in the paper
and the first thing a reviewer will attack. This adds candidates from sources
j12 never tried, and attempts two recoveries from its failure list.

New sources:
  pulp-platform/axi          axi_cdc, a full AXI clock-domain crossing built
                             from cc_cdc_fifo_gray_{src,dst}. Substantially
                             larger than anything currently in the corpus.
  alexforencich/verilog-pcie dma_psdpram_async, plain Verilog, clk_wr/clk_rd.

Recovery attempt:
  ot_prim_lc_sync            failed in j12 with "could not find package
                             lc_ctrl_reg_pkg". Supply the package.

Reuses j12's fetch/resolve/synthesise machinery unchanged, so anything accepted
here is accepted on exactly the same terms as the existing 16: blackbox-free,
multi-clock, and structurally distinct from every design already in the corpus.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(JROOT))
sys.path.insert(0, str(SCRIPTS))

import jlib as J                                                   # noqa: E402
from corpus_tools import (find_blackboxes, find_clock_ports,        # noqa: E402
                          parse_netlist_structural, synthesize_multi,
                          sv2v_convert)

CORPUS = J.JROOT / "corpus"
NETLISTS = CORPUS / "netlists"
LEVELS = ["O0", "O1", "O2", "O3"]

NEW = [
    # axi_cdc needs two things j40's first pass lacked. axi_intf.sv supplies
    # the AXI_BUS interface that axi_cdc.sv's `axi_cdc_intf` wrapper declares
    # -- a missing source in OUR list, not a defect upstream. The patch works
    # around sv2v 0.0.13, which cannot parse a typedef inside a function body;
    # corpus/patches/README.md carries the full argument.
    # STILL BLOCKED, and now for a precisely identified reason. The patch
    # below gets it past sv2v, so the typedef is no longer the obstacle. What
    # remains is that axi_cdc declares its ports with `parameter type` struct
    # parameters; standalone, nothing binds them, so sv2v lowers them to plain
    # `logic` and Yosys then reads `dst_resp_i.r_valid` as an implicitly
    # declared identifier. That is the same `parameter type data_t = logic`
    # limitation Section V already reports, now with a concrete instance.
    #
    # Upstream's own concrete-typed wrapper, axi_cdc_intf, is not a way out:
    # it takes AXI_BUS *interface* ports, and sv2v cannot express those in
    # Verilog-2005, so it drops the module entirely -- the lowered file
    # contains only axi_cdc_src, axi_cdc_dst and axi_cdc.
    #
    # The only remaining route is a concrete-typed wrapper written by us,
    # which would make the design semi-synthetic and put it in the wrapped
    # corpus of Section V-H rather than the found corpus. We leave it out.
    dict(name="axi_cdc", repo="https://github.com/pulp-platform/axi",
         top="axi_cdc", sv=True,
         incdirs=["axi/include", "common_cells/include"],
         srcs=["axi/src/axi_pkg.sv",
               "axi/src/axi_intf.sv",
               "axi/src/axi_cdc_src.sv",
               "axi/src/axi_cdc_dst.sv",
               "axi/src/axi_cdc.sv",
               "common_cells/src/cc_cdc_fifo_gray.sv"],
         patch=[("      typedef shortint unsigned SU;\n",
                 "",
                 "removed in-function typedef from beat_upper_byte"),
                ("  /// Index of lowest byte in beat (see A3-51).\n",
                 "  // PATCHED: hoisted out of beat_upper_byte's body; sv2v\n"
                 "  // 0.0.13 cannot parse a typedef inside a function.\n"
                 "  typedef shortint unsigned SU;\n"
                 "  /// Index of lowest byte in beat (see A3-51).\n",
                 "hoisted SU typedef to package scope")]),
    dict(name="pcie_dma_psdpram_async",
         repo="https://github.com/alexforencich/verilog-pcie",
         top="dma_psdpram_async", sv=False,
         srcs=["verilog-pcie/rtl/dma_psdpram_async.v"]),

    # --- second pass: shortlisted by j45 from repos already on disk ---
    # ZipCPU's clock-crossing bridges. Plain Verilog, and wbxclk/axixclk build
    # on `afifo`, which is already in the corpus as wb2axip_afifo -- so expect
    # de-duplication to have something to say if the bridge is thin.
    dict(name="wb2axip_wbxclk", repo="https://github.com/ZipCPU/wb2axip",
         top="wbxclk", sv=False,
         srcs=["wb2axip/rtl/wbxclk.v", "wb2axip/rtl/afifo.v"]),
    dict(name="wb2axip_axixclk", repo="https://github.com/ZipCPU/wb2axip",
         top="axixclk", sv=False,
         srcs=["wb2axip/rtl/axixclk.v", "wb2axip/rtl/afifo.v"]),
    dict(name="wb2axip_apbxclk", repo="https://github.com/ZipCPU/wb2axip",
         top="apbxclk", sv=False,
         srcs=["wb2axip/rtl/apbxclk.v", "wb2axip/rtl/afifo.v"]),

    # OpenTitan CDC primitives j12 never tried, using the dependency set j12
    # established for the other prim modules.
    dict(name="ot_prim_edn_req", repo="https://github.com/lowRISC/opentitan",
         top="prim_edn_req", sv=True, defines=["SYNTHESIS"],
         incdirs=["opentitan/hw/ip/prim/rtl"],
         # prim_alert_pkg is a plain missing source, the same class of fix as
         # the lc_ctrl_reg_pkg recovery above -- not a tool limitation.
         srcs=["opentitan/hw/ip/prim/rtl/prim_alert_pkg.sv",
               "opentitan/hw/ip/prim/rtl/prim_edn_req.sv",
               "opentitan/hw/ip/prim/rtl/prim_sync_reqack_data.sv",
               "opentitan/hw/ip/prim/rtl/prim_sync_reqack.sv",
               "opentitan/hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "opentitan/hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_fifo_async_sram_adapter",
         repo="https://github.com/lowRISC/opentitan",
         top="prim_fifo_async_sram_adapter", sv=True,
         defines=["SYNTHESIS"], incdirs=["opentitan/hw/ip/prim/rtl"],
         srcs=["opentitan/hw/ip/prim/rtl/prim_fifo_async_sram_adapter.sv",
               "opentitan/hw/ip/prim/rtl/prim_sync_reqack.sv",
               "opentitan/hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "opentitan/hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_reg_cdc_arb",
         repo="https://github.com/lowRISC/opentitan",
         top="prim_reg_cdc_arb", sv=True,
         defines=["SYNTHESIS"], incdirs=["opentitan/hw/ip/prim/rtl"],
         # prim_pulse_sync is instantiated by prim_reg_cdc_arb and was simply
         # not on the list; without it Yosys blackboxes the cell, which the
         # blackbox guard would then reject anyway.
         srcs=["opentitan/hw/ip/prim/rtl/prim_reg_cdc_arb.sv",
               "opentitan/hw/ip/prim/rtl/prim_pulse_sync.sv",
               "opentitan/hw/ip/prim/rtl/prim_sync_reqack.sv",
               "opentitan/hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "opentitan/hw/ip/prim_generic/rtl/prim_flop.sv"]),
]


def apply_patch(cand):
    """Apply a documented source patch, in a scratch copy, before lowering.

    Third-party RTL enters this corpus unmodified. The exception is a source
    our TOOLCHAIN cannot read, and then only when the patched code provably
    cannot reach the extracted crossing. corpus/patches/README.md states the
    rule and carries the argument for each patch.

    The edit is made on a copy under corpus/patched/, never on the checked-out
    upstream tree, so the clone stays pristine and the diff stays visible.
    """
    spec = cand.get("patch")
    if not spec:
        return [CORPUS / s for s in cand["srcs"]], None

    dest = CORPUS / "patched" / cand["name"]
    out, applied = [], []
    for rel in cand["srcs"]:
        src = CORPUS / rel
        text = src.read_text(encoding="utf-8", errors="replace")
        for find, repl, why in spec:
            if find in text:
                text = text.replace(find, repl, 1)
                applied.append(why)
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_text(text, encoding="utf-8")
        out.append(tgt)
    # A patch that matched nothing is a silent no-op, and a silent no-op in a
    # build step is how a stale workaround outlives the thing it worked around.
    if not applied:
        raise RuntimeError(
            f"{cand['name']}: patch matched nothing -- upstream has changed, "
            f"re-read corpus/patches/README.md before trusting this result")
    print(f"    patched {cand['name']}: {'; '.join(applied)}")
    return out, applied


def try_design(cand):
    """Synthesise one candidate at all four levels and triage the result."""
    srcs, patched = apply_patch(cand)
    missing = [str(s) for s in srcs if not s.exists()]
    if missing:
        return dict(name=cand["name"], status="missing sources",
                    detail=missing[:3], usable=False)

    # Both helpers return (ok, message) rather than raising; a truthy return
    # value is not success, and treating it as such is how a hollow netlist
    # gets into a corpus.
    incdirs = [str(CORPUS / d) for d in cand.get("incdirs", [])]
    use = srcs
    if cand.get("sv"):
        conv = CORPUS / f"{cand['name']}.sv2v.v"
        # OpenTitan wraps its assertions in macros that sv2v cannot expand
        # unless SYNTHESIS is defined; j12 established this and the defines
        # were simply not plumbed through here.
        ok, msg = sv2v_convert(srcs, conv, incdirs=incdirs,
                               defines=cand.get("defines", ()))
        if not ok:
            return dict(name=cand["name"], status="sv2v failed",
                        detail=str(msg)[:200], usable=False)
        use = [conv]

    out = {}
    for lvl in LEVELS:
        jf = NETLISTS / f"{cand['name']}.{lvl}.json"
        ok, msg = synthesize_multi(use, cand["top"], jf, level=lvl,
                                   incdirs=incdirs, sv=False)
        if not ok:
            return dict(name=cand["name"], status=f"synth failed @{lvl}",
                        detail=str(msg)[:200], usable=False)
        out[lvl] = jf

    nl_json = json.loads(out["O0"].read_text())
    bb = find_blackboxes(nl_json)
    if bb:
        return dict(name=cand["name"], status="blackboxed",
                    detail=dict(list(bb.items())[:4]), usable=False)

    clocks, _, _, top = find_clock_ports(nl_json)
    nl = parse_netlist_structural(out["O0"])
    n_ff = len(nl.ff_idx)
    rec = dict(name=cand["name"], repo=cand["repo"], top=cand["top"],
               status="elaborated", n_clocks=len(clocks),
               clock_names=sorted(clocks), n_cells=len(nl.cells),
               n_ff=n_ff, sv2v=bool(cand.get("sv")),
               # Recorded so a patched design can never be mistaken for an
               # untouched one when the corpus table is read back.
               patched=patched,
               cell_hash=cell_hash(nl_json),
               usable=len(clocks) >= 2)
    return rec


def cell_hash(netlist_json):
    """Multiset of synthesized cell types, hashed. Same method as j12.

    Section V states that de-duplicating on the cell multiset rather than on
    file hash or module name is mandatory, because counting vendored copies as
    independent designs inflates the design count that sets every interval in
    the paper. The three ZipCPU bridges below all instantiate `afifo`, which is
    already in the corpus, so this is not a hypothetical concern.
    """
    cells = sorted(c["type"] for m in netlist_json["modules"].values()
                   for c in m.get("cells", {}).values())
    return hashlib.md5(repr(cells).encode()).hexdigest()[:12]


def main():
    J.banner("j40  corpus expansion")
    existing = json.loads(
        (J.RESULTS / "j12_corpus_triage.json").read_text())
    have = {r["name"] for r in existing}
    known_hashes = {r.get("cell_hash") for r in existing if r.get("cell_hash")}
    print(f"  existing corpus: {len(existing)} tried, "
          f"{sum(1 for r in existing if r.get('usable') and 'duplicate_of' not in r)} accepted")

    results = []
    for cand in NEW:
        if cand["name"] in have:
            print(f"  {cand['name']}: already tried, skipping")
            continue
        print(f"\n  --- {cand['name']} ---", flush=True)
        rec = try_design(cand)
        results.append(rec)
        if rec.get("usable"):
            print(f"      USABLE  clocks={rec['n_clocks']} "
                  f"FFs={rec['n_ff']} cells={rec['n_cells']} "
                  f"{rec['clock_names']}")
        else:
            print(f"      {rec['status']}: "
                  f"{str(rec.get('detail', ''))[:120]}")

    J.banner("De-duplication (cell multiset)")
    # Check new designs against the existing corpus AND against each other.
    seen = {r["cell_hash"]: r["name"] for r in existing
            if r.get("cell_hash") and r.get("usable")
            and "duplicate_of" not in r}
    ok = []
    for r in [x for x in results if x.get("usable")]:
        h = r["cell_hash"]
        if h in seen:
            r["duplicate_of"] = seen[h]
            r["usable"] = False
            print(f"  {r['name']:30s} DUPLICATE of {seen[h]}  (hash {h})")
        else:
            seen[h] = r["name"]
            ok.append(r)
            print(f"  {r['name']:30s} distinct        (hash {h})")

    J.banner("Result")
    print(f"  {len(ok)} of {len(results)} new candidates accepted "
          f"after de-duplication")
    for r in ok:
        print(f"    {r['name']:32s} {r['n_ff']:4d} FFs  "
              f"{r['n_cells']:5d} cells")
    if ok:
        # Derived, not the literal 16 this line used to carry: that was the
        # corpus size when j40 was first written and it stayed put through the
        # merge to 21. Exactly the drift j47 exists to catch in the paper.
        n_now = len({r["name"] for r in json.loads(
            (J.RESULTS / "j12_corpus_triage.json").read_text())
            if r.get("usable")})
        print(f"\n  corpus would grow from {n_now} to {n_now + len(ok)} "
              f"usable designs (before de-duplication)")
        print("  NOT yet merged into j12_corpus_triage.json -- de-duplicate "
              "against\n  the existing cell hashes first, then re-run j25/j32 "
              "and the audits.")
    J.save("j40_expand_corpus", dict(candidates=NEW, results=results,
                                     n_accepted=len(ok)))


if __name__ == "__main__":
    main()
