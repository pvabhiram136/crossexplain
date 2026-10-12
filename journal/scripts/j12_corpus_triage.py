"""P1.1 + P1.3 - triage candidate open RTL for the enlarged real corpus.

The plan (JOURNAL_PLAN.md Part 4, P1) calls for 40 to 60 *independent designs*,
since clustered interval width is set by the design count and not by the
crossing count. Before committing to any core, we need to know which ones
elaborate under yowasp-yosys at all, and which actually contain multiple clock
domains. Budget expectation from the plan was a 30 to 40% rejection rate.

This script fetches each candidate, synthesizes it at O0, and records:
  elaborate  did yosys produce a netlist
  clocks     distinct clock domains, found structurally (corpus_tools)
  FFs        flip-flop count, a proxy for whether the design is worth having
  crossings  capture flip-flops whose fan-in cone crosses a domain

A candidate is useful to us only if clocks >= 2 and crossings > 0.

Run:  python journal/scripts/j12_corpus_triage.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jlib as J                                              # noqa: E402
from corpus_tools import (synthesize_multi, quick_stats, find_clock_ports,  # noqa: E402
                          sv2v_convert, sv2v_available)

CORPUS = J.JROOT / "corpus"
CORPUS.mkdir(exist_ok=True)

# Candidates chosen for genuine multi-clock content: asynchronous FIFOs,
# CDC primitives, and peripheral cores with a separate interface clock.
# Plain Verilog is preferred; SystemVerilog is flagged since yosys support
# is partial.
CANDIDATES = [
    # --- AXI-stream / wishbone async FIFOs (plain Verilog) ---
    dict(name="axis_async_fifo", repo="https://github.com/alexforencich/verilog-axis",
         top="axis_async_fifo", srcs=["rtl/axis_async_fifo.v"], sv=False),
    dict(name="axis_async_adapter", repo="https://github.com/alexforencich/verilog-axis",
         top="axis_async_fifo_adapter", srcs=["rtl/axis_async_fifo_adapter.v",
                                              "rtl/axis_async_fifo.v",
                                              "rtl/axis_adapter.v"], sv=False),
    dict(name="wb2axip_afifo", repo="https://github.com/ZipCPU/wb2axip",
         top="afifo", srcs=["rtl/afifo.v"], sv=False),
    dict(name="eth_axis_async", repo="https://github.com/alexforencich/verilog-ethernet",
         top="axis_async_fifo", srcs=["lib/axis/rtl/axis_async_fifo.v"], sv=False),
    # --- basejump_stl CDC primitives, minimal dependency sets ---
    # Passing whole directories fails: bsg_misc contains constructs the wasm
    # yosys build cannot parse. Each entry lists only what its top needs.
    # bsg_mem_1r1w instantiates bsg_mem_1r1w_synth with a `.*` wildcard port
    # connection. sv2v can only expand `.*` if the target module is in the
    # same invocation, otherwise it passes the wildcard through and yosys
    # rejects it in Verilog-2005 mode. The synth file must be listed.
    dict(name="bsg_async_fifo", repo="https://github.com/bespoke-silicon-group/basejump_stl",
         top="bsg_async_fifo", sv=True, incdirs=["bsg_misc"],
         srcs=["bsg_async/bsg_async_fifo.sv", "bsg_async/bsg_async_ptr_gray.sv",
               "bsg_async/bsg_sync_sync.sv", "bsg_misc/bsg_and.sv",
               "bsg_misc/bsg_mux.sv", "bsg_misc/bsg_dff.sv",
               "bsg_misc/bsg_dff_reset.sv", "bsg_misc/bsg_gray_to_binary.sv",
               "bsg_mem/bsg_mem_1r1w.sv", "bsg_mem/bsg_mem_1r1w_synth.sv"]),
    dict(name="bsg_async_ptr_gray", repo="https://github.com/bespoke-silicon-group/basejump_stl",
         top="bsg_async_ptr_gray", sv=True, incdirs=["bsg_misc"],
         srcs=["bsg_async/bsg_async_ptr_gray.sv", "bsg_async/bsg_sync_sync.sv",
               "bsg_misc/bsg_dff.sv", "bsg_misc/bsg_dff_reset.sv",
               "bsg_misc/bsg_gray_to_binary.sv", "bsg_misc/bsg_mux.sv"]),
    dict(name="bsg_launch_sync_sync", repo="https://github.com/bespoke-silicon-group/basejump_stl",
         top="bsg_launch_sync_sync", sv=True, incdirs=["bsg_misc"],
         srcs=["bsg_async/bsg_launch_sync_sync.sv", "bsg_async/bsg_sync_sync.sv",
               "bsg_misc/bsg_dff.sv", "bsg_misc/bsg_dff_reset.sv"]),
    dict(name="bsg_sync_sync", repo="https://github.com/bespoke-silicon-group/basejump_stl",
         top="bsg_sync_sync", sv=True, incdirs=["bsg_misc"],
         srcs=["bsg_async/bsg_sync_sync.sv"]),
    dict(name="bsg_async_credit_counter",
         repo="https://github.com/bespoke-silicon-group/basejump_stl",
         top="bsg_async_credit_counter", sv=True, incdirs=["bsg_misc"],
         srcs=["bsg_async/bsg_async_credit_counter.sv",
               "bsg_async/bsg_async_ptr_gray.sv", "bsg_async/bsg_sync_sync.sv",
               "bsg_misc/bsg_dff.sv", "bsg_misc/bsg_dff_reset.sv",
               "bsg_misc/bsg_gray_to_binary.sv", "bsg_misc/bsg_mux.sv",
               "bsg_misc/bsg_counter_w_overflow.sv",
               "bsg_misc/bsg_counter_up_down_variable.sv"]),
    # --- pulp common_cells CDC family ---
    dict(name="pulp_cdc_2phase", repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_2phase", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], srcs=["src/cc_cdc_2phase.sv"],
         incdirs=["include"], sv=True),
    dict(name="pulp_cdc_4phase", repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_4phase", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], incdirs=["include"], sv=True,
         srcs=["src/cc_cdc_4phase.sv", "src/cc_cdc_2phase.sv",
               "src/cc_spill_register.sv",
               "src/cc_spill_register_flushable.sv"]),
    dict(name="pulp_cdc_fifo_gray", repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_fifo_gray", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], incdirs=["include"], sv=True,
         srcs=["src/cc_cdc_fifo_gray.sv", "src/cc_cdc_2phase.sv",
               "src/cc_binary_to_gray.sv", "src/cc_gray_to_binary.sv",
               "src/cc_spill_register.sv",
               "src/cc_spill_register_flushable.sv"]),
    dict(name="pulp_cdc_fifo_2phase", repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_fifo_2phase", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], incdirs=["include"], sv=True,
         srcs=["src/cc_cdc_fifo_2phase.sv", "src/cc_cdc_2phase.sv",
               "src/cc_binary_to_gray.sv", "src/cc_gray_to_binary.sv"]),
    dict(name="pulp_cdc_2phase_clearable",
         repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_2phase_clearable", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], incdirs=["include"], sv=True,
         srcs=["src/cc_pkg.sv", "src/cc_cdc_2phase_clearable.sv", "src/cc_cdc_2phase.sv",
               "src/cc_cdc_reset_ctrlr.sv", "src/cc_cdc_4phase.sv"]),
    dict(name="pulp_cdc_reset_ctrlr",
         repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_reset_ctrlr", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], incdirs=["include"], sv=True,
         srcs=["src/cc_pkg.sv", "src/cc_cdc_reset_ctrlr.sv", "src/cc_cdc_2phase.sv",
               "src/cc_cdc_4phase.sv"]),
    dict(name="pulp_cdc_fifo_gray_clearable",
         repo="https://github.com/pulp-platform/common_cells",
         top="cc_cdc_fifo_gray_clearable", extra_repos=[dict(repo="https://github.com/pulp-platform/tech_cells_generic",
                           srcs=["src/rtl/tc_sync.sv"])], incdirs=["include"], sv=True,
         srcs=["src/cc_pkg.sv", "src/cc_cdc_fifo_gray_clearable.sv", "src/cc_cdc_2phase.sv",
               "src/cc_cdc_reset_ctrlr.sv", "src/cc_cdc_4phase.sv",
               "src/cc_binary_to_gray.sv", "src/cc_gray_to_binary.sv",
               "src/cc_spill_register.sv",
               "src/cc_spill_register_flushable.sv"]),
    # --- OpenTitan prim CDC modules (sparse checkout; repo is very large) ---
    dict(name="ot_prim_fifo_async", repo="https://github.com/lowRISC/opentitan",
         top="prim_fifo_async", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_fifo_async.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_fifo_async_simple", repo="https://github.com/lowRISC/opentitan",
         top="prim_fifo_async_simple", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_fifo_async_simple.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv",
               "hw/ip/prim/rtl/prim_sync_reqack_data.sv",
               "hw/ip/prim/rtl/prim_sync_reqack.sv"]),
    dict(name="ot_prim_sync_reqack", repo="https://github.com/lowRISC/opentitan",
         top="prim_sync_reqack", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_sync_reqack.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_sync_reqack_data", repo="https://github.com/lowRISC/opentitan",
         top="prim_sync_reqack_data", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_sync_reqack_data.sv",
               "hw/ip/prim/rtl/prim_sync_reqack.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_pulse_sync", repo="https://github.com/lowRISC/opentitan",
         top="prim_pulse_sync", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_pulse_sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_reg_cdc", repo="https://github.com/lowRISC/opentitan",
         top="prim_reg_cdc", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_reg_cdc.sv",
               "hw/ip/prim/rtl/prim_reg_cdc_arb.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv",
               "hw/ip/prim/rtl/prim_pulse_sync.sv"]),
    # --- remaining OpenTitan synchronizer modules ---
    dict(name="ot_prim_multibit_sync", repo="https://github.com/lowRISC/opentitan",
         top="prim_multibit_sync", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_multibit_sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_sync_slow_fast", repo="https://github.com/lowRISC/opentitan",
         top="prim_sync_slow_fast", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_sync_slow_fast.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_lc_sync", repo="https://github.com/lowRISC/opentitan",
         top="prim_lc_sync", sv=True,
         sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl", "hw/ip/lc_ctrl/rtl"],
         defines=["SYNTHESIS"], incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_util_pkg.sv",
               "hw/ip/lc_ctrl/rtl/lc_ctrl_pkg.sv",
               "hw/ip/lc_ctrl/rtl/lc_ctrl_state_pkg.sv",
               "hw/ip/prim/rtl/prim_lc_sync.sv",
               "hw/ip/prim/rtl/prim_sec_anchor_buf.sv",
               "hw/ip/prim_generic/rtl/prim_buf.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_mubi4_sync", repo="https://github.com/lowRISC/opentitan",
         top="prim_mubi4_sync", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_mubi_pkg.sv",
               "hw/ip/prim/rtl/prim_mubi4_sync.sv",
               "hw/ip/prim/rtl/prim_sec_anchor_buf.sv",
               "hw/ip/prim_generic/rtl/prim_buf.sv",
               "hw/ip/prim_generic/rtl/prim_xor2.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    dict(name="ot_prim_clock_gating_sync", repo="https://github.com/lowRISC/opentitan",
         top="prim_clock_gating_sync", sv=True, sparse=["hw/ip/prim/rtl", "hw/ip/prim_generic/rtl"], defines=["SYNTHESIS"],
         incdirs=["hw/ip/prim/rtl"],
         srcs=["hw/ip/prim/rtl/prim_clock_gating_sync.sv",
               "hw/ip/prim_generic/rtl/prim_clock_gating.sv",
               "hw/ip/prim_generic/rtl/prim_flop_2sync.sv",
               "hw/ip/prim_generic/rtl/prim_flop.sv"]),
    # --- single-clock peripheral cores, kept to measure the rejection rate ---
    dict(name="uart16550", repo="https://github.com/freecores/uart16550",
         top="uart_top", srcs=["rtl/verilog"], sv=False,
         incdirs=["rtl/verilog"], glob="*.v"),
    dict(name="i2c_master", repo="https://github.com/freecores/i2c",
         top="i2c_master_top", srcs=["rtl/verilog"], sv=False,
         incdirs=["rtl/verilog"], glob="*.v"),
    dict(name="simple_spi", repo="https://github.com/freecores/simple_spi",
         top="simple_spi_top", srcs=["rtl/verilog"], sv=False,
         incdirs=["rtl/verilog"], glob="*.v"),
]


def fetch(cand):
    dest = CORPUS / cand["repo"].rstrip("/").split("/")[-1]
    if dest.exists():
        return dest, "cached"
    sparse = cand.get("sparse")
    try:
        if sparse:
            # OpenTitan is far too large to clone whole for a handful of prim
            # files, so use a blobless sparse checkout of just the paths we
            # need.
            r = subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none",
                 "--sparse", "--quiet", cand["repo"], str(dest)],
                capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                r = subprocess.run(
                    ["git", "-C", str(dest), "sparse-checkout", "set", *sparse],
                    capture_output=True, text=True, timeout=900)
        else:
            r = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", cand["repo"],
                 str(dest)],
                capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "clone timeout"
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["clone failed"])[-1][:120]
    return dest, "cloned"


def resolve_sources(root: Path, cand):
    """Resolve the manifest's source list, reporting anything missing.

    Silently dropping a nonexistent path hides manifest errors: the design
    then fails later with a confusing message about an undefined module, or
    worse, succeeds with a silently pruned hierarchy.
    """
    out, missing = [], []
    for s in cand["srcs"]:
        p = root / s
        if p.is_dir():
            out.extend(sorted(p.glob(cand.get("glob", "*.v"))))
        elif p.exists():
            out.append(p)
        else:
            missing.append(s)
    return out, missing


J.banner("P1.3  Corpus triage: which open cores can we actually use?")
print(f"  {len(CANDIDATES)} candidates, target is 40 to 60 usable designs\n")

rows = []
for cand in CANDIDATES:
    t0 = time.time()
    row = dict(name=cand["name"], repo=cand["repo"], top=cand["top"])
    root, how = fetch(cand)
    if root is None:
        row.update(status="fetch failed", detail=how)
        rows.append(row)
        print(f"  {cand['name']:24s} FETCH FAILED  {how}")
        continue
    row["fetch"] = how

    srcs, missing = resolve_sources(root, cand)

    # Some cores depend on technology-cell libraries kept in a separate
    # repository. pulp's cdc_* modules instantiate tc_sync, which lives in
    # pulp-platform/tech_cells_generic; without it yosys blackboxes the
    # synchronizer and the netlist is hollow exactly where the crossing is.
    for extra in cand.get("extra_repos", []):
        eroot, ehow = fetch(extra)
        if eroot is None:
            row.update(status="fetch failed", detail=f"{extra['repo']}: {ehow}")
            rows.append(row)
            print(f"  {cand['name']:24s} DEP FETCH FAILED  {ehow}")
            break
        esrcs, emissing = resolve_sources(eroot, extra)
        srcs.extend(esrcs)
        missing.extend(f"{extra['repo'].split('/')[-1]}:{m}" for m in emissing)
    else:
        pass
    if row.get("status") == "fetch failed":
        continue

    if missing:
        row["missing_sources"] = missing
        print(f"  {cand['name']:24s} WARN missing sources: {missing}")
    if not srcs:
        row.update(status="no sources", detail=str(cand["srcs"]))
        rows.append(row)
        print(f"  {cand['name']:24s} NO SOURCES   {cand['srcs']}")
        continue
    row["n_files"] = len(srcs)
    incs = [root / d for d in cand.get("incdirs", [])]

    # SystemVerilog sources are lowered to Verilog-2005 by sv2v first; the
    # yosys frontend cannot parse `parameter type` or token-pasted module
    # names. Recorded so the triage table shows which designs needed it.
    if cand.get("sv"):
        conv = CORPUS / "sv2v" / f"{cand['name']}.v"
        ok, msg = sv2v_convert(srcs, conv, incdirs=incs,
                               defines=cand.get("defines", ()))
        row["sv2v"] = "ok" if ok else msg
        if not ok:
            row.update(status="sv2v failed", detail=msg)
            rows.append(row)
            print(f"  {cand['name']:24s} SV2V FAILED  {msg[:80]}")
            continue
        srcs, incs = [conv], []

    jf = CORPUS / "netlists" / f"{cand['name']}.O0.json"
    ok, msg = synthesize_multi(
        srcs, cand["top"], jf, "O0", incdirs=incs,
        sv=False, cwd=root)
    if not ok:
        row.update(status="elaborate failed", detail=msg)
        rows.append(row)
        print(f"  {cand['name']:24s} ELAB FAILED  {msg[:90]}")
        continue

    st = quick_stats(jf, cand["top"])
    # An empty netlist is a failure, not a single-clock design. It means
    # `hierarchy -top` pruned everything, usually because the real top is a
    # macro-generated name that does not match what we asked for. Counting it
    # as "1-clock" would misreport the reason, and letting it reach the
    # de-duplicator would make every empty design a duplicate of every other.
    # A blackboxed module means the netlist is hollow where that module was.
    # For CDC work this is usually the synchronizer itself, so the design must
    # be rejected rather than measured. See corpus_tools.find_blackboxes.
    if st["blackboxes"]:
        row.update(status="blackboxed", detail=str(st["blackboxes"]),
                   n_cells=st["n_cells"], n_ff=st["n_ff"])
        rows.append(row)
        print(f"  {cand['name']:24s} BLACKBOX     undefined: {st['blackboxes']}")
        continue

    if st["n_cells"] == 0 or st["n_ff"] == 0:
        row.update(status="empty netlist",
                   detail=f"top '{cand['top']}' resolved to 0 cells / 0 FFs",
                   n_cells=st["n_cells"], n_ff=st["n_ff"])
        rows.append(row)
        print(f"  {cand['name']:24s} EMPTY        top pruned to nothing")
        continue

    row.update(status="elaborated", **{k: st[k] for k in
                                       ("n_clocks", "n_cells", "n_ff")})
    row["clock_names"] = st["clocks"][:6]

    # crossing census using the structural clock map
    n_cross = 0
    try:
        sys.path.insert(0, str(J.ROOT))
        import crossexplain.netlist_graph as NG
        data = json.loads(jf.read_text())
        clocks, clock_bits, tname, top = find_clock_ports(data, cand["top"])
        if len(clocks) >= 2:
            n_cross = -1     # measured in P1.4 once extraction is generalized
    except Exception as e:                                     # noqa: BLE001
        row["detail"] = f"census error: {e}"
    row["usable"] = bool(st["n_clocks"] >= 2 and st["n_ff"] > 0)
    row["secs"] = round(time.time() - t0, 1)
    rows.append(row)
    flag = "USABLE " if row["usable"] else "1-clock"
    print(f"  {cand['name']:24s} {flag}  clocks={st['n_clocks']:2d} "
          f"FFs={st['n_ff']:5d} cells={st['n_cells']:6d}  "
          f"{st['clocks'][:4]}  ({row['secs']}s)")

J.banner("De-duplication")
# The same async FIFO is vendored across many repositories. Counting
# repositories would badly overcount independent designs, and independence is
# exactly what the clustered intervals depend on, so duplicates are worthless:
# they are perfectly correlated. Identity is the multiset of synthesized cell
# types plus the flip-flop count.
import hashlib                                                # noqa: E402

seen = {}
for r in rows:
    if r.get("status") != "elaborated":
        continue
    jf = CORPUS / "netlists" / f"{r['name']}.O0.json"
    d = json.loads(jf.read_text())
    cells = sorted(c["type"] for m in d["modules"].values()
                   for c in m.get("cells", {}).values())
    h = hashlib.md5(repr(cells).encode()).hexdigest()[:12]
    r["cell_hash"] = h
    if h in seen:
        r["duplicate_of"] = seen[h]
        print(f"  {r['name']:24s} DUPLICATE of {seen[h]}  (hash {h})")
    else:
        seen[h] = r["name"]

J.banner("Triage summary")
ok = [r for r in rows if r.get("usable")]
distinct = [r for r in ok if "duplicate_of" not in r]
elab = [r for r in rows if r.get("status") == "elaborated"]
print(f"  candidates tried      : {len(rows)}")
print(f"  elaborated            : {len(elab)}")
print(f"  multi-clock           : {len(ok)}")
print(f"  multi-clock, DISTINCT : {len(distinct)}   <- the number that counts")
print(f"  rejection rate        : "
      f"{100 * (1 - len(distinct) / max(1, len(rows))):.0f}%"
      f"   (plan budgeted 30 to 40%)")
if distinct:
    print(f"\n  distinct usable designs: {[r['name'] for r in distinct]}")
    print(f"  total FFs across them  : {sum(r['n_ff'] for r in distinct)}")
    print(f"\n  progress toward P1 target of 40 to 60 designs: "
          f"{len(distinct)}/40 ({100 * len(distinct) / 40:.0f}%)")
fails = [r for r in rows if r.get("status") not in ("elaborated",)]
if fails:
    print("\n  rejected:")
    for r in fails:
        print(f"    {r['name']:24s} {r.get('status'):18s} {str(r.get('detail'))[:80]}")

J.save("j12_corpus_triage", rows)
