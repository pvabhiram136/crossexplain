"""P1 support: make the pipeline work on RTL we did not write.

Two things block external designs.

1. Clock discovery is name-based. `netlist_graph.parse_netlist` treats a port
   as a clock only if its name starts with `clk`, which is true of our
   generator and of nothing else. Real cores use `wb_clk_i`, `s_axis_aclk`,
   `i_rd_clk`, `PCLK`, and so on.

   The fix here is structural and needs no naming convention at all: a clock
   is any signal that reaches the clock pin of at least one flip-flop. We
   trace each flip-flop's clock pin backwards through buffers and inverters to
   its ultimate driver, and take the set of distinct drivers as the clock set.
   This also survives clock buffering and simple clock gating, which the
   name-based version does not.

2. Synthesis assumes one file and a top module literally called `top`.
   External cores are multi-file with arbitrary top names, and often need an
   include path.

Neither change touches the conference artifact; both live here until P1
settles, at which point the clock discovery should be promoted into
`crossexplain/netlist_graph.py` alongside `journal/grouping.py`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

BUF_TYPES = {"$_BUF_", "$_NOT_", "$buf", "$not"}
CLK_PINS = ("CLK", "C", "clk", "clock", "CK")


def _is_ff_type(t: str) -> bool:
    tl = t.lower()
    return (tl.startswith("$_dff") or tl.startswith("$_sdff")
            or tl.startswith("$_dffe") or tl.startswith("$_sdffe")
            or tl.startswith("$_dlatch") or tl.startswith("$_aldff")
            or tl in {"$dff", "$dffe", "$sdff", "$adff", "$aldff", "$dlatch"})


def find_clock_ports(netlist_json: dict, top_name: str | None = None):
    """Structurally identify clock nets and the ports that drive them.

    Returns (clock_port_names, clock_bit_to_index, top_module_dict).
    """
    mods = netlist_json["modules"]
    top = None
    if top_name and top_name in mods:
        top = mods[top_name]
    else:
        for name, m in mods.items():
            attrs = m.get("attributes", {})
            if attrs.get("top") in (1, "1",
                                    "00000000000000000000000000000001"):
                top = m
                top_name = name
                break
    if top is None:
        top_name, top = next(iter(mods.items()))

    ports = top.get("ports", {})
    cells = top.get("cells", {})

    # driver map: output bit -> (cell name, cell type)
    driver = {}
    for cname, cj in cells.items():
        dirs = cj.get("port_directions", {})
        for pname, bits in cj.get("connections", {}).items():
            d = dirs.get(pname) or ("output" if pname in ("Y", "Q") else "input")
            if d != "output":
                continue
            for b in bits:
                if isinstance(b, int):
                    driver[b] = (cname, cj["type"])

    inputs = {}
    for pname, pj in ports.items():
        if pj["direction"] == "input":
            for b in pj["bits"]:
                if isinstance(b, int):
                    inputs[b] = pname

    # every bit that lands on a flip-flop clock pin
    clk_bits = set()
    for cname, cj in cells.items():
        if not _is_ff_type(cj["type"]):
            continue
        conns = cj.get("connections", {})
        for pin in CLK_PINS:
            if pin in conns:
                for b in conns[pin]:
                    if isinstance(b, int):
                        clk_bits.add(b)
                break

    # walk each clock bit back through buffers/inverters to its source
    def root_of(bit, limit=32):
        seen = set()
        cur = bit
        for _ in range(limit):
            if cur in inputs or cur in seen:
                return cur
            seen.add(cur)
            drv = driver.get(cur)
            if drv is None:
                return cur
            cname, ctype = drv
            if ctype not in BUF_TYPES:
                return cur              # driven by real logic: gated clock
            cj = cells[cname]
            src = [b for p, bs in cj.get("connections", {}).items()
                   if p in ("A", "a") for b in bs if isinstance(b, int)]
            if not src:
                return cur
            cur = src[0]
        return cur

    roots = defaultdict(set)
    for b in clk_bits:
        roots[root_of(b)].add(b)

    # name each clock domain by its driving port where possible
    named = []
    for r, bits in roots.items():
        named.append((inputs.get(r, f"__gen_clk_{r}"), r, bits))
    named.sort(key=lambda x: x[0])

    clock_ports = [n for n, _, _ in named]
    clock_bits = {}
    for idx, (_, r, bits) in enumerate(named):
        clock_bits[r] = idx
        for b in bits:
            clock_bits[b] = idx
    return clock_ports, clock_bits, top_name, top


# ------------------------------------------------------------ SystemVerilog
# Yosys 0.67's Verilog frontend covers a subset of SystemVerilog. The CDC
# libraries worth having (pulp common_cells, basejump_stl, OpenTitan prims)
# sit outside it: `parameter type` and preprocessor token-pasted module names
# both fail to parse. sv2v (BSD 3-Clause, zachjs/sv2v v0.0.13) lowers them to
# Verilog-2005 first. It is needed only to BUILD the corpus; reproducing
# results from released netlists stays pip-only.
SV2V = Path(__file__).resolve().parent / "tools" / "sv2v-Windows" / "sv2v.exe"


def sv2v_available() -> bool:
    return SV2V.exists()


def sv2v_convert(sources, out_v: Path, incdirs=(), defines=(), timeout=300):
    """Lower SystemVerilog sources to a single Verilog-2005 file.

    Returns (ok, message). All sources are converted together so that
    cross-file type and parameter resolution works.
    """
    out_v = Path(out_v)
    out_v.parent.mkdir(parents=True, exist_ok=True)
    if not SV2V.exists():
        return False, "sv2v not installed"
    cmd = [str(SV2V)]
    cmd += [f"-I{Path(d).as_posix()}" for d in incdirs]
    cmd += [f"-D{d}" for d in defines]
    cmd += [Path(s).as_posix() for s in sources]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"sv2v timeout after {timeout}s"
    if r.returncode != 0:
        lines = [x.strip() for x in (r.stderr or r.stdout or "").splitlines()
                 if x.strip()]
        # sv2v emits many "Warning:" lines before the fatal one; taking the
        # first few hid the actual cause during triage. Prefer real errors.
        errs = [x for x in lines
                if "Warning:" not in x and ("rror" in x or "could not" in x
                                            or "does not exist" in x
                                            or "Parse" in x)]
        pick = errs or lines[-3:]
        return False, " | ".join(pick[:3])[:300]
    if not r.stdout.strip():
        return False, "sv2v produced empty output"
    out_v.write_text(r.stdout, encoding="utf-8")
    return True, "ok"


# ------------------------------------------------------------------ synthesis
_SCRIPTS = {
    "O0": "{read}; hierarchy -top {top}; proc; flatten; techmap; opt_clean; "
          "write_json {jfile}",
    "O1": "{read}; hierarchy -top {top}; proc; flatten; opt; techmap; opt; "
          "write_json {jfile}",
    "O2": "{read}; hierarchy -top {top}; synth -top {top} -flatten -noabc; "
          "write_json {jfile}",
    "O3": "{read}; hierarchy -top {top}; synth -top {top} -flatten -noabc; "
          "share -aggressive; opt -full; opt_merge -share_all; opt -full; "
          "write_json {jfile}",
}


def synthesize_multi(sources, top: str, jfile: Path, level: str = "O0",
                     incdirs=(), sv: bool = False, timeout: int = 600,
                     cwd: Path | None = None):
    """Synthesize a multi-file external design.

    Returns (ok, message). `message` carries the yosys tail on failure, which
    is what the triage table records.
    """
    jfile = Path(jfile)
    jfile.parent.mkdir(parents=True, exist_ok=True)
    flags = "-sv " if sv else ""
    inc = "".join(f"-I{Path(d).as_posix()} " for d in incdirs)
    files = " ".join(Path(s).as_posix() for s in sources)
    read = f"read_verilog {flags}{inc}{files}"
    script = _SCRIPTS[level].format(
        read=read, top=top, jfile=jfile.resolve().as_posix())
    code = ("import sys, yowasp_yosys\n"
            f"sys.exit(yowasp_yosys.run_yosys({['-q', '-p', script]!r}))\n")
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    if r.returncode != 0 or not jfile.exists():
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        msg = " | ".join(t.strip() for t in tail[-3:]) if tail else "unknown"
        return False, msg[:300]
    return True, "ok"


def find_blackboxes(netlist_json: dict):
    """Cell types instantiated but never defined, with instance counts.

    This guard exists because of a real near-miss. OpenTitan's
    `prim_fifo_async` instantiates `prim_flop_2sync` for both pointer
    synchronizers, but that module is an abstract prim whose implementation
    lives in a different directory. Without it, yosys silently emits a
    blackbox: the design still elaborates, still reports two clock domains,
    and still looks usable, while the actual synchronizer flip-flops, that is
    the entire crossing, are absent from the netlist.

    Any design with a blackbox on the crossing path is unusable as CDC ground
    truth, so triage must reject it rather than quietly measure a hollow
    circuit.
    """
    mods = set(netlist_json["modules"])
    counts = {}
    for m in netlist_json["modules"].values():
        for c in m.get("cells", {}).values():
            t = c["type"]
            if not t.startswith("$") and t not in mods:
                counts[t] = counts.get(t, 0) + 1
    return counts


def quick_stats(jfile: Path, top_name: str | None = None):
    """Clock and flip-flop census for a synthesized external design."""
    data = json.loads(Path(jfile).read_text())
    clocks, clock_bits, tname, top = find_clock_ports(data, top_name)
    cells = top.get("cells", {})
    n_ff = sum(1 for c in cells.values() if _is_ff_type(c["type"]))
    return dict(top=tname, clocks=clocks, n_clocks=len(clocks),
                n_cells=len(cells), n_ff=n_ff,
                blackboxes=find_blackboxes(data))


def parse_netlist_structural(jfile: Path, top_name: str | None = None):
    """`netlist_graph.parse_netlist`, but with clocks found structurally.

    The stock parser treats a port as a clock only if its name starts with
    `clk`. Of the sixteen found corpus designs only the six OpenTitan ones
    satisfy that; `axis_async_fifo` uses `s_clk`/`m_clk`, `wb2axip_afifo` uses
    `i_rclk`/`i_wclk`, and every pulp module uses `src_clk_i`/`dst_clk_i`. Those
    ten designs therefore reported zero clocks and yielded no crossings at all,
    which silently restricted the held-out confirmation set to one vendor.

    This re-derives each flip-flop's domain from `find_clock_ports`, which
    identifies a clock as anything reaching a flip-flop clock pin.
    """
    import json as _json
    from crossexplain.netlist_graph import parse_netlist as _parse

    nl = _parse(jfile)
    data = _json.loads(Path(jfile).read_text())
    clocks, clock_bits, tname, top = find_clock_ports(data, top_name)
    if not clocks:
        return nl

    # per-cell clock-pin bit, matching the stock parser's cell ordering
    cells_json = top.get("cells", {})
    name_to_clkbit = {}
    for cname, cj in cells_json.items():
        if not _is_ff_type(cj["type"]):
            continue
        conns = cj.get("connections", {})
        for pin in CLK_PINS:
            if pin in conns:
                bits = [b for b in conns[pin] if isinstance(b, int)]
                if bits:
                    name_to_clkbit[cname] = bits[0]
                break

    domain = list(nl.domain)
    for idx in nl.ff_idx:
        cname = nl.cells[idx][0]
        b = name_to_clkbit.get(cname)
        domain[idx] = clock_bits.get(b, -1) if b is not None else -1
    nl.domain = domain
    nl.clocks = list(clocks)
    return nl
